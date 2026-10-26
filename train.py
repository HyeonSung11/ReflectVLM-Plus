import os
import sys
import time
import pandas as pd
from PIL import Image
import wandb
from absl import app, flags
import random
import uuid
import copy
import numpy as np
import imageio
import mujoco
import json
import subprocess
import torch
import gc

from roboworld.utils.config import define_flags, get_user_flags
from roboworld.utils.logger import WandBLogger, set_random_seed
from roboworld.envs.generator import generate_xml
from roboworld.envs.asset_path_utils import full_path_for
from roboworld.envs.mujoco.franka.franka_assembly import (
    FrankaAssemblyEnv, AssemblyOracle
)
from roboworld.agent.oracle import OracleAgent
from roboworld.agent.utils import parse_act_txt, get_prompt

FLAGS = flags.FLAGS

FLAGS_DEF = define_flags(
    agent_type=("expert", "string", "Type of agent (llava, expert, random)"),
    camera_name=("table_back", "string", "Camera name."),
    model_path=("/path/to/model", "string", "Path to model"),
    model_base=(None, "string", "Path to base model"),
    revise_action=(False, "bool", "Revise an initially proposed action"),
    dummy_revised_action=(False, "bool", "If True, use original action as revised action. "
                          "This is for data collection when the agent is not trained to reflect yet "
                          "(usually the first iteration of DAgger)."),
    imagine_future_steps=(0, "integer", "Generate/simulate future image."),
    diffuser_pretrained_model=(None, "string", "Path to pretrained diffuser"),
    diffuser_unet_dir=(None, "string", "Path to unet model of diffuser"),
    diffuser_vae_dir=(None, "string", "Path to vae model of diffuser"),
    level=("all", "string", "Level of difficulty (medium, hard, or all)"),
    seed=(42, "integer", "Random seed."),
    reset_seed_start=(0, "integer", "first seed to reset env"),
    max_steps=(50, "integer", "Max number of decision steps in a trajectory"),
    n_trajs=(100, "integer", "Number of trajectories."),
    repeat_per_env=(1, "integer", "Number of trajectories for each env/board."),
    n_iterations=(1, "integer", "Number of post-training iterations."),
    save_dir=("datasets/data_v9", "string", "Directory for saving data."),
    train_args=("", "string", "Arguments to pass to train_vlm.py."),
    start_traj_id=(0, "integer", "Starting trajectory index"),
    start_board_id=(0, "integer", "Starting board index"),
    oracle_prob=(0.5, "float", "Probability of executing oracle action at each timestep"),
    record=(True, "bool", "Record video."),
    record_frame_skip=(5, "integer", "Skip between recorded frames."),
    logging=WandBLogger.get_default_config(),
)


def imagine_with_sim(env, agent, first_action, goal_img, history, obj_labels, traj_dir, t):
    env_recording = env.is_recording
    if env_recording:
        env._record = False
    _env_state = env.__getstate__()
    _plan = [first_action]
    _imgs = []
    for i in range(FLAGS.imagine_future_steps):
        try:
            err = env.act_txt(_plan[-1])
            _img = env.read_pixels(camera_name=FLAGS.camera_name)
            _imgs.append(_img)
            Image.fromarray(_img).save(
                os.path.join(traj_dir, f"sim-{t}-{'-'.join(_plan).replace(' ', '_')}.png")
            )
        except Exception as e:
            print(f"Error during imagining into future: {e}")
            break

        if i + 1 == FLAGS.imagine_future_steps or env.is_success():
            break

        _a = agent.act(_img, goal_img, inp=get_prompt(
            version="propose", history=history + _plan, obj_labels=obj_labels
        ))
        if _a == "done":
            break
        _plan.append(_a)
    
    env.__setstate__(_env_state)
    if env_recording:
        env._record = True

    return _plan, _imgs


def imagine_with_diffusion(diffusion_sim, agent, first_action, img, goal_img, history, obj_labels, traj_dir, t):
    _plan = [first_action]
    _img = img
    _imgs = []
    for i in range(FLAGS.imagine_future_steps):
        try:
            next_img = diffusion_sim.forward(curr_image=_img, act_text=_plan[-1])
            _imgs.append(next_img)
            Image.fromarray(next_img).save(os.path.join(traj_dir, f"gen-{t}-{'-'.join(_plan).replace(' ', '_')}.png"))
        except Exception as e:
            print(f"Error during imagining into future with diffusion: {e}")
            break

        _img = next_img
        if i + 1 == FLAGS.imagine_future_steps:
            break
        
        _a = agent.act(_img, goal_img, inp=get_prompt(
            version="propose", history=history + _plan, obj_labels=obj_labels
        ))
        if _a == "done":
            break

        _plan.append(_a)

    return _plan, _imgs


def build_env(env_seed, xml_filename, render_mode="offscreen"):
    xml, info = generate_xml(seed=env_seed)
    if (FLAGS.level == 'medium' and info['n_bodies'] > 5) or (FLAGS.level == 'hard' and info['n_bodies'] <= 5):
        return None, None
    xml.write_to_file(filename=xml_filename)

    board_name = "brick_1"
    fixture_name = None  # "fixture"
    peg_ids = [j + 1 for j in range(1, info['n_bodies'])]
    peg_names = [f"brick_{j+1}" for j in range(1, info['n_bodies'])]
    peg_descriptions = [info['brick_descriptions'][peg_name] for peg_name in peg_names]

    peg_labels = [" ".join(pd.split()[:1]) for pd in peg_descriptions]
    peg_labels_shuffled = peg_labels.copy()
    random.shuffle(peg_labels_shuffled)

    env = FrankaAssemblyEnv(board_name=board_name, fixture_name=fixture_name, peg_names=peg_names, peg_descriptions=peg_descriptions,
                            render_mode=render_mode, frame_skip=20, model_name=xml_filename, max_episode_length=50000,
                            magic_attaching=True)
    env_info = {
        "peg_ids": peg_ids,
        "peg_names": peg_names,
        "peg_descriptions": peg_descriptions,
        "peg_labels": peg_labels,
        "peg_labels_shuffled": peg_labels_shuffled,
        "dependencies": info["dependencies"]
    }
    
    return env, env_info

def purge_trajectory_loops(traj_entries):
    """
    궤적 데이터에서 Case 1, 2, 3에 해당하는 역행동 루프를 찾아 제거합니다.
    """
    if not traj_entries:
        return [], 0
    
    cleaned_entries = []
    purge_count = 0
    i = 0
    while i < len(traj_entries):
        current_entry = traj_entries[i]
        current_act = current_entry.get('action_description', "")
        
        # 마지막 스텝이거나 'done' 또는 파싱 불가능하면 통과
        if not current_act or current_act == 'done' or i + 1 >= len(traj_entries):
            cleaned_entries.append(current_entry)
            i += 1
            continue
            
        next_entry = traj_entries[i+1]
        next_act = next_entry.get('action_description', "")
        
        try:
            curr_p, curr_obj = parse_act_txt(current_act)
            next_p, next_obj = parse_act_txt(next_act)
        except:
            cleaned_entries.append(current_entry)
            i += 1
            continue
        
        is_loop = False
        # 같은 물체에 대해 역행동이 일어났는지 확인
        if curr_obj == next_obj and curr_obj is not None:
            # Case 1 & 2: 집었다가 바로 제자리에 놓기/끼우기 (Table/Board 루프)
            if curr_p == "pick up" and next_p in ["put down", "insert"]:
                is_loop = True
            # Case 3: 놓거나 끼운 직후에 다시 집기 (Manipulation 루프)
            elif curr_p in ["put down", "insert"] and next_p == "pick up":
                is_loop = True
        
        if is_loop:
            print(f"--- [Purge] Loop detected in traj {current_entry['trajectory_id']}: {current_act} -> {next_act}. Removing these steps.")
            i += 2 # 루프인 두 스텝을 건너뜀 (제거)
            purge_count += 1
        else:
            cleaned_entries.append(current_entry)
            i += 1
            
    # 제거된 스텝만큼 step_id 및 history 재조정
    new_history = []
    for idx, entry in enumerate(cleaned_entries):
        entry['step_id'] = idx
        entry['history'] = copy.deepcopy(new_history)
        new_history.append(entry['action_description'])
        
    return cleaned_entries, purge_count

def prepare_vlm_json(data_list, output_path, image_folder):
    vlm_dataset = []

    for idx, entry in enumerate(data_list):
        traj_id = entry['trajectory_id']
        step_id = entry['step_id']
        expert_action = entry['oracle_action']
        
        # object_descriptions가 리스트이므로 바로 join 합니다.
        if isinstance(entry['object_descriptions'], list):
            colors_str = ", ".join(entry['object_descriptions'])
        else:
            # 혹시나 딕셔너리일 경우를 대비한 안전 코드
            colors_str = ", ".join(entry['object_descriptions'].keys())
            
        history_str = ", ".join(entry['history']) if entry['history'] else "none"

        # 1. Action Proposal
        proposal_prompt = (
            "There is a puzzle consisting of a board and several pieces with different colors on the table. "
            "The goal is to assemble the puzzle with the robot arm. In each step, one of the following four actions "
            "can be taken: pick up [obj], put down [obj], reorient [obj], and insert [obj], where [obj] refers to the "
            f"piece to be manipulated. The image of the goal state is: <image>. The image of the current state is: <image>. "
            f"The most recently executed actions are: {history_str}. What action should be taken next? "
            f"Note that [obj] should be a color chosen from the following list: {colors_str}."
        )

        proposal_sample = {
            "id": f"prop_{traj_id}_{step_id}",
            "image": [
                entry['final_goal_image'], # 첫 번째 <image>
                entry['image']             # 두 번째 <image>
            ],
            "conversations": [
                {"from": "human", "value": proposal_prompt},
                {"from": "gpt", "value": expert_action}
            ]
        }
        vlm_dataset.append(proposal_sample)

        # 2. Reflection
        reflection_image_path = None
        reflection_plan = entry.get("agent_plan", [])
        
        # Check if the explicit path (e.g. from relabeling) works
        if "imagined_future_image" in entry and os.path.exists(os.path.join(image_folder, entry["imagined_future_image"])):
            reflection_image_path = entry["imagined_future_image"]
        
        # If not, try to find the simulation image by searching backwards (for simulation cases)
        elif reflection_plan:
             for i in range(len(reflection_plan), 0, -1):
                temp_plan = reflection_plan[:i]
                plan_str = "-".join(temp_plan).replace(' ', '_')
                
                # Try both 'sim-' and 'gen-' prefixes
                found = False
                for prefix in ["sim", "gen"]:
                    relative_path = f"{traj_id}/{prefix}-{step_id}-{plan_str}.png"
                    if os.path.exists(os.path.join(image_folder, relative_path)):
                        reflection_image_path = relative_path
                        found = True
                        break
                if found:
                    break

        if reflection_image_path and reflection_plan:
            init_plan_str = ", ".join(reflection_plan)
            num_steps = len(reflection_plan)

            reflection_prompt = (
                "There is a puzzle consisting of a board and several pieces with different colors on the table. "
                "The goal is to assemble the puzzle with the robot arm. In each step, one of the following four actions "
                "can be taken: pick up [obj], put down [obj], reorient [obj], and insert [obj], where [obj] refers to the "
                f"piece to be manipulated. The image of the goal state is: <image>. The image of the current state is: <image>. "
                f"The most recently executed actions are: {history_str}. The next {num_steps} steps planned by the model is "
                f"{init_plan_str}, from which we are going to only execute the first action. Note that if the full plan "
                "was executed sequentially, the future state would be: <image>. What action should be taken for the "
                f"immediate next step? Note that [obj] should be a color chosen from the following list: {colors_str}. "
                "You can modify the initial plan if it leads to an undesired future state."
            )

            reflection_sample = {
                "id": f"refl_{traj_id}_{step_id}",
                "image": [
                    entry['final_goal_image'],      # 첫 번째 <image>
                    entry['image'],                  # 두 번째 <image>
                    reflection_image_path            # 세 번째 <image>
                ],
                "conversations": [
                    {"from": "human", "value": reflection_prompt},
                    {"from": "gpt", "value": expert_action}
                ]
            }
            vlm_dataset.append(reflection_sample)
        elif "agent_plan" in entry:
            # 이미지를 하나도 찾지 못했다면 경고를 띄우고 건너뜁니다.
            print(f"Warning: No valid future image for traj {traj_id} step {step_id}. Skipping reflection sample.")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vlm_dataset, f, indent=2, ensure_ascii=False)


def main(_):
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    set_random_seed(FLAGS.seed)
    env_seed = FLAGS.seed
    render_mode = 'offscreen'
    candidate_act_list = ["pick up", "put down", "insert", "reorient"]
    xml_filename = full_path_for(f"tmp_{uuid.uuid4()}.xml")

    # check flags
    assert FLAGS.agent_type in {"llava", "expert", "random"}, f"Unknown agent type `{FLAGS.agent_type}`"
    assert FLAGS.level in {"medium", "hard", "all"}, f"Unknown assembly level `{FLAGS.level}`"
    if FLAGS.revise_action:
        from roboworld.agent.diffuser import DiffusionSim
        assert FLAGS.agent_type == "llava"
        if FLAGS.diffuser_pretrained_model is not None:
            diffusion_sim = DiffusionSim(
                pretrained_model=FLAGS.diffuser_pretrained_model, 
                unet_dir=FLAGS.diffuser_unet_dir, 
                vae_dir=FLAGS.diffuser_vae_dir, 
            )
        else:
            diffusion_sim = None

    board_cnt, traj_cnt, succ_cnt = 0, 0, 0
    data = []
    total_post_training_data = []
    traj_id = FLAGS.start_traj_id
    board_id = FLAGS.start_board_id
    
    for k in range(FLAGS.n_iterations):
        if k == 0:
            current_model_path = FLAGS.model_path
        else:
            current_model_path = os.path.join(FLAGS.save_dir, f"checkpoint-iter-{k-1}")
        
        if FLAGS.agent_type == "llava":
            from roboworld.agent.llava import LlavaAgent
            agent = LlavaAgent(model_path=current_model_path, model_base=FLAGS.model_base)

        current_iter_data = []
        iter_stats = [] # 이번 이터레이션 통계 저장용 리스트
        iter_traj_start = traj_id

        while traj_id < iter_traj_start + FLAGS.n_trajs:
            env, env_info = build_env(env_seed, xml_filename, render_mode)
            if env is None:
                env_seed += 1
                continue
            oracle = AssemblyOracle(env=env, brick_ids=env_info["peg_ids"], 
                                    brick_descriptions=env_info["peg_descriptions"], 
                                    dependencies=env_info["dependencies"])
            oracle_agent = OracleAgent(oracle)

            idx_in_env = 0
            reset_seed = FLAGS.reset_seed_start
            while idx_in_env < FLAGS.repeat_per_env:
                print(">" * 10, f"Trajectory {traj_id}", ">" * 10)
                iter_dir = os.path.join(FLAGS.save_dir, f"iter_{k}") # iter_k 폴더 경로 생성
                traj_dir = os.path.join(iter_dir, str(traj_id))      # save_dir/iter_k/traj_id 구조
                os.makedirs(traj_dir, exist_ok=True)
                reset_seed += 1
                env.reset(seed=reset_seed)
                goal_img = env.goal_images.get(FLAGS.camera_name, None)
                assert goal_img is not None

                # initialize data
                traj_succ = False
                exec_act_list = [None] * FLAGS.max_steps
                oracle_act_list = [None] * FLAGS.max_steps
                agent_act_list = [None] * FLAGS.max_steps
                agent_plan_list = [None] * FLAGS.max_steps
                agent_act_revised_list = [None] * FLAGS.max_steps
                traj_key_frames = []
                question_list, answer_list = [], []
                history = []
                history_list = []
                visual_warning_count = 0 # Step 1 카운터 초기화
                total_time, inference_time, rollout_time = 0., 0., 0.

                if FLAGS.record:
                    env.record_on(record_frame_skip=FLAGS.record_frame_skip)
                    env.render()

                # start rollout
                total_time_t0 = time.time()
                for t in range(FLAGS.max_steps):
                    print(f"[Step {t}]")
                    oracle_action = oracle_agent.act()
                    oracle_act_list[t] = oracle_action
                    oracle_action_primitive, oracle_brick_color = parse_act_txt(oracle_action)
                    print("Oracle action:", oracle_action)
                    
                    img = env.read_pixels(camera_name=FLAGS.camera_name)
                    traj_key_frames.append(img)

                    inp = get_prompt(version="propose", history=history, obj_labels=env_info["peg_labels_shuffled"])
                    print("Q:", inp)
                    question_list.append(inp)
                    history_list.append(copy.deepcopy(history))

                    inference_t0, rollout_t0, rollout_t = None, time.time(), None
                    agent_action_revised = None

                    try:
                        # get action with agent
                        if FLAGS.agent_type == "expert":
                            agent_action = oracle_action
                        elif FLAGS.agent_type == "random":
                            agent_action = f"{np.random.choice(candidate_act_list)} {np.random.choice(env_info['peg_labels'])}"
                        else:
                            inference_t0 = time.time()
                            agent_action = agent.act(img, goal_img, inp)

                            # reflect and revise action
                            if FLAGS.revise_action:                            
                                assert FLAGS.imagine_future_steps > 0

                                if diffusion_sim is None:
                                    _plan, _all_imgs = imagine_with_sim(
                                        env=env, agent=agent, first_action=agent_action, goal_img=goal_img, history=history, 
                                        obj_labels=env_info["peg_labels_shuffled"], traj_dir=traj_dir, t=t
                                    )
                                else:
                                    _plan, _all_imgs = imagine_with_diffusion(
                                        diffusion_sim=diffusion_sim, agent=agent, first_action=agent_action, img=img, goal_img=goal_img, 
                                        history=history, obj_labels=env_info["peg_labels_shuffled"], traj_dir=traj_dir, t=t
                                    )
                                agent_plan_list[t] = copy.deepcopy(_plan)
                                future_img = _all_imgs[-1] if _all_imgs else img

                                if FLAGS.dummy_revised_action:
                                    agent_action_revised = agent_action
                                else:
                                    # --- [Visual Stagnation Detection] ---
                                    warning_msg = None
                                    if _all_imgs:
                                        curr_emb = agent.get_visual_embedding(img)
                                        future_embs = [agent.get_visual_embedding(f_img) for f_img in _all_imgs]
                                        
                                        # Calculate similarities with current image
                                        sims = [torch.nn.functional.cosine_similarity(curr_emb, f_emb).item() for f_emb in future_embs]
                                        
                                        stagnation_idx = -1
                                        for idx, s in enumerate(sims):
                                            if s > 0.98: # Threshold for stagnation
                                                stagnation_idx = idx
                                                break
                                        
                                        if stagnation_idx != -1:
                                            warning_msg = f"Note: Visual analysis predicts that after step {stagnation_idx+1}, the environment will return to a state visually identical to the current one. This plan is redundant."
                                            visual_warning_count += 1
                                        else:
                                            # Check for score plateau (Intermediate stagnation)
                                            for i in range(len(sims)):
                                                for j in range(i + 1, len(sims)):
                                                    if abs(sims[i] - sims[j]) < 0.005:
                                                        warning_msg = f"Note: Steps {i+1} through {j+1} show no visual progress. Your actions in this interval are ineffective."
                                                        visual_warning_count += 1
                                                        break
                                                if warning_msg: break
                                    
                                    inp2 = get_prompt(version="reflect", history=history, obj_labels=env_info["peg_labels_shuffled"], 
                                                      initial_plan=_plan, warning_msg=warning_msg)
                                    print("Q2:", inp2)
                                    agent_action_revised = agent.act(img, goal_img, inp2, next_image=future_img)

                                print("*" * 20, f"Step {t} reflection summary", "*" * 20)
                                print(f"Initial plan: {_plan}\n=> Revised action: {agent_action_revised}")
                                print("*" * 20, f"End of reflection summary", "*" * 20)

                            inference_time += (time.time() - inference_t0)

                        agent_act_list[t] = agent_action
                        print("A:", agent_action)

                        # process revised action
                        assert FLAGS.revise_action == (agent_action_revised is not None)
                        if agent_action_revised is not None:
                            try:
                                assert len(agent_action_revised.strip().split(' ')) <= 3, "Bad output format."
                                _p, _o = parse_act_txt(agent_action_revised)
                                assert _p in candidate_act_list and _o in env_info["peg_labels"], "Bad output format."
                            except Exception as e:
                                print(f"Error during parsing `agent_action_revised`({agent_action_revised}): {e}")
                                agent_action_revised = agent_action
                            agent_act_revised_list[t] = agent_action_revised
                            agent_action = agent_action_revised # Use the revised action as the final action to execute
                        print("A2:", agent_action)
                        
                        # parse and choose action
                        agent_action_primitive, agent_brick_color = parse_act_txt(agent_action)
                        if random.uniform(0, 1) < FLAGS.oracle_prob:
                            exec_action_primitive = oracle_action_primitive
                            exec_brick_color = oracle_brick_color
                        else:
                            exec_action_primitive = agent_action_primitive
                            exec_brick_color = agent_brick_color
                        exec_action = "done" if exec_action_primitive == "done" else " ".join([exec_action_primitive, exec_brick_color])
                        exec_act_list[t] = exec_action
                        
                        # add action to history
                        history.append(exec_action)

                        # execute
                        print("Executed action:", exec_action)
                        if exec_brick_color not in env.peg_colors:
                            print(f"Unknown object '{exec_brick_color}'")
                            continue
                        if exec_action == "done":
                            rollout_t = 0
                            break
                        else:
                            rollout_t0 = time.time()
                            err = env.act_txt(exec_action)
                            if err != 0:
                                print(f"Error {err} when executing `{exec_action}`")
                            rollout_t = time.time() - rollout_t0

                    except Exception as e:
                        print(e)
                        if rollout_t is None:
                            rollout_t = time.time() - rollout_t0
                        break  # fail

                    rollout_time += rollout_t

                    # check success
                    if env.is_success():
                        traj_succ = True
                        succ_cnt += 1
                        break
                
                last_img = env.read_pixels(camera_name=FLAGS.camera_name)
                total_time = time.time() - total_time_t0

                if env.is_recording:
                    env.record_off()

                traj_key_frames.append(last_img)
                last_img_path = os.path.join(traj_dir, f"{len(question_list)}.png")
                Image.fromarray(last_img).save(last_img_path)
                goal_img_path = os.path.join(traj_dir, f"goal.png")
                Image.fromarray(goal_img).save(goal_img_path)

                # log data
                print("Success:", traj_succ)
                print(f"Inference time: {inference_time}, Rollout time: {rollout_time}")
                wandb_logger.log({
                    "success": int(traj_succ), 
                    "accumulated_success_cnt": succ_cnt,
                    "accumulated_success_rate": succ_cnt / (traj_cnt + 1),
                    "total_inference_time": inference_time,
                    "total_rollout_time": rollout_time,
                    "average_inference_time": inference_time / len(question_list),
                    "average_rollout_time": rollout_time / len(question_list),
                    "total_steps": len(question_list),
                    "total_time": total_time
                }, step=traj_cnt)

                if FLAGS.record:
                    wandb_logger.log_video(
                        np.stack(env.frames).transpose((0, 3, 1, 2)), fps=60,
                        caption=f"traj{traj_id}_seed{env_seed}_{reset_seed}_{['fail', 'succ'][traj_succ]}"
                    )

                temp_traj_entries = []
                for i, question in enumerate(question_list):
                    img_path = os.path.join(traj_dir, f"{i}.png")
                    Image.fromarray(traj_key_frames[i]).save(img_path)

                    entry = {
                        "trajectory_id": traj_id,
                        "board_id": board_id,
                        "env_seed": env_seed,
                        "reset_seed": reset_seed,
                        "step_id": i,
                        "action_description": exec_act_list[i],
                        "oracle_action": oracle_act_list[i],
                        "agent_action": agent_act_list[i],
                        "history": history_list[i],
                        "image": f"iter_{k}/{traj_id}/{i}.png",
                        "next_image": f"iter_{k}/{traj_id}/{i + 1}.png",
                        "final_goal_image": f"iter_{k}/{traj_id}/goal.png",
                        "traj_success": int(traj_succ),
                        "traj_total_steps": len(question_list),
                        "object_descriptions": copy.deepcopy(env_info['peg_descriptions']),
                        "object_dependencies": copy.deepcopy(env_info["dependencies"])
                    }
                    if FLAGS.revise_action:
                        entry["agent_action_revised"] = agent_act_revised_list[i]
                        entry["agent_plan"] = copy.deepcopy(agent_plan_list[i])
                        if isinstance(agent_plan_list[i], (list, tuple)):
                            _plan_str = "-".join(agent_plan_list[i]).replace(' ', '_')
                        else:
                            _plan_str = str(agent_plan_list[i]).replace(' ', '_')
                        prefix = "gen" if diffusion_sim is not None else "sim"
                        entry["imagined_future_image"] = f"iter_{k}/{traj_id}/{prefix}-{i}-{_plan_str}.png"

                    temp_traj_entries.append(entry)

                # --- [Step 2: Post-hoc Purge Logic] ---
                cleaned_entries, purge_count = purge_trajectory_loops(temp_traj_entries)
                current_iter_data.extend(cleaned_entries)
                data.extend(cleaned_entries)

                # 통계 기록
                iter_stats.append({
                    "Traj_ID": traj_id,
                    "Visual_Warnings": visual_warning_count,
                    "Purged_Loops": purge_count,
                    "Original_Steps": len(temp_traj_entries),
                    "Final_Steps": len(cleaned_entries),
                    "Success": int(traj_succ)
                })

                idx_in_env += 1
                traj_id += 1
                traj_cnt += 1

                # --- [Relabeling Logic] ---
                H = FLAGS.imagine_future_steps if FLAGS.imagine_future_steps > 0 else 3
                traj_len = len(cleaned_entries) # 정제된 데이터 길이 사용
                start_idx = len(data) - traj_len
                for t in range(traj_len):
                    entry = data[start_idx + t]
                    if "imagined_future_image" not in entry:
                        future_t = t + H
                        if future_t < traj_len:
                            future_img_path = f"iter_{k}/{traj_id-1}/{future_t}.png"
                            future_plan = []
                            valid_plan = True
                            for k_step in range(H):
                                act = data[start_idx + t + k_step]['action_description']
                                if act == 'done' or not act:
                                    valid_plan = False
                                    break
                                future_plan.append(act)
                            if valid_plan:
                                entry["imagined_future_image"] = future_img_path
                                entry["agent_plan"] = future_plan

            env_seed += 1
            board_id += 1
            board_cnt += 1

        total_post_training_data.extend(current_iter_data)
        iter_df = pd.DataFrame(current_iter_data)
        iter_df.to_csv(os.path.join(iter_dir, f"meta_iter_{k}.csv"))
        
        # --- [Improvement Report 저장] ---
        stats_df = pd.DataFrame(iter_stats)
        stats_df.to_csv(os.path.join(iter_dir, f"improvement_report_iter_{k}.csv"), index=False)
        print(f"--- Iteration {k} Improvement Report Saved. ---")
        
        env.close() # Close env here, after all trajectories for the iteration are done
        
        # --- Memory Cleanup Before Training ---
        if 'agent' in locals():
            print("Cleaning up agent...")
            del agent
        if 'diffusion_sim' in locals() and diffusion_sim is not None:
            print("Cleaning up diffusion simulator...")
            del diffusion_sim
        gc.collect()
        torch.cuda.empty_cache()
        print("VRAM cleanup complete.")
        # --------------------------------------

        print(f"--- Iteration {k}: Starting Fine-tuning ---")
        training_json_path = os.path.join(iter_dir, f"train_data_iter_{k}.json")
        prepare_vlm_json(total_post_training_data, training_json_path, FLAGS.save_dir)
        output_dir = os.path.join(FLAGS.save_dir, f"checkpoint-iter-{k}")
        
        # Check GPU count for distributed training
        try:
             num_gpus = torch.cuda.device_count()
        except ImportError:
             num_gpus = 1

        #if num_gpus > 1:
            # Use sys.executable -m torch.distributed.run for multi-GPU training
        #    train_cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(num_gpus), "--master_port", str(random.randint(10000, 60000)), "train_vlm.py"]
        #else:
        train_cmd = [sys.executable, "train_vlm.py"]

        # Add arguments from FLAGS.train_args
        if FLAGS.train_args:
            train_cmd.extend(FLAGS.train_args.split())
        
        # Add dynamic arguments (overriding if necessary)
        train_cmd.extend([
            "--model_name_or_path", current_model_path,
            "--data_path", training_json_path,
            "--image_folder", FLAGS.save_dir,
            "--output_dir", output_dir
        ])

        # Add DeepSpeed config if multiple GPUs are used
        if num_gpus > 1:
             # Ensure the config file path is correct relative to the execution directory
             train_cmd.extend(["--deepspeed", "config/zero2.json"])

        try:
            print(f"Running command: {' '.join(train_cmd)}")
            subprocess.run(train_cmd, check=True)
            print(f"--- Iteration {k}: Fine-tuning Completed. ---")
        except subprocess.CalledProcessError as e:
            print(f"Training failed at iteration {k}: {e}")
            break

    # save data
    df = pd.DataFrame(data)
    meta_path = os.path.join(FLAGS.save_dir, "meta.csv")
    df.to_csv(meta_path)

    # log summary
    print("=" * 20)
    print("Total # boards:", board_cnt)
    print("Total # trajectories:", traj_cnt)
    print("Total # samples(steps):", len(data))
    succ_rate = succ_cnt / traj_cnt
    print(f"Success rate: {succ_rate} ({succ_cnt}/{traj_cnt})")
    wandb_logger.log({"success_rate": succ_rate})

    os.remove(xml_filename)


if __name__ == "__main__":
    app.run(main)
