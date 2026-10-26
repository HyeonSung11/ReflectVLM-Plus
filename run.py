import os
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
import torch

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
    load_4bit=(False, "bool", "Load model in 4-bit quantization"),
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
    save_dir=("datasets/data_v9", "string", "Directory for saving data."),
    start_traj_id=(0, "integer", "Starting trajectory index"),
    start_board_id=(0, "integer", "Starting board index"),
    oracle_prob=(0.5, "float", "Probability of executing oracle action at each timestep"),
    record=(True, "bool", "Record video."),
    record_frame_skip=(5, "integer", "Skip between recorded frames."),
    logging=WandBLogger.get_default_config(),
)

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
            t_id = current_entry['trajectory_id']
            s_id = current_entry['step_id']
            if t_id in [8, 11]:
                print(f"[EVENT] Traj {t_id} Step {s_id}: Loop Purged ({current_act} -> {next_act})")
            print(f"--- [Purge] Loop detected: {current_act} -> {next_act}. Removing these steps.")
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

    # initialize agent
    agent = None
    if FLAGS.agent_type == "llava":
        from roboworld.agent.llava import LlavaAgent
        agent = LlavaAgent(model_path=FLAGS.model_path,
                           model_base=FLAGS.model_base,
                           load_4bit=FLAGS.load_4bit)

    board_cnt, traj_cnt, succ_cnt = 0, 0, 0
    data = []
    eval_stats = []
    traj_id = FLAGS.start_traj_id
    board_id = FLAGS.start_board_id
    
    while traj_id < FLAGS.start_traj_id + FLAGS.n_trajs:

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
            traj_dir = os.path.join(FLAGS.save_dir, str(traj_id))
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
            visual_warning_count = 0 
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

                inference_t0, rollout_t0, rollout_t = None, None, None
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
                                # --- [Step 1: Visual Stagnation Detection 로직 추가] ---
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
                                        if traj_id in [8, 11]:
                                            print(f"[EVENT] Traj {traj_id} Step {t}: Stagnation Detected (Type: Redundant)")
                                        warning_msg = f"Note: Visual analysis predicts that after step {stagnation_idx+1}, the environment will return to a state visually identical to the current one. This plan is redundant."
                                        visual_warning_count += 1
                                    else:
                                        # Check for score plateau (Intermediate stagnation)
                                        for i in range(len(sims)):
                                            for j in range(i + 1, len(sims)):
                                                if abs(sims[i] - sims[j]) < 0.005:
                                                    if traj_id in [8, 11]:
                                                        print(f"[EVENT] Traj {traj_id} Step {t}: Stagnation Detected (Type: Plateau)")
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
                        if traj_id in [8, 11] and agent_action != agent_action_revised:
                            print(f"[EVENT] Traj {traj_id} Step {t}: Replanning Occurred ({agent_action} -> {agent_action_revised})")
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
                    print(f"Error during step {t}: {e}")
                    if rollout_t is None and rollout_t0 is not None:
                        rollout_t = time.time() - rollout_t0
                    else:
                        rollout_t = 0
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
                    "image": f"{traj_id}/{i}.png",
                    "next_image": f"{traj_id}/{i + 1}.png",
                    "final_goal_image": f"{traj_id}/goal.png",
                    "traj_success": int(traj_succ),
                    "traj_total_steps": len(question_list),
                    "object_descriptions": copy.deepcopy(env_info['peg_descriptions']),
                    "object_dependencies": copy.deepcopy(env_info["dependencies"])
                }
                if FLAGS.revise_action:
                    entry["agent_action_revised"] = agent_act_revised_list[i]
                    entry["agent_plan"] = copy.deepcopy(agent_plan_list[i])

                temp_traj_entries.append(entry)

            # --- [Step 2: Loop Purging 적용] ---
            cleaned_entries, purge_count = purge_trajectory_loops(temp_traj_entries)
            data.extend(cleaned_entries)

            # 통계 기록
            eval_stats.append({
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

        env.close()
        env_seed += 1
        board_id += 1
        board_cnt += 1

    # save data
    df = pd.DataFrame(data)
    meta_path = os.path.join(FLAGS.save_dir, "meta.csv")
    df.to_csv(meta_path)

    # --- [Improvement Report 저장] ---
    stats_df = pd.DataFrame(eval_stats)
    stats_df.to_csv(os.path.join(FLAGS.save_dir, "eval_improvement_report.csv"), index=False)
    print(f"--- Evaluation Improvement Report Saved at {FLAGS.save_dir} ---")

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
