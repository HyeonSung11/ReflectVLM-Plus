#!/bin/bash
# 환경 변수 설정
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=1  # 에러 방지를 위해 확실히 1장만 사용
# 인자 확인
if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 {sim|diffusion}"
    exit 1
fi
dynamics_model=$1
exp_name="new_train_reflect_vlm_${dynamics_model}_debug"
# --- Training Arguments (4090 단일 GPU / QLoRA 최적화) ---
# DeepSpeed를 쓰지 않으므로 설정을 단순화합니다.
train_args="--lora_enable True \
--lora_r 128 \
--lora_alpha 256 \
--mm_projector_lr 2e-5 \
--bits 4 \
--double_quant True \
--quant_type nf4 \
--bf16 True \
--fp16 False \
--tf32 True \
--version v1 \
--vision_tower openai/clip-vit-large-patch14-336 \
--mm_projector_type mlp2x_gelu \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 16 \
--evaluation_strategy no \
--save_strategy steps \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 1e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to wandb"
# 실행 파라미터 설정
model_path="yunhaif/ReflectVLM-llava-v1.5-13b-base"
if [ $dynamics_model == "diffusion" ]; then
    diffusion_args="--diffuser_pretrained_model=yunhaif/ReflectVLM-diffusion"
else
    diffusion_args=""
fi
# train.py 실행 (수정하신 대로 단일 프로세스로 동작)
python train.py \
    --seed=0 \
    --n_trajs=400 \
    --repeat_per_env=2 \
    --n_iterations=3 \
    --save_dir="datasets/$exp_name" \
    --train_args="$train_args" \
    --start_traj_id=0 \
    --start_board_id=0 \
    --logging.online=False \
    --logging.group=$exp_name \
    --logging.prefix='train' \
    --agent_type="llava" \
    --revise_action=True \
    --imagine_future_steps=5 \
    $diffusion_args \
    --level='hard' \
    --oracle_prob=0.5 \
    --model_path=$model_path \
    --record=True