# Setup

Tested on Ubuntu 22.04, CUDA 12.1, Python 3.9, one NVIDIA RTX 4090 (24 GB) for training and RTX 3090 for evaluation.
The environment follows the original [ReflectVLM](https://github.com/yunhaif/reflect-vlm) instructions (see `docs/README_ReflectVLM.md`).

## 1. Environment

```bash
conda create -n reflectvlm python=3.9 -y
conda activate reflectvlm
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

If PyTorch and CUDA versions conflict, reinstall PyTorch for CUDA 12.1 and then flash-attn:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install "deepspeed>=0.14.3"
```

MuJoCo system libraries: see `mujoco.sh`.

## 2. Models

Downloaded automatically from Hugging Face on first run:

- `yunhaif/ReflectVLM-llava-v1.5-13b-base` (base VLM policy)
- `yunhaif/ReflectVLM-diffusion` (diffusion dynamics model)

The 13B model is loaded in 4-bit (QLoRA / bitsandbytes) so that it fits on a single 24 GB GPU. Weights that do not fit are offloaded to `offload/` (git-ignored).

## 3. Post-training with ReflectVLM+

```bash
bash scripts/train_reflect_vlm.sh diffusion   # or: sim
```

`train.py` collects 400 trajectories with the reflection loop, applies Visual Stagnation Detection during collection and Trajectory Loop Purging before fine-tuning, then fine-tunes the policy with LoRA (`llava/train/train.py`). Checkpoints are written under `datasets/<exp_name>/`.

## 4. Evaluation

Edit `model_path` in `scripts/eval_reflect_vlm.sh` to point to your checkpoint, then:

```bash
bash scripts/eval_reflect_vlm.sh diffusion   # or: sim
```

Results are written under `logs/<exp_name>/`.

## Where the ReflectVLM+ changes are

| File | Change |
|---|---|
| `run.py` | Visual Stagnation Detection in the reflection loop (CLIP cosine similarity, α = 0.98 single-step, β = 0.005 interval); 4-bit loading flag |
| `train.py` | **New.** Post-training driver: trajectory collection with reflection, `purge_trajectory_loops`, JSON export for LoRA fine-tuning |
| `roboworld/agent/llava.py` | `get_visual_embedding` (CLIP vision tower, mean-pooled, L2-normalized) |
| `roboworld/agent/utils.py` | Warning message injected into the reflection prompt |
| `llava/model/builder.py` | 4-bit quantized loading, skip LoRA merge under quantization |
| `scripts/train_reflect_vlm.sh`, `train_vlm.py`, `config/zero2.json` | Single-GPU QLoRA training script and config |
