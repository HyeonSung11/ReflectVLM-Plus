import sys
import os

# 현재 디렉토리를 path에 추가하여 llava 모듈을 찾을 수 있게 함
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 메모리 효율을 위해 Flash Attention 적용
# (GPU 환경이 아니거나 호환되지 않는 경우 경고가 뜰 수 있음)
try:
    from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
except ImportError:
    print("Flash Attention not found or failed to patch. Continuing without it.")
except Exception as e:
    print(f"Error applying Flash Attention patch: {e}. Continuing without it.")

from llava.train.train import train

if __name__ == "__main__":
    train()
