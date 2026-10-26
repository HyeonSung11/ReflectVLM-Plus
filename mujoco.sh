#!/bin/bash

# Reflect-VLM MuJoCo GUI 문제 해결 스크립트
# 사용법: bash fix_mujoco_gui.sh

echo "=== Reflect-VLM MuJoCo GUI 문제 해결 스크립트 ==="
echo ""

# 1. 시스템 패키지 설치
echo "1. 시스템 패키지 설치 중..."
sudo apt-get update

echo "필수 OpenGL/MuJoCo 라이브러리 설치..."
sudo apt-get install -y libglfw3 libglew2.0 libgl1-mesa-glx libosmesa6-dev
sudo apt-get install -y libglfw3-dev libglew-dev patchelf
sudo apt-get install -y libgl1-mesa-dev xpra

echo "✓ 시스템 패키지 설치 완료"
echo ""

# 2. 환경 변수 설정
echo "2. 환경 변수 설정..."

# .bashrc에 환경 변수 추가
if ! grep -q "MUJOCO_GL" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# MuJoCo 렌더링 설정" >> ~/.bashrc
    echo "export MUJOCO_GL=egl" >> ~/.bashrc
    echo "export PYOPENGL_PLATFORM=egl" >> ~/.bashrc
    echo "✓ ~/.bashrc에 환경 변수 추가됨"
else
    echo "! 환경 변수가 이미 설정되어 있습니다"
fi

# 현재 세션에 환경 변수 설정
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

echo "✓ 환경 변수 설정 완료"
echo ""

# 3. MuJoCo 설치 확인
echo "3. MuJoCo 설치 확인..."
python3 -c "
import mujoco
print(f'MuJoCo 버전: {mujoco.__version__}')
print('✓ MuJoCo Python 패키지 확인됨')
" 2>/dev/null || echo "! MuJoCo Python 패키지가 설치되지 않았습니다"

echo ""

# 4. 간단한 테스트
echo "4. MuJoCo 렌더링 테스트..."
python3 -c "
import mujoco
import os

# 환경 변수 확인
print(f'MUJOCO_GL: {os.environ.get("MUJOCO_GL", "NOT SET")}')
print(f'PYOPENGL_PLATFORM: {os.environ.get("PYOPENGL_PLATFORM", "NOT SET")}')

# 간단한 모델 생성 및 테스트
try:
    model = mujoco.MjModel.from_xml_string('''
        <mujoco>
            <worldbody>
                <geom type="plane" size="1 1 0.1"/>
                <body pos="0 0 1">
                    <geom type="sphere" size="0.1"/>
                </body>
            </worldbody>
        </mujoco>
    ''')
    data = mujoco.MjData(model)
    print('✓ MuJoCo 모델 생성 성공')

    # 렌더링 테스트 (오프스크린)
    renderer = mujoco.Renderer(model, 400, 300)
    renderer.update_scene(data)
    pixels = renderer.render()
    print('✓ 오프스크린 렌더링 성공')

except Exception as e:
    print(f'! 렌더링 테스트 실패: {e}')
" 2>/dev/null || echo "! MuJoCo 테스트 실행 중 오류 발생"

echo ""
echo "=== 해결책 스크립트 완료 ==="
echo ""
echo "다음 단계:"
echo "1. 새 터미널을 열거나 'source ~/.bashrc' 실행"
echo "2. python scripts/interact.py 실행 테스트"
echo "3. VSCode에서 F5로 디버깅 테스트"
echo ""
echo "추가 도움이 필요하면 다음을 확인하세요:"
echo "- AMD GPU 사용 시 드라이버 다운그레이드 고려"
echo "- Wayland 환경에서는 X11로 전환 시도"
echo "- 헤드리스 환경에서는 MUJOCO_GL=osmesa 사용"
