"""单张采图（无窗口，适合脚本化连拍）。

用法（在项目根目录 vision-roughness/ 下运行）：
    python src/capture_single.py

输出：
    保存图片到 data/captures/capture_single_<时间戳>.jpg

相机型号与曝光全部由 src/config.py 决定（当前为海康工业相机），没有命令行参数。
用途：需要一次拍很多张时（例如重复装夹拍 10 张评估尺寸重复性），
      文件名带时间戳、不会互相覆盖。
"""
import sys
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, "src")
import config
from camera_adapter import create_camera

SAVE_DIR = Path("data/captures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"打开相机（{config.CAMERA_TYPE}）...")
    with create_camera(config.CAMERA_TYPE, **config.CAMERA_KWARGS) as cam:
        print("采图中...")
        frame = cam.capture()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = SAVE_DIR / f"capture_single_{timestamp}.jpg"
    cv2.imwrite(str(save_path), frame)
    print(f"已保存: {save_path}")
    return save_path


if __name__ == "__main__":
    import argparse

    # 无参数：相机与曝光统一由 src/config.py 决定（换相机只改那一处）。
    # 见 capture_demo.py 的说明：保留空 parser 是为了 -h 与拒绝过时参数。
    argparse.ArgumentParser(
        description="单张采图（相机由 src/config.py 决定）").parse_args()
    main()
