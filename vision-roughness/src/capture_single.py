"""单张采图（无窗口，适合脚本化连拍）。

用法（在项目根目录 vision-roughness/ 下运行）：
    python src/capture_single.py

输出：
    保存图片到 data/captures/capture_single_<时间戳>.jpg

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


def main(device_id: int = None):
    cam_kwargs = dict(config.CAMERA_KWARGS)
    if device_id is not None:
        cam_kwargs["device_id"] = device_id   # webcam 模式下可覆盖 config 中的编号

    print(f"打开相机（{config.CAMERA_TYPE}）...")
    with create_camera(config.CAMERA_TYPE, **cam_kwargs) as cam:
        print("采图中...")
        frame = cam.capture()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = SAVE_DIR / f"capture_single_{timestamp}.jpg"
    cv2.imwrite(str(save_path), frame)
    print(f"已保存: {save_path}")
    return save_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None,
                    help="摄像头编号（仅 webcam 模式有效，默认读 config.py）")
    args = ap.parse_args()

    main(device_id=args.device)
