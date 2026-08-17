"""单张摄像头采图 + 推理（无窗口，适合快速测试）。

用法：
    python src/capture_single.py

输出：
    保存图片到 data/captures/capture_single_*.jpg
    打印预测类别和置信度
"""
import sys
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, "src")
from camera_adapter import WebcamCamera
from infer import predict

SAVE_DIR = Path("data/captures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def main(device_id: int = 0, model_name: str = "resnet"):
    print(f"打开摄像头 device_id={device_id}...")
    with WebcamCamera(device_id=device_id) as cam:
        print("采图中...")
        frame = cam.capture()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = SAVE_DIR / f"capture_single_{timestamp}.jpg"
    cv2.imwrite(str(save_path), frame)
    print(f"已保存: {save_path}")

    print("推理中...")
    cls, conf = predict(str(save_path), model_name=model_name)
    print(f"预测结果: {cls}  置信度: {conf:.2%}")
    return cls, conf


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="摄像头编号")
    ap.add_argument("--model", default="resnet", choices=["glcm", "resnet"])
    args = ap.parse_args()

    main(device_id=args.device, model_name=args.model)
