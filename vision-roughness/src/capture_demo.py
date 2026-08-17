"""摄像头采图 + 推理 Demo。

用法（在项目根目录 vision-roughness/ 下运行）：
    python src/capture_demo.py

操作：
    空格键：保存当前帧并调用 ResNet18 推理
    q 键：退出

注意：
    本脚本使用电脑摄像头临时验证链路，预测结果仅用于演示，
    不代表生产环境下的粗糙度分类精度。
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, "src")
from camera_adapter import WebcamCamera
from infer import predict

SAVE_DIR = Path("data/captures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def main(device_id: int = 0, model_name: str = "resnet"):
    print("启动摄像头 Demo...")
    print("按 [空格] 拍照并推理，按 [q] 退出")

    with WebcamCamera(device_id=device_id) as cam:
        while True:
            frame = cam.capture()
            cv2.imshow("Camera Demo - SPACE: capture, Q: quit", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = SAVE_DIR / f"capture_{timestamp}.jpg"
                cv2.imwrite(str(save_path), frame)
                print(f"\n已保存: {save_path}")

                try:
                    cls, conf = predict(str(save_path), model_name=model_name)
                    print(f"预测结果: {cls}  置信度: {conf:.2%}")
                except Exception as e:
                    print(f"推理失败: {e}")

    cv2.destroyAllWindows()
    print("Demo 结束")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="摄像头编号，默认 0")
    ap.add_argument("--model", default="resnet", choices=["glcm", "resnet"], help="推理模型")
    args = ap.parse_args()

    main(device_id=args.device, model_name=args.model)
