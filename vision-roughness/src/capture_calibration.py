"""用摄像头采集棋盘格标定照片。

用法：
    python src/capture_calibration.py

操作：
    空格键：保存当前帧到 data/calibration/
    q 键：退出

建议：
    手持/固定标定板，从不同角度拍摄 10~20 张，覆盖整个视野和不同姿态。
"""
import sys
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, "src")
from camera_adapter import WebcamCamera

SAVE_DIR = Path("data/calibration")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def main(device_id: int = 0):
    print("启动标定图像采集...")
    print("按 [空格] 保存当前帧，按 [q] 退出")

    saved_count = len(list(SAVE_DIR.glob("calib_*.jpg")))

    with WebcamCamera(device_id=device_id) as cam:
        while True:
            frame = cam.capture()
            display = frame.copy()
            cv2.putText(display, f"Saved: {saved_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Calibration Capture - SPACE: save, Q: quit", display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = SAVE_DIR / f"calib_{timestamp}.jpg"
                cv2.imwrite(str(save_path), frame)
                saved_count += 1
                print(f"已保存: {save_path}")

    cv2.destroyAllWindows()
    print(f"\n共采集 {saved_count} 张标定照片，保存在 {SAVE_DIR}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="摄像头编号")
    args = ap.parse_args()

    main(device_id=args.device)
