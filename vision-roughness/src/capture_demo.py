"""相机取景 + 手动采图（带预览窗口）。

用法（在项目根目录 vision-roughness/ 下运行）：
    python src/capture_demo.py

操作：
    空格键：保存当前帧到 data/captures/
    q 键：退出

相机型号与曝光全部由 src/config.py 决定（当前为海康工业相机），没有命令行参数。
用途：架相机时看实时画面、对焦、确认零件在视野中心；空格随手存图。
日常测量请直接用 测量.bat（相机现拍 → 结果窗口）。
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
    print("启动相机取景...")
    print("按 [空格] 保存当前帧，按 [q] 退出")

    with create_camera(config.CAMERA_TYPE, **config.CAMERA_KWARGS) as cam:
        win = "Camera - SPACE: save, Q: quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 960)   # 12MP 原图太大，窗口缩放到 1280x960
        while True:
            frame = cam.capture()
            cv2.imshow(win, frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = SAVE_DIR / f"capture_{timestamp}.jpg"
                cv2.imwrite(str(save_path), frame)
                print(f"已保存: {save_path}")

    cv2.destroyAllWindows()
    print("结束")


if __name__ == "__main__":
    import argparse

    # 无参数：相机与曝光统一由 src/config.py 决定（换相机只改那一处）。
    # 之所以保留这个空 parser —— 一是让 -h 能出说明，二是让过时的参数
    # （如以前那个只对笔记本摄像头有效的 --device）报错，而不是被静默忽略。
    argparse.ArgumentParser(
        description="相机取景 + 手动采图（相机由 src/config.py 决定）").parse_args()
    main()
