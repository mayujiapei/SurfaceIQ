"""尺寸测量主流程骨架：去畸变 → 边缘 → 拟合 → 换算 → 判公差"""
import cv2, numpy as np, pickle

calib = pickle.load(open("models/camera_calib.pkl", "rb"))
MM_PER_PIXEL = 0.02      # 由标定尺实测得到，填你的值

def measure_part(img_path, spec_mm, tol_mm):
    img = cv2.imread(str(img_path))
    if img is None:
        return {"result": "NG", "reason": "无法读取图片"}

    img = cv2.undistort(img, calib["K"], calib["dist"])     # 1. 去畸变
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. 边缘检测（现场光照固定后用固定阈值；不稳定用自适应）
    edges = cv2.Canny(gray, 50, 150)

    # 3. 找轮廓，取最大轮廓做最小外接矩形（测长宽场景）
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"result": "NG", "reason": "未检测到轮廓"}

    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    w_px, h_px = rect[1]

    # 4. 像素 → 毫米，对比公差
    w_mm, h_mm = w_px * MM_PER_PIXEL, h_px * MM_PER_PIXEL
    ok = abs(w_mm - spec_mm) <= tol_mm
    return {"width_mm": round(w_mm, 3), "height_mm": round(h_mm, 3),
            "result": "OK" if ok else "NG"}
