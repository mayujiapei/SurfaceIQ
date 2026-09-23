r"""外圈检测对比试验：Canny+轮廓 vs 缩图 HoughCircles。

结果用来决定 find_circles 的实现路线。
用法:
    python src\edge_ring.py <图片路径>
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def fit_circle_parts(idx, cnts, label):
    """打印轮廓的 minEnclosingCircle / fitEllipse。"""
    c = cnts[idx]
    (mcx, mcy), mr = cv2.minEnclosingCircle(c)
    (ecx, ecy), (a1, a2), ang = cv2.fitEllipse(c)
    area = cv2.contourArea(c)
    print(f"  {label}: 面积 {area:.0f} | minEnclosing: 圆心({mcx:.0f},{mcy:.0f}) 直径{2*mr:.0f}px"
          f" | fitEllipse: 圆心({ecx:.0f},{ecy:.0f}) {a1:.0f}x{a2:.0f} 角{ang:.0f}°")


def try_canny(gray_bgr: np.ndarray, name: str):
    """路线1: Canny -> 闭运算 -> 最大外轮廓。”"""
    print(f"\n===== 路线1: CANNY ({name}) =====")
    gray = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    c1 = int(thr * 0.4)
    c2 = int(thr * 1.2)
    edges = cv2.Canny(blur, c1, c2)
    print(f"  Canny({c1},{c2}), 边缘像素 {np.count_nonzero(edges)}")
    # 闭运算把断开的环连起来；膨胀一下把外圈变成闭合区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # 找白色连片的轮廓（把边缘环"填充"：直接找 closed 上最大轮廓的边界即可）
    cnts, hier = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = np.array([cv2.contourArea(c) for c in cnts])
    top = np.argsort(-areas)[:5]
    for i in top:
        fit_circle_parts(i, cnts, f"轮廓{i} area={areas[i]:.0f}")


def try_hough(gray_bgr: np.ndarray, name: str, target_w: int = 1200):
    """路线2: 缩图 HoughCircles，多档参数扫圆。返回(缩放比, 圆列表)。"""
    print(f"\n===== 路线2: HOUGH ({name}, 缩到宽{target_w}) =====")
    gray = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2GRAY)
    h0, w0 = gray.shape
    scale = target_w / w0
    small = cv2.resize(gray, (target_w, int(h0 * scale)), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (9, 9), 0)
    print(f"  缩图 {small.shape[1]}x{small.shape[0]}, 缩放比 {scale:.4f}")

    found = []
    for param2 in (30, 38, 46):
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
            param1=80, param2=param2, minRadius=15, maxRadius=int(target_w * 0.6),
        )
        n = 0 if circles is None else len(circles[0])
        print(f"  param2={param2}: {n} 个圆")
        if circles is not None:
            c = np.round(circles[0]).astype(int)
            srt = np.argsort(-c[:, 2])
            for i in srt[:12]:
                x, y, r = c[i]
                cx, cy, rr = x / scale, y / scale, r / scale
                print(f"    小图({x},{y},{r}) -> 原图({cx:.0f},{cy:.0f}) 直径{2*rr:.0f}px")
                found.append((cx, cy, 2 * rr))
    return found


if __name__ == "__main__":
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/captures/capture_20260908_115557.jpg")
    img = cv2.imread(str(p))
    if img is None:
        raise SystemExit(f"无法读取: {p}")
    try_canny(img, p.stem)
    try_hough(img, p.stem)
