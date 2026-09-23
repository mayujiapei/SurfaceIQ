r"""圆环零件图像分析：摸清外圈/中心孔/螺栓孔/背景的灰度分布，为 find_circles 定参数。

用法:
    python src\explore_ring.py <图片路径>
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def analyze(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"无法读取: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"图像: {w}x{h}, 灰度均值 {gray.mean():.1f}, min {gray.min()}, max {gray.max()}")

    # 1) Otsu 全图二值化，找最大轮廓 = 金属(可能和孔连片)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thr_val, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Otsu 阈值: {thr_val:.0f}")
    cnts, hier = cv2.findContours(otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise SystemExit("Otsu 没有轮廓")
    areas = np.array([cv2.contourArea(c) for c in cnts])
    outer_idx = int(np.argmax(areas))
    outer = cnts[outer_idx]
    x, y, bw, bh = cv2.boundingRect(outer)
    print(f"最大轮廓: 面积 {areas[outer_idx]:.0f}px², 包围盒 ({x},{y},{bw}x{bh}), 层级 {hier[0][outer_idx]}")

    (m_cx, m_cy), m_r = cv2.minEnclosingCircle(outer)
    print(f"外圈 minEnclosingCircle: 圆心 ({m_cx:.1f},{m_cy:.1f}) 直径 {2*m_r:.1f}px")
    (e_cx, e_cy), (e_ax, e_ax2), e_ang = cv2.fitEllipse(outer)
    print(f"外圈 fitEllipse: 圆心 ({e_cx:.1f},{e_cy:.1f}) 长短轴 {e_ax:.1f}x{e_ax2:.1f} 角度 {e_ang:.1f}°")

    # 2) 按半径环形分区统计灰度（了解中心孔/螺栓孔/金属各亮度）
    cy_, cx_ = e_cy, e_cx
    rr = np.sqrt((np.arange(w)[None, :] - cx_) ** 2 + (np.arange(h)[:, None] - cy_) ** 2)
    for low, high, label in [
        (0.00, 0.70, "中心孔区(0~70%R)"),
        (0.70, 0.78, "孔边缘过渡(70~78%R)"),
        (0.78, 0.92, "螺栓孔环带(78~92%R)"),
        (0.92, 1.00, "外圈边带(92~100%R)"),
    ]:
        ring_mask = (rr >= e_ax * low) & (rr < e_ax * high) & (rr < e_ax * 1.05)
        vals = gray[ring_mask]
        print(f"  {label}: 均值 {vals.mean():.1f}, "
              f"P5 {np.percentile(vals,5):.0f}, P50 {np.percentile(vals,50):.0f}, "
              f"P95 {np.percentile(vals,95):.0f}")

    # 3) 掩膜内候选孔：固定阈值 -> 黑色区域 = 孔
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(cx_), int(cy_)), int(e_ax * 0.99), 255, -1)
    for thr in (100, 128, 150):
        _, bin2 = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY_INV)  # 暗=孔
        bin2 = cv2.bitwise_and(bin2, bin2, mask=mask)
        cnts2, hier2 = cv2.findContours(bin2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        big = [c for c in cnts2 if cv2.contourArea(c) > 500]
        print(f"  阈值 {thr}: 暗区轮廓 {len(cnts2)} 个, 面积>500px 的 {len(big)} 个")
        big.sort(key=cv2.contourArea, reverse=True)
        for c in big[:8]:
            (ecx, ecy), (a1, a2), ang = cv2.fitEllipse(c)
            a = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            circ = 4 * np.pi * a / (perimeter * perimeter) if perimeter > 0 else 0
            print(f"    面积 {a:8.0f}  椭圆 {a1:.0f}x{a2:.0f} 圆度 {circ:.2f} "
                  f"圆心 ({ecx:.0f},{ecy:.0f}) 距中心 {np.hypot(ecx-cx_, ecy-cy_):.0f}px")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "../data/captures/capture_20260908_115557.jpg"
    analyze(Path(p))
