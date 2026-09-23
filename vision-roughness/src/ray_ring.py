r"""射线剖面：从给定中心沿 8 方向找灰度梯度极值(亚像素)，输出每个结构边缘的距离。

用法:
    python src\ray_ring.py <图片路径> [cx] [cy]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

N_RAY = 8  # 8 方向


def subpix_edge(g, g0, g1):
    """在窗口内找梯度峰值，抛物线插值亚像素。g 是射线灰度序列。"""
    grad = np.gradient(g)  # 亚像素平滑后梯度
    edges = []
    for i in range(2, len(grad) - 2):
        if grad[i] > 3.5 and grad[i] >= grad[i - 1] and grad[i] >= grad[i + 1]:
            a, b, c = grad[i - 1], grad[i], grad[i + 1]
            denom = a - 2 * b + c
            delta = 0.5 * (a - c) / denom if denom != 0 else 0
            edges.append((i + delta, b))
        if grad[i] < -3.5 and grad[i] <= grad[i - 1] and grad[i] <= grad[i + 1]:
            a, b, c = grad[i - 1], grad[i], grad[i + 1]
            denom = a - 2 * b + c
            delta = 0.5 * (a - c) / denom if denom != 0 else 0
            edges.append((i + delta, b))
    return edges


def main(p: Path, cx0, cy0):
    img = cv2.imread(str(p))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 5)  # 大尺度平滑，滤掉金属表面纹理
    h, w = gray.shape
    print(f"中心 ({cx0},{cy0})  图像 {w}x{h}")
    max_r = int(np.hypot(max(cx0, w - cx0), max(cy0, h - cy0)))
    print(f"{'角度':>6} | 显著边缘 (距离px, 梯度强度, 灰值) | 方向 0°=右, 90°=下")
    for k in range(N_RAY):
        ang = np.deg2rad(k * 360 / N_RAY)
        dx, dy = np.cos(ang), np.sin(ang)
        d = np.arange(0, max_r, 0.5)  # 0.5px 步长
        xs = (cx0 + dx * d).astype(int).clip(0, w - 1)
        ys = (cy0 + dy * d).astype(int).clip(0, h - 1)
        g = gray[ys, xs].astype(float)
        edges = subpix_edge(g, -1, 12)
        # 去重：相隔<3px 的取最强
        uniq = []
        for dist, amp in edges:
            if uniq and dist - uniq[-1][0] < 4:
                if amp > uniq[-1][1]:
                    uniq[-1] = (dist, amp)
            else:
                uniq.append((dist, amp))
        txt = "  ".join(f"{d0*0.5:.0f}px(v{a:.0f},g{g[min(int(d0), len(g)-1)]:.0f})" for d0, a in uniq)
        print(f"{k*45:>6}° | {txt if txt else '-'}")


if __name__ == "__main__":
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/captures/capture_20260908_115557.jpg")
    cx = float(sys.argv[2]) if len(sys.argv) > 2 else 2032.0
    cy = float(sys.argv[3]) if len(sys.argv) > 3 else 1430.0
    main(p, cx, cy)
