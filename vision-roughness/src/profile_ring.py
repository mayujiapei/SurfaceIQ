r"""径向分析：定圆心 + 边缘距离直方图，找外圈/中心孔/螺栓孔环带的半径峰。

原理: 真实圆在"边缘点到圆心距离"直方图上形成尖峰; 噪声弧则弥散。
用法:
    python src\profile_ring.py <图片路径>
"""
import sys
from pathlib import Path

import cv2
import numpy as np

BIN = 2  # 直方图 bin 宽度 (px)


def coarse_center(edges):
    """粗圆心: Canny 边缘密度(下采样 16)的质心，金属表面纹理密度远高于背景。"""
    h, w = edges.shape
    ds = edges[::16, ::16] > 0  # 密采样
    ys, xs = np.nonzero(ds)
    if len(xs) < 10:
        raise SystemExit("边缘点太少")
    cx, cy = float(xs.mean() * 16), float(ys.mean() * 16)
    print(f"粗圆心(边缘密度质心): ({cx:.0f},{cy:.0f})  边缘采样点 {len(xs)}")
    return cx, cy


def cross_projection(edges):
    """列/行投影找外圈四边: 左右弧->列峰, 上下弧->行峰。返回(外圈box)。"""
    h, w = edges.shape
    col = edges.sum(axis=0) / h
    row = edges.sum(axis=1) / w
    # 平滑(宽50)避免单列噪声
    k = int(50 / 2) * 2 + 1
    col_s = cv2.GaussianBlur(col.reshape(1, -1), (1, k), 0).ravel()
    row_s = cv2.GaussianBlur(row.reshape(1, -1), (1, k), 0).ravel()

    def peaks(v, thr_ratio=0.35):
        out = []
        mn, mx = v.min(), v.max()
        thr = mx * thr_ratio
        for i in range(1, len(v) - 1):
            if v[i] >= v[i - 1] and v[i] >= v[i + 1] and v[i] > thr:
                out.append((int(i), float(v[i])))
        # 合并邻峰
        merged = []
        for i, a in out:
            if merged and i - merged[-1][0] <= k:
                if a > merged[-1][1]:
                    merged[-1] = (i, a)
            else:
                merged.append((i, a))
        return merged

    cp, rp = peaks(col_s), peaks(row_s)
    print("列投影峰(x,幅度):", [(x, round(a, 4)) for x, a in cp[:12]])
    print("行投影峰(y,幅度):", [(y, round(a, 4)) for y, a in rp[:12]])
    if len(cp) >= 2 and len(rp) >= 2:
        left, right = cp[0][0], cp[-1][0]
        top, bottom = rp[0][0], rp[-1][0]
        return (left, top, right, bottom)
    return None


def radial_hist(edges, cx, cy, r_min=100, r_max=2000):
    """所有边缘点到圆心的距离 -> (bin边缘, 直方图)。"""
    ys, xs = np.nonzero(edges)
    d = np.hypot(xs - cx, ys - cy)
    keep = (d >= r_min) & (d <= r_max)
    d = d[keep]
    bins = np.arange(r_min, r_max + BIN, BIN)
    hist, _ = np.histogram(d, bins=bins)
    return bins[:-1] + BIN / 2, hist


def find_peaks(centers, hist, min_pts, exclude=None):
    """简单峰值: 局部最大 + 边界最低点(峰谷) + 峰宽=连续>50%峰高范围。"""
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] >= min_pts:
            peaks.append(i)
    # 合并相邻候选，取最高
    merged = []
    for i in peaks:
        if merged and i - merged[-1][0] <= 3:
            if hist[i] > hist[merged[-1][0]]:
                merged[-1] = (i, hist[i])
        else:
            merged.append((i, hist[i]))
    out = []
    for i, h in merged:
        r = centers[i]
        if exclude and exclude[0] <= r <= exclude[1]:
            continue
        # 峰宽取 r±10% 内低于半高的最近点
        lo, hi = i, i
        while lo > 0 and hist[lo - 1] >= h * 0.5 and (centers[lo] - centers[i]) > -r * 0.1:
            lo -= 1
        while hi < len(hist) - 1 and hist[hi + 1] >= h * 0.5 and (centers[hi] - centers[i]) < r * 0.1:
            hi += 1
        width = centers[hi] - centers[lo]
        out.append((r, h, width))
    out.sort(key=lambda t: -t[1])
    return out


def refine_center(edges, cx, cy, radius_band, search=200, step=10):
    """在粗圆心附近网格搜索: 峰高度最大的圆心。"""
    best = (cx, cy, -1)
    for dy in range(-search, search + 1, step):
        for dx in range(-search, search + 1, step):
            ccx, ccy = cx + dx, cy + dy
            centers, hist = radial_hist(edges, ccx, ccy, radius_band[0], radius_band[1])
            amp = hist.max()
            if amp > best[2]:
                best = (ccx, ccy, amp)
    print(f"圆心精化: ({best[0]:.0f},{best[1]:.0f})  峰高 {best[2]}")
    return best[0], best[1]

def main(p: Path):
    img = cv2.imread(str(p))
    if img is None:
        raise SystemExit(f"无法读取: {p}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blur, int(thr * 0.3), int(thr * 0.9))
    n = np.count_nonzero(edges)
    print(f"Canny({int(thr*0.3)},{int(thr*0.9)}) 边缘点 {n}")

    cx, cy = coarse_center(edges)
    print("\n十字投影定位外圈:")
    box = cross_projection(edges)
    if box:
        left, top, right, bottom = box
        cx, cy = (left + right) / 2, (top + bottom) / 2
        print(f"  外圈 box ({left},{top})-({right},{bottom}) -> 中心 ({cx:.0f},{cy:.0f}) "
              f"直径 {right-left}px x {bottom-top}px")
    else:
        cx, cy = refine_center(edges, cx, cy, (700, 1800), search=250, step=10)

    centers, hist = radial_hist(edges, cx, cy)
    peaks = find_peaks(centers, hist, min_pts=int(n * 0.003))
    print("\n径向距离直方图峰值(半径, 点数, 峰宽):")
    for r, h, w_ in peaks[:12]:
        print(f"  r={r:7.1f}px  n={h:6d}  宽{w_:6.1f}px  -> 直径 {2*r:7.1f}px")

    # 保存直方图 + 边缘可视化
    dbg = gray.copy()
    dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)
    cv2.circle(dbg, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    cv2.imwrite("debug/radial_edges.jpg", dbg)
    np.savez("debug/radial_hist.npz", centers=centers, hist=hist, cx=cx, cy=cy)
    print(f"\n已保存: debug/radial_edges.jpg, debug/radial_hist.npz (圆心 {cx:.1f},{cy:.1f})")


if __name__ == "__main__":
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/captures/capture_20260908_115557.jpg")
    main(p)
