r"""圆环零件完整测量管线（find_circles 原型）：
    外圈/中心孔:  72射线亚像素边缘 + 椭圆拟合
    6 螺栓孔:     粗暗斑挖模板 -> 模板匹配峰值 -> 等角共圆校验(滤假峰/补缺位)
                  -> 每孔 48射线亚像素椭圆拟合

用法:
    python src\fit_ellipse_ring.py <图片路径>
输出: 外圈/中心孔椭圆、6孔直径(px)；调试图 debug/fit_debug.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

N_RAY = 72

# 外圈候选的可信门槛（自适应定位用；实测依据见 docs/自适应定位改造计划.md）
OUTER_COVER_MIN = 0.6        # 保留点数 ≥ 射线数×此比例（误锁只有 22/72，正确为 54~72/72）
OUTER_AXIS_RATIO_MAX = 1.8   # 外圈长短轴比上限（斜视视角的物理上限）
OUTER_MAX_CANDIDATES = 60    # 候选上限，防自适应搜索跑飞
OUTER_CENTER_MIN_RATIO = 0.5  # 圆心最强径向峰 < 全局最强峰×此比例就丢弃（零件不在这儿）


# ---------------- 射线亚像素边缘 ----------------

def ray_profile(gray, cx, cy, ang, max_r):
    d = np.arange(0, max_r, 0.5)
    xs = (cx + np.cos(ang) * d).astype(int).clip(0, gray.shape[1] - 1)
    ys = (cy + np.sin(ang) * d).astype(int).clip(0, gray.shape[0] - 1)
    return d, gray[ys, xs].astype(float)


def find_edge_points(gray, cx, cy, r_lo, r_hi, polarity=-1, min_grad=1.2, n_ray=N_RAY,
                     mode="steepest"):
    """射线法找亚像素边缘点集。polarity=-1 下降沿(亮->暗)；+1 上升沿。

    mode='steepest': 窗口内最陡梯度(默认, 用于外圈/中心孔等单一结构);
    mode='outer_first': 从窗口外侧往内找第一个满足极性的边(用于孔外缘)。
    """
    pts = []
    for k in range(n_ray):
        ang = 2 * np.pi * k / n_ray
        d, g = ray_profile(gray, cx, cy, ang, r_hi * 1.08)
        m = (d >= r_lo) & (d <= r_hi)
        di, gi = d[m], g[m]
        if len(gi) < 8:
            continue
        grads = np.gradient(gi)
        if mode == "outer_first":
            j = None
            for jj in range(len(grads) - 3, 3, -1):
                if polarity < 0:
                    if grads[jj] < -min_grad and grads[jj] <= grads[jj - 1] and grads[jj] <= grads[jj + 1]:
                        j = jj
                        break
                else:
                    if grads[jj] > min_grad and grads[jj] >= grads[jj - 1] and grads[jj] >= grads[jj + 1]:
                        j = jj
                        break
            if j is None:
                continue
            di_j = di[j] - 0.5
        else:
            j = int(np.argmin(grads)) if polarity < 0 else int(np.argmax(grads))
            score = -grads[j] if polarity < 0 else grads[j]
            if score < min_grad:
                continue
            j = min(max(j, 1), len(grads) - 2)
            di_j = di[j]
        a, b, c = grads[j - 1], grads[j], grads[j + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0
        x, y = cx + np.cos(ang) * (di_j + delta), cy + np.sin(ang) * (di_j + delta)
        pts.append((x, y))
    return np.array(pts).reshape(-1, 2) if pts else np.zeros((0, 2))


def filter_and_fit(pts, cx, cy, r0, outlier=0.09, min_pts=20):
    """圆等价归一化距离滤离群 + 椭圆拟合迭代。返回(cv2 ell, 点数)。"""
    pts = np.asarray(pts, float)
    r = r0
    for _ in range(3):
        if len(pts) < min_pts:
            return None, 0
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        keep = np.abs(d - r) <= r * outlier
        pts = pts[keep]
        if len(pts) < min_pts:
            return None, 0
        ell = cv2.fitEllipse(pts.astype(np.float32))
        (cx, cy), (a1, a2), ang = ell
        r = (a1 + a2) / 4
    return ell, len(pts)


# ---------------- 椭圆归一化坐标系(用于共圆校验) ----------------

def to_norm(pts, ocx, ocy, oa1, oa2, oang):
    """图像坐标 -> 外圈椭圆归一化单位圆坐标。"""
    ca, sa = np.cos(np.deg2rad(-oang)), np.sin(np.deg2rad(-oang))
    pts = np.asarray(pts, float)
    dx = (pts[..., 0] - ocx) * ca - (pts[..., 1] - ocy) * sa
    dy = (pts[..., 0] - ocx) * sa + (pts[..., 1] - ocy) * ca
    return dx / (oa1 / 2), dy / (oa2 / 2)


def from_norm(nx, ny, ocx, ocy, oa1, oa2, oang):
    """归一化单位圆坐标 -> 图像坐标。"""
    ca, sa = np.cos(np.deg2rad(oang)), np.sin(np.deg2rad(oang))
    dx, dy = nx * (oa1 / 2), ny * (oa2 / 2)
    return ocx + dx * ca - dy * sa, ocy + dx * sa + dy * ca


# ---------------- 螺栓孔: 模板匹配 + 等角共圆校验 ----------------

def rough_templates(sm, band, min_area=3000):
    """环带内粗找暗连通域，按圆度x面积选最像孔的区域做模板。"""
    bg = cv2.blur(sm, (201, 201))
    norm = sm / np.maximum(bg, 1)
    dark = cv2.bitwise_and(((norm < 0.62).astype(np.uint8) * 255), band)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * a / (per * per) if per else 0
        if circ < 0.55:
            continue
        score = circ * np.sqrt(a)
        if best is None or score > best[0]:
            best = (score, c)
    if best is None:
        return None
    x, y, w2, h2 = cv2.boundingRect(best[1])
    cx, cy = int(x + w2 / 2), int(y + h2 / 2)
    t = sm[cy - 125:cy + 125, cx - 125:cx + 125].astype(np.float32)
    if t.shape != (250, 250):
        t = cv2.resize(t, (250, 250))
    return t - t.mean()


def template_locate(sm, template, expect=12, scale=3):
    """缩图模板匹配，返回峰 (原图x, 原图y, 响应)，按响应降序。"""
    h, w = sm.shape
    ts = cv2.resize(template, (template.shape[1] // scale, template.shape[0] // scale),
                    interpolation=cv2.INTER_AREA)
    ts = ts - ts.mean()
    s = cv2.resize(sm, (w // scale, h // scale), interpolation=cv2.INTER_AREA).astype(np.float32)
    res = cv2.matchTemplate(s, ts, cv2.TM_CCOEFF_NORMED)
    peaks = []
    rr = res.copy()
    while True:
        _, mx, _, loc = cv2.minMaxLoc(rr)
        if mx < 0.30:
            break
        x = loc[0] * scale + ts.shape[1] // 2 * scale
        y = loc[1] * scale + ts.shape[0] // 2 * scale
        peaks.append((x, y, float(mx)))
        cv2.circle(rr, loc, 150, -1, -1)
        if len(peaks) >= expect:
            break
    return peaks


def hexagon_slots(cands, ocx, ocy, oa1, oa2, oang, tol_ang=13, tol_r=0.15):
    """RANSAC 六边形假设检验: 以命中数最高的候选为相位锚, 生成 6 个等角槽位。

    返回 [(期望角, 归一化半径, 峰orNone)]。
    """
    if not cands:
        return None
    nx, ny = to_norm([(c[0], c[1]) for c in cands], ocx, ocy, oa1, oa2, oang)
    r2 = np.hypot(nx, ny)
    th2 = (np.degrees(np.arctan2(ny, nx)) + 360) % 360
    best_i, best_score = -1, -1
    for i in range(len(cands)):
        hits = 0
        for j in range(len(cands)):
            if i == j:
                continue
            dmod = ((th2[j] - th2[i]) % 60 + 60) % 60
            dd = min(dmod, 60 - dmod)              # 与锚+60k 的最短角距
            if dd < tol_ang and abs(r2[j] - r2[i]) / r2[i] < tol_r:
                hits += 1
        score = hits + cands[i][2] * 0.5               # 同命中数时响应高的优先
        if score > best_score:
            best_i, best_score = i, score
    p0 = cands[best_i]
    th0, r0 = th2[best_i], r2[best_i]
    slots = []
    for k in range(6):
        exp_ang = (th0 + 60 * k) % 360
        dd = (th2 - exp_ang + 180) % 360 - 180       # 绝对角差(最近360)
        m = (np.abs(dd) < tol_ang) & (np.abs(r2 - r0) / r0 < tol_r) & \
            (np.array([c[2] for c in cands]) >= 0.35)
        got = [c for c, ok in zip(cands, m) if ok]
        best = max(got, key=lambda c: c[2]) if got else None
        slots.append((exp_ang, r0, best))
    return slots


# ---------------- 螺栓孔直径: 螺丝头锚定 + 沉孔口阈值交叉 + 净弧椭圆拟合 ----------------

def hole_head_center(norm, x, y, win=160, thr=0.72):
    """暗斑内被包围的亮区(螺丝头)质心 = 孔锚点。

    沉孔里有斜置螺丝头(亮), 外圈是暗环带; 暗环与外圈暗带(磨损带/中心孔
    槽缘)可能粘连, 但螺丝头永远独立且居孔内近心。win=160 局部窗口内做
    norm<0.72 掩膜 + 开/闭运算 + 补洞, 取"被包围亮区"中离模板中心最近
    的组件质心作为锚点。
    """
    x0, y0 = max(0, int(x) - win), max(0, int(y) - win)
    x1, y1 = min(norm.shape[1], int(x) + win), min(norm.shape[0], int(y) + win)
    m = (norm[y0:y1, x0:x1] < thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    ff = pad.copy()
    ffmask = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, ffmask, (0, 0), 255)  # 外圈pad1px, 种子必在背景
    added = cv2.bitwise_and(pad | cv2.bitwise_not(ff), cv2.bitwise_not(pad))[1:-1, 1:-1]
    nlab, lab, stats, cent = cv2.connectedComponentsWithStats(added, 8)
    best = None
    for k in range(1, nlab):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < 200:
            continue
        dd = np.hypot(cent[k][0] - (x - x0), cent[k][1] - (y - y0))
        if best is None or dd < best[0]:
            best = (dd, k, area)
    if best is None:
        # 回退: 暗斑(包含种子点的组件, 填充后)质心; 种子不在暗区则原样返回
        k = lab[int(y) - y0, int(x) - x0]
        if k == 0 or stats[k, cv2.CC_STAT_AREA] < 2500:
            return x, y, None
        comp = (lab == k).astype(np.uint8) * 255
        pad2 = cv2.copyMakeBorder(comp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
        ff2 = pad2.copy()
        fmask = np.zeros((pad2.shape[0] + 2, pad2.shape[1] + 2), np.uint8)
        cv2.floodFill(ff2, fmask, (0, 0), 255)
        filled = (pad2 | cv2.bitwise_not(ff2))[1:-1, 1:-1]
        M = cv2.moments(filled)
        if M["m00"] == 0:
            return x, y, None
        return x0 + M["m10"] / M["m00"], y0 + M["m01"] / M["m00"], None
    bx, by = cent[best[1]]
    return x0 + bx, y0 + by, best[2]


def _head_radius(norm, hx, hy, n_dir=12, thr=0.72):
    """12方向首暗点距离的中位数 = 螺丝头半径。"""
    starts = []
    for k in range(n_dir):
        ang = 2 * np.pi * k / n_dir
        d, g = ray_profile(norm, hx, hy, ang, 100)
        jv = np.where(g < thr)[0]
        if len(jv):
            starts.append(d[jv[0]])
    return float(np.median(starts)) if starts else 0.0


def rim_cross_points(norm, x, y, head_r, n_ray=N_RAY, thr=0.72, max_r=150, min_w=12):
    """每射线: 头外侧第一个宽度>=min_w 的连续暗游程末端的亚像素 0.72 交叉点。

    沉孔口是 ~20px 宽的缓坡(沉孔锥面+斜视), 无硬边缘, 用固定阈值交叉点
    做可复现定义。暗游程延续到窗口边缘 => 与暗带粘连, 弃。
    跳过短游程(螺丝槽纹/头影阴弧 ~5-15px), 防止提前交叉。
    """
    head_r = int(np.clip(head_r, 0, 60))
    step = 0.5
    pts, angs = [], []
    for k in range(n_ray):
        ang = 2 * np.pi * k / n_ray
        d, g = ray_profile(norm, x, y, ang, max_r)
        dark = g < thr
        i = head_r + 6
        while i < len(dark):
            while i < len(dark) and not dark[i]:
                i += 1
            j = i
            while j < len(dark) and dark[j]:
                j += 1
            if j - i >= min_w / step:      # 宽游程 = 沉孔口
                if j >= len(dark) - 4:
                    break                  # 延续到窗口边缘 => 粘连, 弃
                if j + 1 < len(g) and g[j + 1] > g[j]:
                    r = d[j] + (thr - g[j]) * (d[j + 1] - d[j]) / (g[j + 1] - g[j])
                else:
                    r = d[j]
                pts.append((x + np.cos(ang) * r, y + np.sin(ang) * r))
                angs.append(np.degrees(ang))
                break                      # 取第一个宽游程(沉孔口), 其后暗带不计
            i = j + 1
    return (np.array(pts).reshape(-1, 2), np.array(angs)) if pts else (np.zeros((0, 2)), np.zeros(0))


def polar_resid(ell, pts):
    """点到椭圆(极角方向)的径向残差 px。"""
    pts = np.asarray(pts, float)
    (cx, cy), (a1, a2), ang = ell
    ca, sa = np.cos(np.deg2rad(-ang)), np.sin(np.deg2rad(-ang))
    th = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    xr = np.cos(th) * ca + np.sin(th) * sa
    yr = -np.cos(th) * sa + np.sin(th) * ca
    # 解 ((rt*xr)/(a1/2))² + ((rt*yr)/(a2/2))² = 1 的闭式解
    rt = 1.0 / np.sqrt((xr / (a1 / 2)) ** 2 + (yr / (a2 / 2)) ** 2)
    return np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - rt


def fit_clean_arc(pts, angs, cx, cy, min_pts=15):
    """沉孔口鲁棒椭圆拟合: 渐进收紧的极化残差迭代。

    阶段1 用宽松容差(2.5×中位, 12px下限)甩开与暗带/槽缘粘连的远点,
    阶段2 收紧到(1.0×中位, 4px下限)只留干净弧段的点。
    结果须通过形状约束(轴比 1.0-2.2, 尺寸 30-320px)防圆弧塌陷;
    塌陷则退回宽松版本(3×中位, 9px下限)。
    """
    pts = np.asarray(pts, float)
    if len(pts) < min_pts:
        return None, 0
    ell = cv2.fitEllipse(pts.astype(np.float32))

    def _shape_ok(e):
        (_, _), (a1, a2), _ = e
        return 30 <= min(a1, a2) and max(a1, a2) <= 320 \
            and 1.0 <= max(a1, a2) / min(a1, a2) <= 2.2

    for rate, floor in [(2.5, 12.0), (1.6, 7.0), (1.1, 5.0), (1.0, 4.0)]:
        res = polar_resid(ell, pts)
        keep = np.abs(res) <= max(np.median(np.abs(res)) * rate, floor)
        pts = pts[keep]
        if len(pts) < min_pts:
            return None, len(pts)
        ell = cv2.fitEllipse(pts.astype(np.float32))
    if _shape_ok(ell):
        return ell, len(pts)
    # 塌陷回退: 宽松版再走一遍
    ell = cv2.fitEllipse(pts.astype(np.float32))
    for _ in range(3):
        res = polar_resid(ell, pts)
        keep = np.abs(res) <= max(np.median(np.abs(res)) * 3, 9)
        pts = pts[keep]
        if len(pts) < min_pts:
            return None, len(pts)
        ell = cv2.fitEllipse(pts.astype(np.float32))
    if _shape_ok(ell):
        return ell, len(pts)
    return None, len(pts)


def measure_bolt(norm, x, y):
    """单螺栓孔: 头锚定 -> 阈值交叉点 -> 净弧拟合(+二遍校正)。返回 (ellipse, n, 锚点)。"""
    hx, hy, _ = hole_head_center(norm, x, y)
    hr = _head_radius(norm, hx, hy)
    pts, angs = rim_cross_points(norm, hx, hy, hr)
    ell, n = fit_clean_arc(pts, angs, hx, hy) if len(pts) >= 15 else (None, 0)
    if ell is None:
        return None, 0, (hx, hy)
    (ex, ey), _, _ = ell
    pts2, angs2 = rim_cross_points(norm, ex, ey, hr)
    if len(pts2) >= 15:
        ell2, n2 = fit_clean_arc(pts2, angs2, ex, ey)
        if ell2 is not None:
            ell, n = ell2, n2
    return ell, n, (hx, hy)


def _edge_map(gray):
    """Otsu 定阈值 -> Canny 边缘图（配方沿用诊断脚本 profile_ring.py）。"""
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.Canny(blur, int(thr * 0.3), int(thr * 0.9))


def _coarse_centers(gray, edges):
    """候选圆心：图像中心(保底) + 边缘密度质心 + Otsu 最大轮廓外接圆心。

    实测（型号2 偏置图，盘心偏出画面中心 604px）：后两个各自都能独立得到正确
    结果，且三者互相一致——所以不需要哪一个特别准，多给几个让打分去挑。
    """
    h, w = gray.shape
    out = [(w / 2.0, h / 2.0)]
    ys, xs = np.nonzero(edges[::16, ::16])
    if len(xs) >= 10:
        out.append((float(xs.mean() * 16), float(ys.mean() * 16)))
    try:
        _, bw = cv2.threshold(cv2.GaussianBlur(gray, (9, 9), 0), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            (ccx, ccy), _ = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
            out.append((float(ccx), float(ccy)))
    except cv2.error:
        pass
    return out


def _radial_peaks(ey, ex, cx, cy, r_min, r_max, top=6, bin_px=2.0, merge_ratio=0.08):
    """边缘点到 (cx,cy) 的距离直方图取峰 = 候选外圈半径。返回 [(峰点数, 半径)]。

    真实圆在"边缘点距离"直方图上形成尖峰，噪声弧则弥散（续 profile_ring.py 的思路）。
    实测型号2：盘缘峰宽仅 6px。

    **邻近峰要合并**（merge_ratio=8%）：圆心估不准时真半径会被抹成一片，
    1143/1149/1157 这种邻居其实都指向同一个答案，全试一遍纯属浪费。
    ey/ex 是 np.nonzero(edges) 的结果，调用方算一次传进来。
    """
    d = np.hypot(ex - cx, ey - cy)
    d = d[(d >= r_min) & (d <= r_max)]
    if len(d) < 100:
        return []
    bins = np.arange(r_min, r_max + bin_px, bin_px)
    hist, _ = np.histogram(d, bins=bins)
    centers = bins[:-1] + bin_px / 2
    raw = [(int(hist[i]), float(centers[i])) for i in range(1, len(hist) - 1)
           if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]]
    raw.sort(key=lambda t: -t[0])
    out = []
    for cnt, r in raw:
        if all(abs(r - r2) > r2 * merge_ratio for _, r2 in out):
            out.append((cnt, r))
        if len(out) >= top:
            break
    return out


def _outer_candidates(gray, w, h, cx0, cy0, init_r):
    """外圈候选 (cx, cy, r0, polarity)，**旧先验排最前以保证基线读数不变**。

    历史先验只假设了"零件在画面正中 + 外圈半径 ≈ 短边的 44% + 边缘是亮→暗"，
    零件一挪位置/换型号就全废。数据驱动的候选由
    「候选圆心 × 径向峰半径 × 两种极性」组成。

    极性必须一起试：型号1 是暗金属在白背景上（下降沿，72/72），型号2 是灰盘放在
    白布上、盘外更亮（上升沿 55/72，下降沿只有 22/72）——写死下降沿直接废掉。

    这是个生成器：旧先验那三个候选排在前面，**边缘图与直方图是等它被继续迭代时
    才算的**。所以旧先验一旦过了门槛就 break，自适应那套开销（实测约 400ms）
    根本不会发生——只有快速路径失败时才付这个代价。
    """
    short = min(w, h)
    for r0 in [init_r or short * 0.44, short * 0.40, short * 0.48]:
        yield cx0, cy0, r0, -1
    try:
        edges = _edge_map(gray)
    except cv2.error:
        return
    ey, ex = np.nonzero(edges)
    if len(ex) < 100:
        return
    per_center = []
    for (ccx, ccy) in _coarse_centers(gray, edges):
        peaks = _radial_peaks(ey, ex, ccx, ccy, short * 0.10, short * 0.55, top=6)
        if peaks:
            per_center.append(((ccx, ccy), peaks))
    if not per_center:
        return
    # 峰最弱的圆心直接丢：零件不在那儿，试也是白试。实测型号2 图像中心的最强峰
    # 只有最佳圆心的 41%，丢得掉；零件真居中时它本来就是最强的，留得下。
    strongest = max(p[1][0][0] for p in per_center)
    n = 0
    for (ccx, ccy), peaks in per_center:
        if peaks[0][0] < strongest * OUTER_CENTER_MIN_RATIO:
            continue
        for _, r in peaks:
            for pol in (-1, +1):
                n += 1
                if n > OUTER_MAX_CANDIDATES:
                    return
                yield ccx, ccy, float(r), pol


def _fit_outer_at(sm, cx, cy, r0, pol):
    """单个粗候选：射线找边 + 鲁棒圆滤 + 椭圆拟合。"""
    pts = find_edge_points(sm, cx, cy, r0 * 0.88, r0 * 1.16, pol, 2.0, n_ray=N_RAY,
                           mode="outer_first")
    return filter_and_fit(pts, cx, cy, r0)


def _refine_outer(sm, ell, n, pol):
    """粗结果精化：以粗椭圆中心/半径做紧窗口再采一轮（仍取最外侧边缘）。"""
    (cx1, cy1), (aa1, aa2), _ = ell
    req = (aa1 + aa2) / 4
    pts = find_edge_points(sm, cx1, cy1, req * 0.93, req * 1.07, pol, 2.0, n_ray=N_RAY,
                           mode="outer_first")
    ell2, n2 = filter_and_fit(pts, cx1, cy1, req * 1.0)
    return (ell2, n2) if ell2 is not None else (ell, n)


def _outer_ok(ell, n, w, h):
    """外圈候选是否可信：点数够 + 轴比合理 + 不出画幅。

    **残差 RMS 不能当门槛**：实测基线自身就有 41px、错误极性在标定图上也有 41px
    （斜视外缘本身是个有厚度的带，不是一条线），两者区分不开。点数和画幅才管用：
    型号2 那次误锁用点只有 22/72，外接框还溢出画幅（x 到 -48、y 到 3174），
    两条里任意一条都挡得住。
    """
    if n < OUTER_COVER_MIN * N_RAY:
        return False
    (cx, cy), (a1, a2), _ = ell
    if min(a1, a2) <= 0 or max(a1, a2) / min(a1, a2) > OUTER_AXIS_RATIO_MAX:
        return False
    s = max(a1, a2) / 2          # 保守：按长半轴当各向外接半径
    return (cx - s >= 0) and (cx + s <= w) and (cy - s >= 0) and (cy + s <= h)


# ---------------- 型号档案驱动的螺栓孔（装好紧固件的暗芯圆孔） ----------------

def core_cross_points(gray, x, y, r_lo, r_hi, n_ray=48, frac=0.5):
    """从暗芯向外：灰度升到 (芯内 + frac×(外侧-芯内)) 的亚像素交叉点。

    型号2 的"孔"是装好的紧固件：中间一个暗芯 + 外面一圈亮金属环，跟型号1 的
    "暗沉孔 + 亮螺丝头"正好反着。实测(2026-10-07)这个**中分界**定义对得上标称
    1.78mm（36.4px vs 标称 36.8px，差 1.1%），而用 find_edge_points 的"最陡上升沿"
    会量成 50.7px（偏 38%）——所以这里刻意不复用最陡梯度那套。
    """
    pts = []
    for k in range(n_ray):
        ang = 2 * np.pi * k / n_ray
        d, gv = ray_profile(gray, x, y, ang, r_hi)
        m = (d >= r_lo) & (d <= r_hi)
        dd, gg = d[m], gv[m]
        if len(gg) < 8:
            continue
        lo_v = float(gg[:max(1, len(gg) // 4)].min())      # 内侧 = 暗芯
        hi_v = float(np.median(gg[len(gg) // 2:]))         # 外侧 = 亮环
        if hi_v - lo_v < 8:                                # 对比度不够就别硬给点
            continue
        thr = lo_v + frac * (hi_v - lo_v)
        idx = np.where(gg > thr)[0]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        if i == 0:
            r = dd[0]
        else:
            dr = gg[i] - gg[i - 1]
            r = dd[i - 1] + (thr - gg[i - 1]) * (dd[i] - dd[i - 1]) / dr if dr else dd[i]
        pts.append((x + np.cos(ang) * r, y + np.sin(ang) * r))
    return np.array(pts).reshape(-1, 2) if pts else np.zeros((0, 2))


def core_seed(sm, x, y, r_exp):
    """把种子点挪到暗芯的亮度重心上。

    相位搜索给的位置有十几 px 误差，而暗芯半径才 ~20px——偏这么多会让射线上的
    中分界系统性外扩（实测某一个孔因此从 40px 涨到 47.7px，是 4 孔里唯一的离群）。
    """
    w = int(max(6.0, r_exp * 1.2))
    x0, y0 = max(0, int(x) - w), max(0, int(y) - w)
    x1, y1 = min(sm.shape[1], int(x) + w), min(sm.shape[0], int(y) + w)
    loc = sm[y0:y1, x0:x1].astype(float)
    if loc.size < 4:
        return float(x), float(y)
    thr = loc.min() + 0.5 * (float(np.median(loc)) - loc.min())
    wgt = np.clip(thr - loc, 0.0, None)
    if wgt.sum() <= 0:
        return float(x), float(y)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return float((xs * wgt).sum() / wgt.sum()), float((ys * wgt).sum() / wgt.sum())


def measure_core_hole(sm, x, y, r_exp):
    """量暗芯圆孔：先把种子挪到暗芯重心 -> 射线找中分界 -> 中位半径当直径。
    返回 ((cx,cy), dia_px, 方向数)。

    这里**刻意不拟合椭圆**：实测暗芯有一侧常连到阴影，一拟合就被拉成 1.3 左右的轴比、
    均值因此偏高 7~11%（4 孔 1.66~2.28mm）；中位半径对少数方向上的粘连不敏感，
    4 孔均值对上标称 1.78mm 只差 1% 量级。
    代价是丢掉椭圆方位——型号2 这种接近正拍的机位不需要它。
    """
    x, y = core_seed(sm, x, y, r_exp)
    pts = core_cross_points(sm, x, y, r_exp * 0.4, r_exp * 2.2)
    if len(pts) < 12:
        return None, 0.0, len(pts)
    d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
    med = float(np.median(d))
    keep = np.abs(d - med) <= max(med * 0.25, 3.0)
    if keep.sum() >= 12:
        pts, d = pts[keep], d[keep]
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    return (cx, cy), 2.0 * float(np.median(d)), int(len(d))


def profile_slots(profile, norm, ocx, ocy, oa1, oa2, oang, phase_step=1.0):
    """按档案的显式角度定位各孔：先扫整体相位，再给出各孔角度。

    型号2 的 4 孔是 2x2 矩形布局（实测相邻间隔 72.8/107.4/72.0/107.8°，**不是 90° 等分**），
    所以不能沿用"等分 N 槽"。档案里存的是**相对角度**，绝对相位由图像数据自己找
    （扫一圈让模板角度尽量落在暗处），零件转个角度也不怕。
    """
    rel = profile.get("bolt_angles_rel")
    if not rel:
        return None
    rel = np.asarray(rel, float)
    r_norm = float(profile.get("bolt_circle_ratio", 0.8))
    h, w = norm.shape
    best, best_ph = None, 0.0
    for ph in np.arange(0.0, 360.0, phase_step):
        xs, ys = from_norm(np.cos(np.deg2rad(ph + rel)) * r_norm,
                           np.sin(np.deg2rad(ph + rel)) * r_norm, ocx, ocy, oa1, oa2, oang)
        vals = [norm[int(round(y)), int(round(x))] for x, y in zip(xs, ys)
                if 0 <= int(round(y)) < h and 0 <= int(round(x)) < w]
        if len(vals) < len(rel):
            continue
        score = -float(np.mean(vals))          # 孔是暗的：越暗越像
        if best is None or score > best:
            best, best_ph = score, float(ph)
    return [(best_ph + a) % 360 for a in rel], r_norm


def detect(img, init_c=None, init_r=None, profile=None):
    """完整检测管线(BGR图 -> 测量结果 dict)。

    profile: 型号档案(dict)。None 时行为与无档案版本完全一致（基线路径）。
             feature="core_hole" 的型号（装好紧固件的暗芯圆孔）会改走
             profile_slots + measure_core_hole，不再依赖中心孔。

    返回:
        outer: (center,(axis1,axis2),angle[,n]) 或 None
        center_hole: 同上
        bolts: [{cx,cy,dia,a1,a2,status,n}]
        缺外圈/中心孔时螺栓孔不测。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sm = cv2.GaussianBlur(gray, (0, 0), 5)

    # ---- 外圈: 旧先验优先(保基线)，不行再自适应定位 ----
    cx0, cy0 = init_c or (w / 2, h / 2)
    outer = n_o = None
    for (cx, cy, r0, pol) in _outer_candidates(gray, w, h, cx0, cy0, init_r):
        ell, n = _fit_outer_at(sm, cx, cy, r0, pol)
        if ell is None:
            continue
        ell, n = _refine_outer(sm, ell, n, pol)
        if _outer_ok(ell, n, w, h):
            outer, n_o = ell, n
            break
    if outer is None:
        return {"outer": None, "center_hole": None, "bolts": []}
    (ocx, ocy), (oa1, oa2), oang = outer

    # ---- 中心孔: 外圈内暗->亮的上升沿 ----
    r_outer = (oa1 + oa2) / 4
    pts_h = find_edge_points(sm, ocx, ocy, r_outer * 0.6, r_outer * 0.78, +1, 1.2, n_ray=72)
    hole, n_h = filter_and_fit(pts_h, ocx, ocy, r_outer * 0.7)
    if hole is None:
        hole = None
        hcx, hcy, ha1, ha2, hang = ocx, ocy, 0.0, 0.0, 0.0
    else:
        (hcx, hcy), (ha1, ha2), hang = hole

    # ---- 螺栓孔 ----
    bolts = []
    prof = profile or {}
    mmpx = prof.get("mm_per_px")
    if (prof.get("feature") == "core_hole" and prof.get("bolt_angles_rel")
            and prof.get("bolt_mm") and mmpx):
        # 装好紧固件的"暗芯圆孔"型号（型号2）：按档案角度定位，量暗芯。
        # 这条路不需要中心孔，所以不走下面那个 `if hole is not None` 的门。
        norm = sm / np.maximum(cv2.blur(sm, (201, 201)), 1)
        slots = profile_slots(prof, norm, ocx, ocy, oa1, oa2, oang)
        sm2 = cv2.GaussianBlur(gray, (0, 0), 2)      # 暗芯只有几十 px，σ=5 会糊过头
        r_exp = (float(prof["bolt_mm"]) / 2) / float(mmpx)
        if slots:
            angs_abs, r_norm = slots
            for ang_abs in angs_abs:
                x, y = from_norm(np.cos(np.deg2rad(ang_abs)) * r_norm,
                                 np.sin(np.deg2rad(ang_abs)) * r_norm,
                                 ocx, ocy, oa1, oa2, oang)
                ell, dia, n = measure_core_hole(sm2, x, y, r_exp)
                if ell is None:
                    bolts.append({"cx": x, "cy": y, "dia": 0.0, "a1": 0.0, "a2": 0.0,
                                  "n": n, "status": "失败"})
                    continue
                (ex, ey) = ell
                bolts.append({"cx": ex, "cy": ey, "dia": dia, "a1": dia, "a2": dia,
                              "ang": 0.0, "n": n, "status": "OK"})
    elif hole is not None:
        band = np.zeros((h, w), np.uint8)
        cv2.ellipse(band, (int(ocx), int(ocy)),
                    (int(oa1 / 2 * 0.98), int(oa2 / 2 * 0.98)), oang, 0, 360, 255, -1)
        cv2.ellipse(band, (int(hcx), int(hcy)),
                    (int(ha1 / 2 * 1.05), int(ha2 / 2 * 1.05)), hang, 0, 360, 0, -1)
        norm = sm / np.maximum(cv2.blur(sm, (201, 201)), 1)
        tmpl = rough_templates(sm, band)
        if tmpl is not None:
            peaks = template_locate(sm, tmpl)
            cands = []
            for x, y, v in peaks:
                dd = np.hypot(x - ocx, y - ocy)
                if not (r_outer * 0.5 < dd < r_outer * 1.05):
                    continue
                if len(find_edge_points(sm, x, y, 40, 140, -1, 0.8, n_ray=48)) < 12:
                    continue
                cands.append((x, y, v))
            slots = hexagon_slots(cands, ocx, ocy, oa1, oa2, oang)
            if slots:
                for exp_ang, r_norm, use in slots:
                    if use is not None:
                        x, y = use[0], use[1]
                    else:
                        x, y = from_norm(np.cos(np.deg2rad(exp_ang)) * r_norm,
                                         np.sin(np.deg2rad(exp_ang)) * r_norm,
                                         ocx, ocy, oa1, oa2, oang)
                    ell, n, (hx, hy) = measure_bolt(norm, x, y)
                    if ell is None:
                        bolts.append({"cx": x, "cy": y, "dia": 0.0, "a1": 0.0, "a2": 0.0,
                                      "n": 0, "status": "失败"})
                        continue
                    (ex, ey), (ea1, ea2), eang = ell
                    bolts.append({"cx": ex, "cy": ey, "dia": (ea1 + ea2) / 2,
                                  "a1": ea1, "a2": ea2, "ang": eang, "n": n,
                                  "status": "OK"})
    return {"outer": (outer, n_o), "center_hole": (hole, n_h), "bolts": bolts}


def main(p: Path):
    img = cv2.imread(str(p))
    if img is None:
        raise SystemExit(f"无法读取图片: {p}")
    res = detect(img)
    outer, n_o = res["outer"]
    hole, n_h = res["center_hole"]
    if outer is None:
        raise SystemExit("外圈拟合失败")
    (ocx, ocy), (oa1, oa2), oang = outer
    print(f"外圈: 中心({ocx:.1f},{ocy:.1f}) 轴{oa1:.1f}x{oa2:.1f}px 角度{oang:.1f}° 点数{n_o}")
    if hole:
        (hcx, hcy), (ha1, ha2), hang = hole
        print(f"中心孔: 中心({hcx:.1f},{hcy:.1f}) 轴{ha1:.1f}x{ha2:.1f}px 角度{hang:.1f}° 点数{n_h}")
    else:
        print("中心孔: 拟合失败")
    for i, b in enumerate(res["bolts"], 1):
        if b["status"] == "OK":
            print(f"  螺栓孔{i}: d={b['dia']:.1f}px ({b['a1']:.1f}x{b['a2']:.1f})"
                  f" 中心({b['cx']:.0f},{b['cy']:.0f}) 保留{b['n']}点")
        else:
            print(f"  螺栓孔{i}: @({b['cx']:.0f},{b['cy']:.0f}) 直径测量失败")

    # ---- 调试图 ----
    dbg = img.copy()
    if outer:
        cv2.ellipse(dbg, outer, (0, 255, 0), 8)
    if hole:
        cv2.ellipse(dbg, hole, (0, 0, 255), 8)
    for b in res["bolts"]:
        if b["status"] == "OK":
            cv2.ellipse(dbg, ((b["cx"], b["cy"]), (b["a1"], b["a2"]), 0), (255, 0, 0), 6)
    dbg_dir = Path("debug")
    dbg_dir.mkdir(exist_ok=True)
    out = dbg_dir / f"fit_debug_{Path(p).stem}.jpg"
    cv2.imwrite(str(out), dbg)
    print(f"已保存 {out}")


if __name__ == "__main__":
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/captures/capture_20260908_115557.jpg")
    main(p)
