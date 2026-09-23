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
        gs = cv2.GaussianBlur(gi.reshape(1, -1), (1, 7), 0).ravel()
        grads = np.gradient(gs)
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

def kasa_robust(pts, outlier=0.09, iters=3, min_pts=25):
    """无中心先验的代数圆拟合(Käsa) + 残差滤除迭代。返回 (cx, cy, r, 保留点集)。"""
    pts = np.asarray(pts, float)
    for _ in range(iters):
        if len(pts) < min_pts:
            return None
        A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
        b = (pts ** 2).sum(1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = sol[0] / 2, sol[1] / 2
        r = float(np.sqrt(sol[2] + cx * cx + cy * cy))
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        keep = np.abs(d - r) <= r * outlier
        if keep.all():
            return cx, cy, r, pts
        pts = pts[keep]
    return cx, cy, r, pts


def hole_dark_points(sm, x, y, norm, cap=130):
    """以(x,y)为中心取局部暗点集: norm<0.72 且在 cap 圆内。"""
    x, y = int(x), int(y)
    h, w = sm.shape
    x0, y0 = max(0, x - cap), max(0, y - cap)
    x1, y1 = min(w, x + cap), min(h, y + cap)
    m = (norm[y0:y1, x0:x1] < 0.72).astype(np.uint8) * 255
    circ = cv2.circle(np.zeros_like(m), (x - x0, y - y0), cap, 255, -1)
    pts = np.argwhere(cv2.bitwise_and(m, m, mask=circ) > 0)[:, ::-1].astype(float)
    if len(pts):
        pts[:, 0] += x0
        pts[:, 1] += y0
    return pts

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
    rt = np.empty(len(pts))
    for q in range(len(pts)):
        lo, hi = 0.0, max(a1, a2) * 2
        for _ in range(36):
            mid = (lo + hi) / 2
            v = ((mid * xr[q]) / (a1 / 2)) ** 2 + ((mid * yr[q]) / (a2 / 2)) ** 2
            if v > 1:
                hi = mid
            else:
                lo = mid
        rt[q] = lo
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


def detect(img, init_c=None, init_r=None):
    """完整检测管线(BGR图 -> 测量结果 dict)。

    返回:
        outer: (center,(axis1,axis2),angle[,n]) 或 None
        center_hole: 同上
        bolts: [{cx,cy,dia,a1,a2,status,n}]
        缺外圈/中心孔时螺栓孔不测。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sm = cv2.GaussianBlur(gray, (0, 0), 5)

    # ---- 外圈: 从图像中心 + 粗略半径(0.44*短边)起, 失败则试探几个半径 ----
    cx0, cy0 = init_c or (w / 2, h / 2)
    outer = n_o = None
    for r0 in [init_r or min(w, h) * 0.44, min(w, h) * 0.40, min(w, h) * 0.48]:
        pts = find_edge_points(sm, cx0, cy0, r0 * 0.88, r0 * 1.16, -1, 2.0, n_ray=72,
                               mode="outer_first")
        outer, n_o = filter_and_fit(pts, cx0, cy0, r0)
        if outer is not None:
            break
    if outer is None:
        return {"outer": None, "center_hole": None, "bolts": []}
    # 粗结果精化: 以粗椭圆中心/半径做紧窗口再采一轮(仍取最外侧边缘)
    (cx1, cy1), (aa1, aa2), _ = outer
    req = (aa1 + aa2) / 4
    pts = find_edge_points(sm, cx1, cy1, req * 0.93, req * 1.07, -1, 2.0, n_ray=72,
                           mode="outer_first")
    outer2, n2 = filter_and_fit(pts, cx1, cy1, req * 1.0)
    if outer2 is not None:
        outer, n_o = outer2, n2
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

    # ---- 螺栓孔: 模板定位 + 等角校验 -> 每孔亚像素椭圆拟合 ----
    bolts = []
    if hole is not None:
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
