r"""圆环/法兰类零件尺寸测量：外径、中心孔、螺栓孔，一键出毫米数。

两种入口，共用同一套检测与报告逻辑：
    1) 图形界面（日常用，推荐）：双击项目里的 `测量.bat`
       - 相机已接：自动拍照 → 弹出结果窗口，大字显示各尺寸（见 gui_ring.py）
       - 想测已有照片：把照片文件拖到 `测量.bat` 图标上
    2) 命令行（标定 / 排错用）：项目根目录 vision-roughness/ 下
       python src\measure_ring.py --image data\captures\xxx.jpg
       python src\measure_ring.py --ref-mm 88.60      # 用卡尺外径真值重建换算系数

每次测量会在 debug\ 下存两份记录：measure_ring_debug.jpg（画出轮廓的图）
和 报告_年月日_时分秒.txt（文字报告）。

原理：
    相机+镜头+拍摄距离固定后，"毫米/像素"是常数。用卡尺量出零件外径真值
    建立系数（--ref-mm），之后同机位拍的所有零件都能直接输出毫米。
    ⚠️ 相机动了/零件高度变了/换镜头了，必须用 --ref-mm 重建系数。

检测管线（fit_ellipse_ring.detect）：
    外圈/中心孔:  72 射线亚像素边缘(最外侧下降沿/中心孔上升沿) + 椭圆鲁棒拟合
    螺栓孔:       模板匹配定位 + 六孔等角共圆校验 + 螺丝头锚定
                  -> 沉孔口 0.72 亮度阈值交叉 -> 极化残差鲁棒椭圆拟合
"""
import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, "src")
import config
from camera_adapter import create_camera
from fit_ellipse_ring import detect

RATIO_CACHE = Path("models/mm_per_pixel.json")
DEBUG_DIR = Path("debug")
BORDER_MARGIN = 5  # 轮廓距图像边缘小于此值视为"零件超出视野"


class MeasureError(Exception):
    """可以预料、需要原样提示给用户看的测量失败（不是程序缺陷）。"""


@dataclass
class RingResult:
    """一次测量的全部结果：数字 + 文本报告 + 落盘路径。

    供命令行直接打印，也供 gui_ring.py 大字显示——两条路走同一份数据，
    不会出现"窗口一个数、报告另一个数"。
    """
    src_desc: str
    timestamp: datetime
    ratio: float
    ratio_src: str
    outer_mm: float
    outer_px: float
    outer_ellipse: tuple
    center_hole_mm: Optional[float] = None
    center_hole_px: Optional[float] = None
    center_hole_ellipse: Optional[tuple] = None
    bolts: list = field(default_factory=list)      # [{mm,px,a1,a2,cx,cy,status,ok}]
    tilt: float = 1.0
    touches_border: bool = False
    overall: str = "OK"
    overall_judged: bool = False                   # 没给标称值时无 OK/NG 可言
    warnings: list = field(default_factory=list)   # 给 GUI 的通俗提醒
    lines: list = field(default_factory=list)      # 文本报告（与命令行输出一致）
    debug_jpg: Optional[Path] = None
    report_txt: Optional[Path] = None
    debug_img: Optional[np.ndarray] = None         # 画了轮廓的图（GUI 缩略图用）


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。GUI 复用同一套，保证拖照片/标定两种用法参数一致。"""
    p = argparse.ArgumentParser(description="圆环类零件尺寸测量")
    p.add_argument("--image", help="测已有照片；不指定则相机现拍")
    p.add_argument("--ref-mm", type=float, help="零件外径真值(mm)，用于建立换算系数")
    p.add_argument("--mm-per-px", type=float, help="直接指定 mm/px")
    p.add_argument("--spec-od", type=float, help="外径标称值(mm)，用于 OK/NG")
    p.add_argument("--spec-id", type=float, help="中心孔标称值(mm)，用于 OK/NG")
    p.add_argument("--spec-bolt", type=float, help="螺栓孔标称值(mm)，用于 OK/NG")
    p.add_argument("--tol", type=float, default=0.1, help="公差 ±mm，默认 0.1")
    return p


def grab_image(args) -> np.ndarray:
    """从相机拍一张，或从文件读一张，返回 BGR 图像。"""
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise MeasureError(f"无法读取图片: {args.image}")
        return img
    print("打开相机拍照...")
    with create_camera(config.CAMERA_TYPE, **config.CAMERA_KWARGS) as cam:
        frame = cam.capture()
    out = Path("data/captures") / "measure_ring_latest.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"已保存: {out}")
    return frame


def resolve_ratio(args, outer_dia_px) -> tuple:
    """确定 mm/px：--mm-per-px 直接用；--ref-mm 用外径真值算并缓存；否则读缓存。"""
    if args.mm_per_px:
        return args.mm_per_px, "命令行指定"
    if args.ref_mm:
        ratio = args.ref_mm / outer_dia_px
        RATIO_CACHE.parent.mkdir(parents=True, exist_ok=True)
        RATIO_CACHE.write_text(json.dumps({"mm_per_pixel": ratio}))
        return ratio, f"本次标定（外径={args.ref_mm}mm），已缓存到 {RATIO_CACHE}"
    if RATIO_CACHE.exists():
        return json.loads(RATIO_CACHE.read_text())["mm_per_pixel"], f"缓存 {RATIO_CACHE}"
    raise MeasureError(
        "还没有换算系数，先卡尺量出零件外径，然后运行：\n"
        "    python src\\measure_ring.py --ref-mm <外径真值mm>"
    )


def measure(img: np.ndarray, args) -> RingResult:
    """检测 → 换算 → 判定 → 落盘（调试图 + 文字报告），返回结构化结果。

    失败的测量（含贴边等）不一定抛异常，但提示会进 warnings/lines。
    """
    res = detect(img)
    if res["outer"] is None:
        raise MeasureError("外圈拟合失败。检查：对焦、光照、零件是否在画面中心")

    lines: list = []
    warnings: list = []

    def _p(s=""):
        lines.append(s)

    timestamp = datetime.now()
    src_desc = args.image or "相机现拍"

    (outer, n_o) = res["outer"]
    (ocx, ocy), (oa1, oa2), oang = outer
    outer_dia_px = (oa1 + oa2) / 2
    ratio, ratio_src = resolve_ratio(args, outer_dia_px)

    touches_border = (oa1 > img.shape[1] - BORDER_MARGIN * 2 or
                      oa2 > img.shape[0] - BORDER_MARGIN * 2)
    if touches_border:
        _p("⚠️  警告：零件轮廓贴到图像边缘，零件没拍全，测量结果不可信！")
        _p("    把相机架高一点（25mm 镜头：视野宽 ≈ 距离 × 0.3）再重拍。")
        warnings.append("零件超出画面：轮廓贴到图像边缘，结果不可信，请把相机架高重拍")

    od_mm = outer_dia_px * ratio
    _p("\n===== 测量报告 =====")
    _p(f"时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}    图像: {src_desc}")
    _p(f"换算系数: {ratio:.6f} mm/px （来源: {ratio_src}）")
    _p(f"外径:   {od_mm:.3f} mm  ({outer_dia_px:.1f} px, 椭圆 {oa1:.1f}x{oa2:.1f}, 角 {oang:.1f}°)")

    tilt = max(oa1, oa2) / min(oa1, oa2)
    if tilt > 1.02:
        tilt_deg = np.degrees(np.arccos(min(oa1, oa2) / max(oa1, oa2)))
        _p(f"⚠️  透视提醒：外圈椭圆长短轴比 {tilt:.3f}（相当于斜视 {tilt_deg:.0f}°），"
           "各尺寸的毫米读数带位置相关误差；把相机垂直朝下正对零件可消除")
        warnings.append(f"相机斜视约 {tilt_deg:.0f}°（外圈椭圆比 {tilt:.3f}）：各读数带位置相关误差，"
                        "把相机垂直朝下可消除")

    overall = "OK"
    overall_judged = False
    if args.spec_od:
        overall_judged = True
        dev = od_mm - args.spec_od
        ok = abs(dev) <= args.tol
        overall = "OK" if ok else "NG"
        _p(f"        标称 {args.spec_od} ±{args.tol}  偏差 {dev:+.3f}  -> {'OK' if ok else 'NG'}")

    # ---- 中心孔 ----
    if res["center_hole"] is None or res["center_hole"][0] is None:
        hole = None
        hole_mm = hole_px = None
        _p("中心孔: 未检测到（若零件有孔，检查孔内是否反光/遮挡）")
        warnings.append("中心孔没测到：检查孔内是否反光或被遮挡")
    else:
        (hole, n_h) = res["center_hole"]
        (hcx, hcy), (ha1, ha2), hang = hole
        hole_px = (ha1 + ha2) / 2
        hole_mm = hole_px * ratio
        _p(f"中心孔: {hole_mm:.3f} mm  ({hole_px:.1f} px, 椭圆 {ha1:.1f}x{ha2:.1f}, 角 {hang:.1f}°)")
        if args.spec_id:
            overall_judged = True
            dev = hole_mm - args.spec_id
            ok = abs(dev) <= args.tol
            if not ok:
                overall = "NG"
            _p(f"        标称 {args.spec_id} ±{args.tol}  偏差 {dev:+.3f}  -> {'OK' if ok else 'NG'}")

    # ---- 螺栓孔 ----
    bolts_out = []
    if res["bolts"]:
        for i, b in enumerate(res["bolts"], 1):
            if b["status"] != "OK":
                _p(f"螺栓孔{i}: 直径测量失败 @({b['cx']:.0f},{b['cy']:.0f})")
                warnings.append(f"螺栓孔{i} 直径测量失败")
                bolts_out.append({"idx": i, "mm": None, "px": None, "ok": None,
                                  "status": b["status"], "cx": b["cx"], "cy": b["cy"]})
                continue
            b_mm = b["dia"] * ratio
            _p(f"螺栓孔{i}: {b_mm:.3f} mm  ({b['dia']:.1f} px, 椭圆 {b['a1']:.1f}x{b['a2']:.1f}, "
               f"位置({b['cx']:.0f},{b['cy']:.0f}))")
            b_ok = None
            if args.spec_bolt:
                overall_judged = True
                dev = b_mm - args.spec_bolt
                b_ok = abs(dev) <= args.tol
                if not b_ok:
                    overall = "NG"
                _p(f"        标称 {args.spec_bolt} ±{args.tol}  偏差 {dev:+.3f}  -> {'OK' if b_ok else 'NG'}")
            bolts_out.append({"idx": i, "mm": b_mm, "px": b["dia"], "ok": b_ok,
                              "status": b["status"], "a1": b["a1"], "a2": b["a2"],
                              "cx": b["cx"], "cy": b["cy"], "ang": b["ang"]})
        _p("提示: 螺栓孔按沉孔口(norm<0.72 阈值交叉)定义；已用游程宽度过滤+渐缩残差处理暗带/槽缘粘连")
    else:
        _p("未检测到螺栓孔")
        warnings.append("没检测到螺栓孔")

    _p(f"综合判定: {overall}")
    _p("提示: 本测量为像素级拟合，单像素量化误差 = %.3f mm；"
       "要冲微米级需远心镜头+背光+亚像素（见方案讨论）" % ratio)

    # ---- 调试图：画出检测到的椭圆 ----
    dbg = img.copy()
    cv2.ellipse(dbg, outer, (0, 255, 0), 8)
    if hole is not None:
        cv2.ellipse(dbg, hole, (0, 0, 255), 8)
    for b in res["bolts"]:
        if b["status"] == "OK":
            cv2.ellipse(dbg, ((b["cx"], b["cy"]), (b["a1"], b["a2"]), b["ang"]), (255, 0, 0), 6)
    DEBUG_DIR.mkdir(exist_ok=True)
    dbg_path = DEBUG_DIR / "measure_ring_debug.jpg"
    cv2.imwrite(str(dbg_path), dbg)
    _p(f"调试图: {dbg_path}")

    # 报告存盘（留痕/追溯）
    rep_path = DEBUG_DIR / f"报告_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    rep_path.write_text("\n".join(lines), encoding="utf-8")

    return RingResult(
        src_desc=src_desc, timestamp=timestamp, ratio=ratio, ratio_src=ratio_src,
        outer_mm=od_mm, outer_px=outer_dia_px, outer_ellipse=outer,
        center_hole_mm=hole_mm, center_hole_px=hole_px, center_hole_ellipse=hole,
        bolts=bolts_out, tilt=tilt, touches_border=touches_border,
        overall=overall, overall_judged=overall_judged, warnings=warnings,
        lines=lines, debug_jpg=dbg_path, report_txt=rep_path, debug_img=dbg,
    )


def main():
    args = build_parser().parse_args()
    try:
        img = grab_image(args)
        result = measure(img, args)
    except MeasureError as e:
        raise SystemExit(str(e)) from None
    for ln in result.lines:
        print(ln)
    print(f"报告已存: {result.report_txt}")


if __name__ == "__main__":
    main()
