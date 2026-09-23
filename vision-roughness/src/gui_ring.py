r"""圆环尺寸测量结果窗口（tkinter，日常双击 测量.bat 就是这个）。

为什么有这个文件：车间里没人愿意读命令行里的一堆文字。窗口直接大字给出
外径 / 中心孔 / 6 个螺栓孔，带 OK/NG 状态、通俗提醒和轮廓缩略图。

设计要点：
    - 数字与报告都来自 measure_ring.measure() 的同一份结果，不会"窗口一个数、
      报告另一个数"；
    - 测量在后台线程跑（拍照+检测约 2~4 秒），窗口全程可响应，不假死；
    - 用 pythonw 启动（无控制台窗口），所以所有异常都必须翻译成中文显示在
      窗口里，并写日志，绝不能"双击了没反应"；
    - 沿用 测量.bat 的参数习惯：--image 测照片、不传则相机现拍。
"""
import base64
import os
import queue
import sys
import threading
import traceback
from pathlib import Path

# 项目根目录 = 本文件所在 src/ 的上一级；models/ debug/ data/ 都是相对它的路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _bootstrap_path():
    """保证能 import 到同目录模块，并把 cwd 切到项目根目录（相对路径依赖它）。"""
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


_bootstrap_path()

import cv2  # noqa: E402  (必须在 sys.path 处理之后)
import tkinter as tk  # noqa: E402
from tkinter import messagebox  # noqa: E402

from measure_ring import MeasureError, build_parser, grab_image, measure  # noqa: E402

LOG_PATH = PROJECT_ROOT / "debug" / "gui_error.log"
RUN_LOG = PROJECT_ROOT / "debug" / "gui_run.log"

CIRCLED = "①②③④⑤⑥"
COLOR_BG = "#ffffff"
COLOR_TEXT = "#111827"
COLOR_MUTED = "#6b7280"
COLOR_OK = "#1a7f37"
COLOR_NG = "#c62828"
COLOR_BUSY = "#546e7a"
COLOR_WARN_BG = "#fff6d6"
COLOR_WARN_FG = "#8a5300"
COLOR_ERR_BG = "#fdecea"
COLOR_ERR_FG = "#a01b14"
COLOR_BTN = "#1565c0"
COLOR_BTN_ALT = "#eceff1"


def friendly_error(exc: BaseException) -> str:
    """把底层异常翻译成车间里看得懂的一句话。"""
    msg = str(exc)
    if "0x80000203" in msg:
        return ("相机被别的程序占用了（通常是 MVS 客户端）。\n"
                "请先关闭 MVS 软件，再点「再测一次」。")
    if "找不到海康相机" in msg:
        return ("没找到相机。请检查：网线是否插好、相机电源指示灯是否亮、"
                "网卡 IP 是否和相机同一网段。")
    if "全部无法取图" in msg:
        return ("相机能看见但取不到图像。常见原因：被 MVS 客户端占用、网线松动、IP 冲突。\n"
                "先关掉 MVS，再点「再测一次」。")
    if "取图超时" in msg:
        return "取图超时。请检查网线（相机网口指示灯是否亮）以及是否被其他程序占用。"
    if "MVS Python SDK" in msg:
        return "没找到海康相机 SDK，MVS 软件可能没装完整。请把 debug\\gui_error.log 发给技术支持。"
    if "打开摄像头" in msg:
        return "打不开摄像头，请检查驱动或换个 USB 口。"
    # MeasureError 的文案本身就是给用户看的
    return msg if isinstance(exc, MeasureError) else f"测量失败：{msg}"


def write_log(text: str):
    """把详细错误写进日志文件（窗口里只显示翻译过的一句话）。"""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")
    except OSError:
        pass


def pick_font(root) -> str:
    """挑一个系统里有的中文字体，避免显示成方块。"""
    from tkinter import font as tkfont
    available = set(tkfont.families(root))
    for name in ("Microsoft YaHei UI", "Microsoft YaHei", "微软雅黑",
                 "SimHei", "SimSun", "Segoe UI"):
        if name in available:
            return name
    return "TkDefaultFont"


class MeasureWindow:
    def __init__(self, root: tk.Tk, args):
        self.root = root
        self.args = args
        self.q = queue.Queue()
        self.busy = False
        self.result = None
        self._thumb_ref = None          # 必须持引用，否则图片被回收显示空白
        self.ui = pick_font(root)

        screen_w = root.winfo_screenwidth()
        self.thumb_w = 520 if screen_w >= 1280 else 380
        self.thumb_h = int(self.thumb_w * 400 / 520)

        self._build()
        # 窗口先出现（显示"正在拍照测量"），再去干重活
        root.after(80, self.start)

    # ---------------- 界面搭建 ----------------
    def _build(self):
        root = self.root
        root.title("圆环尺寸测量 - SurfaceIQ")
        root.configure(bg=COLOR_BG)

        # 顶部：标题 / 时间 / 状态徽标
        head = tk.Frame(root, bg=COLOR_BG)
        head.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(head, text="圆环尺寸测量", font=(self.ui, 17, "bold"),
                 bg=COLOR_BG, fg=COLOR_TEXT).pack(side="left")
        self.lbl_time = tk.Label(head, text="", font=(self.ui, 10),
                                 bg=COLOR_BG, fg=COLOR_MUTED)
        self.lbl_time.pack(side="left", padx=12)
        self.lbl_status = tk.Label(head, text="测量中", font=(self.ui, 17, "bold"),
                                   bg=COLOR_BUSY, fg="white", padx=16, pady=2)
        self.lbl_status.pack(side="right")

        body = tk.Frame(root, bg=COLOR_BG)
        body.pack(fill="both", expand=True, padx=18, pady=(8, 0))

        # 左列：数字
        left = tk.Frame(body, bg=COLOR_BG)
        left.pack(side="left", anchor="n")
        self.lbl_od = self._number_row(left, "外径", 44)
        self.lbl_id = self._number_row(left, "中心孔", 32, pady=(6, 10))

        tk.Label(left, text="螺栓孔口径（沉孔口，非螺纹孔径）", font=(self.ui, 11),
                 bg=COLOR_BG, fg=COLOR_MUTED).pack(anchor="w")
        bolts = tk.Frame(left, bg=COLOR_BG)
        bolts.pack(anchor="w", pady=(2, 0))
        self.bolt_labels = []
        for i in range(6):
            lbl = tk.Label(bolts, text=f"{CIRCLED[i]}—", font=(self.ui, 19, "bold"),
                           bg=COLOR_BG, fg=COLOR_MUTED)
            lbl.grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 16), pady=1)
            self.bolt_labels.append(lbl)

        # 右列：轮廓缩略图（固定画布尺寸，避免结果出来时窗口跳动）
        right = tk.Frame(body, bg=COLOR_BG)
        right.pack(side="right", anchor="n", padx=(16, 0))
        self.canvas = tk.Canvas(right, width=self.thumb_w, height=self.thumb_h,
                                bg="#eceff1", highlightthickness=1,
                                highlightbackground="#cfd8dc")
        self.canvas.pack()
        self.canvas_text = self.canvas.create_text(
            self.thumb_w // 2, self.thumb_h // 2, text="正在拍照测量…",
            font=(self.ui, 12), fill=COLOR_MUTED)
        tk.Label(right, text="绿=外圈   红=中心孔   蓝=螺栓孔", font=(self.ui, 9),
                 bg=COLOR_BG, fg=COLOR_MUTED).pack(pady=(4, 0))

        # 提醒 / 报错区（默认不显示，有内容才 pack）
        self.msg_frame = tk.Frame(root, bg=COLOR_WARN_BG)
        self.msg_label = tk.Label(self.msg_frame, text="", font=(self.ui, 11),
                                  bg=COLOR_WARN_BG, fg=COLOR_WARN_FG,
                                  justify="left", anchor="w", wraplength=880,
                                  padx=10, pady=6)
        self.msg_label.pack(fill="x")

        # 按钮
        btns = tk.Frame(root, bg=COLOR_BG)
        btns.pack(fill="x", padx=18, pady=(10, 12))
        again_text = "重新测量 (F5)" if self.args.image else "再测一次 (F5)"
        self.btn_again = self._button(btns, again_text, self.start, primary=True)
        self.btn_again.pack(side="left")
        self._button(btns, "看大图", self.open_debug).pack(side="left", padx=8)
        self._button(btns, "打开报告", self.open_report).pack(side="left")
        self._button(btns, "关闭 (Esc)", self.root.destroy).pack(side="right")

        self.lbl_foot = tk.Label(root, text="", font=(self.ui, 9),
                                 bg=COLOR_BG, fg=COLOR_MUTED, justify="left",
                                 anchor="w")
        self.lbl_foot.pack(fill="x", padx=18, pady=(0, 10))

        root.bind("<Escape>", lambda e: self.root.destroy())
        root.bind("<F5>", lambda e: self.start())
        root.protocol("WM_DELETE_WINDOW", self.root.destroy)

        root.update_idletasks()
        w, h = root.winfo_width(), root.winfo_height()
        x = max(0, (root.winfo_screenwidth() - w) // 2)
        y = max(0, (root.winfo_screenheight() - h) // 3)
        root.geometry(f"+{x}+{y}")
        root.minsize(w, h)

    def _number_row(self, parent, title, size, pady=(0, 10)):
        """一行大数字：小标题 + 特大数字 + 单位。返回数字 Label 供更新。"""
        row = tk.Frame(parent, bg=COLOR_BG)
        row.pack(anchor="w", pady=pady)
        tk.Label(row, text=title, font=(self.ui, 12), bg=COLOR_BG,
                 fg=COLOR_MUTED).pack(anchor="w")
        line = tk.Frame(row, bg=COLOR_BG)
        line.pack(anchor="w")
        val = tk.Label(line, text="—", font=(self.ui, size, "bold"),
                       bg=COLOR_BG, fg=COLOR_TEXT)
        val.pack(side="left")
        tk.Label(line, text=" mm", font=(self.ui, max(10, size * 2 // 5)),
                 bg=COLOR_BG, fg=COLOR_MUTED).pack(side="left", anchor="s",
                                                   pady=(0, size // 4))
        return val

    def _button(self, parent, text, command, primary=False):
        kw = dict(text=text, command=command, font=(self.ui, 12, "bold" if primary else "normal"),
                  relief="flat", padx=18, pady=6, cursor="hand2", takefocus=0)
        if primary:
            kw.update(bg=COLOR_BTN, fg="white", activebackground="#0d47a1",
                      activeforeground="white")
        else:
            kw.update(bg=COLOR_BTN_ALT, fg=COLOR_TEXT, activebackground="#dde3e8")
        return tk.Button(parent, **kw)

    # ---------------- 测量流程（后台线程） ----------------
    def start(self):
        if self.busy:
            return
        self.busy = True
        self.btn_again.config(state="disabled")
        self._set_status("测量中", COLOR_BUSY)
        self.lbl_od.config(text="—", fg=COLOR_TEXT)
        self.lbl_id.config(text="—", fg=COLOR_TEXT)
        for i, lbl in enumerate(self.bolt_labels):
            lbl.config(text=f"{CIRCLED[i]}—", fg=COLOR_MUTED)
        self._show_message(None)
        self.canvas.itemconfig(self.canvas_text, text="正在拍照测量…")
        self.canvas.delete("thumb")
        threading.Thread(target=self._worker, daemon=True).start()
        self.root.after(80, self._poll)

    def _worker(self):
        try:
            img = grab_image(self.args)
            self.q.put(("ok", measure(img, self.args)))
        except BaseException as e:          # noqa: BLE001 线程里必须兜住一切
            self.q.put(("err", e))

    def _poll(self):
        try:
            kind, payload = self.q.get_nowait()
        except queue.Empty:
            self.root.after(80, self._poll)
            return
        self.busy = False
        self.btn_again.config(state="normal")
        if kind == "ok":
            self._render(payload)
        else:
            self._render_error(payload)
        self.root.lift()

    # ---------------- 渲染 ----------------
    def _set_status(self, text, color):
        self.lbl_status.config(text=text, bg=color)

    def _show_message(self, text, kind="warn"):
        """提醒区：kind='warn' 黄底提示，'error' 红底报错，None 隐藏。"""
        if not text:
            self.msg_frame.pack_forget()
            return
        bg, fg = (COLOR_WARN_BG, COLOR_WARN_FG) if kind == "warn" else (COLOR_ERR_BG, COLOR_ERR_FG)
        self.msg_frame.config(bg=bg)
        self.msg_label.config(text=text, bg=bg, fg=fg)
        self.msg_frame.pack(fill="x", padx=18, pady=(10, 0))

    def _render(self, r):
        self.lbl_time.config(text=r.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        if r.overall_judged:
            self._set_status(r.overall, COLOR_OK if r.overall == "OK" else COLOR_NG)
        else:
            self._set_status("已测量", COLOR_BUSY)

        self.lbl_od.config(text=f"{r.outer_mm:.3f}", fg=COLOR_TEXT)
        if r.center_hole_mm is None:
            self.lbl_id.config(text="未测到", fg=COLOR_NG)
        else:
            self.lbl_id.config(text=f"{r.center_hole_mm:.3f}", fg=COLOR_TEXT)

        for i, lbl in enumerate(self.bolt_labels):
            b = r.bolts[i] if i < len(r.bolts) else None
            if b is None or b.get("mm") is None:
                lbl.config(text=f"{CIRCLED[i]}—", fg=COLOR_MUTED)
            else:
                color = COLOR_NG if b.get("ok") is False else COLOR_TEXT
                lbl.config(text=f"{CIRCLED[i]}{b['mm']:.3f}", fg=color)

        if r.warnings:
            self._show_message("；\n".join("⚠ " + w for w in r.warnings), "warn")
        else:
            self._show_message(None)

        self._set_thumb(r.debug_img)

        self.lbl_foot.config(
            text=f"图像: {r.src_desc}    换算系数: {r.ratio:.6f} mm/px（{r.ratio_src}）\n"
                 f"报告: {r.report_txt}")
        self.result = r

    def _render_error(self, exc):
        self._set_status("出错", COLOR_NG)
        self.lbl_od.config(text="—", fg=COLOR_MUTED)
        self.lbl_id.config(text="—", fg=COLOR_MUTED)
        for i, lbl in enumerate(self.bolt_labels):
            lbl.config(text=f"{CIRCLED[i]}—", fg=COLOR_MUTED)
        self.canvas.delete("thumb")
        self.canvas.itemconfig(self.canvas_text, text="这次没测成")
        brief = friendly_error(exc)
        hint = "" if self.args.image else "处理完点「再测一次」即可。"
        self._show_message(f"{brief}\n{hint}".strip(), "error")
        write_log(f"[{_now()}] {self.args.image or '相机现拍'}\n"
                  f"{traceback.format_exc()}\n" + "-" * 60)
        self.lbl_foot.config(text=f"详细日志: {LOG_PATH}")

    def _set_thumb(self, img):
        if img is None:
            return
        h, w = img.shape[:2]
        s = min(self.thumb_w / w, self.thumb_h / h)
        small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", small)
        if not ok:
            return
        self._thumb_ref = tk.PhotoImage(data=base64.b64encode(buf.tobytes()))
        self.canvas.delete("thumb")
        self.canvas.create_image(self.thumb_w // 2, self.thumb_h // 2,
                                 anchor="center", image=self._thumb_ref, tags="thumb")

    # ---------------- 按钮动作 ----------------
    def open_debug(self):
        if self.result is not None and self.result.debug_jpg:
            _open_file(self.result.debug_jpg)

    def open_report(self):
        if self.result is not None and self.result.report_txt:
            _open_file(self.result.report_txt)


def _now():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _open_file(path):
    """用系统默认程序打开文件（Windows: 看图器 / 记事本）。"""
    try:
        os.startfile(str(path))          # noqa: S606  (Windows 专用)
    except OSError as e:
        messagebox.showwarning("打不开文件", f"{path}\n{e}")


def main():
    args = build_parser().parse_args()
    if args.image:                       # 先转绝对路径，再切 cwd
        args.image = str(Path(args.image).expanduser().resolve())
    os.chdir(PROJECT_ROOT)               # models/ debug/ data/ 都是相对项目根目录

    root = tk.Tk()
    try:
        MeasureWindow(root, args)
    except Exception:
        # 连窗口都没起来：弹个系统对话框，别出现"双击了没反应"
        write_log(f"[{_now()}] 窗口初始化失败\n{traceback.format_exc()}")
        try:
            messagebox.showerror("程序启动失败",
                                 "结果窗口没能打开。\n详细信息已记录到：\n"
                                 f"{LOG_PATH}")
        except Exception:
            pass
        raise
    root.mainloop()


if __name__ == "__main__":
    # pythonw 启动时没有控制台，print 会报错；把输出引到日志文件里
    try:
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        _log = RUN_LOG.open("w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = _log
    except OSError:
        pass
    main()
