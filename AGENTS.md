# AGENTS.md —— 给后续 AI 助手的仓库须知

本文件只写 README.md 里**没有、写错、或容易被误读**的东西。快速开始、排错速查表、
实测数据表、5 丝精度路线都在 README，不重复，需要时指过去即可。

---

## 0. 开工前先问（硬约束，优先级高于本文件其它条目）

**加任何代码之前，先回答一个问题：「我要加的东西，是不是某个现有文件本来就该有的？」**
答案是「是」就改那个文件——但**改动方案要先报给用户，等他确认再动手**。

- **不许自己另起新文件。** 需要新脚本 / 新窗口 / 新模块时，先把方案和理由说给用户，
  用户明确同意后再写。确有必要的，也要先说清楚**为什么不能改现有的那个文件**。
- **「现有文件是车间在用的入口」不是免改理由。** 该改就改，但改法要先让用户拍板。
- **动手前先读现有实现**，确认要加的能力是不是已经被别的文件实现了——`gui_ring.py` /
  `capture_demo.py` 这类并列工具很容易功能重叠，先读完再判断。
- 用户否决、或你没把握时：**停下来问，不要先写完再解释。**

**反面教材（真实发生过，2026-10-07）**：用户要「实时画面 + 数据同屏」。仓库里已经有
`src/gui_ring.py`（结果窗口，右画布贴的是**测完之后的静态轮廓图**，本来就没有实时画面）。
AI 自行判定「gui_ring.py 是车间在用的稳定入口、不该动」，于是另写了 `src/gui_live.py`。后果：

- 两个窗口各写了一遍数字渲染（外径 / 中心孔 / 6 孔 Label、提醒区、报错渲染）→
  以后改判定颜色只改一边就会漂移；
- `gui_live.py` 反过来 `from gui_ring import ...` → **依赖方向反了**，动旧文件会弄坏新文件；
- 正确做法：在 `gui_ring.py` 上加一块实时画面（适配器本身已在连续采集，循环调
  `cam.capture()` 即可，实测 12MP ≈ 7.7 fps，够实时），**或者先问用户「加还是新建」**。
- **结局**：用户要求「把 gui_live 的功能给 gui_ring」，`src/gui_live.py` 已删除，实时取景并入了
  `gui_ring.py`，`测量.bat` 一个字都不用改。**方向本来就该是改旧的——代价是一次白做的返工，
  外加一段"依赖方向反了"的弯路。**

---

## 1. 项目一句话 + 当前边界

金属圆环/法兰零件的视觉尺寸测量：相机拍图 → 亚像素射线边缘 + 椭圆鲁棒拟合 →
输出外径 / 中心孔 / 6 个螺栓孔的毫米读数。

- **唯一在跑的业务线就是圆环尺寸测量。** commit `7375601` 已删除 EDM 粗糙度分类线与
  旧棋盘格标定路线，**不要以任何理由重新引入**（README 标题里的「锐感」、仓库名里的
  "Roughness"、`**/models/*.pth` 的 gitignore 条目都是历史残留，不是待恢复的功能）。
- 实际依赖只有 `opencv-python` + `numpy`（见 `vision-roughness/requirements.txt`）。

## 2. 运行环境硬约束

- **解释器**：必须 `vision-roughness\venv\Scripts\python.exe`（已实测 3.11.9 / cv2 5.0.0 /
  numpy 2.4.6）。系统 Python 没有 cv2。
- **cwd 必须是 `vision-roughness/`**：`models/`、`debug/`、`data/captures` 全是相对 cwd 的
  `Path`（`src/measure_ring.py:40-41`），且 `measure_ring.py:35` 用 `sys.path.insert(0,"src")`
  找同目录模块。
- **误导性报错（已实测，重点）**：在仓库根执行
  `vision-roughness\venv\Scripts\python.exe vision-roughness\src\measure_ring.py --image data/captures/capture_20260908_115557.jpg`
  图片能读到、检测也跑了，最后却报「还没有换算系数…」。真实原因是 cwd 在仓库根，
  读不到 `vision-roughness/models/mm_per_pixel.json`——**缓存其实存在**。
  看到这条报错先查 cwd，不要去重建标定。
- GUI 启动时会 `os.chdir(PROJECT_ROOT)`（`src/gui_ring.py:655`），所以 GUI 不受 cwd 影响，
  CLI 受影响。这就是「双击 `测量.bat` 能用、命令行手打却报错」的原因。
- GUI 需要显示器和相机，**AI 不要跑 `gui_ring.py` / `测量.bat`**；一切验证走 CLI。

## 3. 验收合约

**项目没有测试框架**（无 pytest / unittest / `def test_`）。下面这条命令是唯一的回归手段，
改动检测相关代码后必须跑：

```bash
cd vision-roughness
venv\Scripts\python.exe src\measure_ring.py --image ..\data\captures\capture_20260908_115557.jpg
```

基线（2026-09-23 已复现，与 README「实测数据」一致）：

- 外径 **88.600 mm**，椭圆 **2599.1x2826.7 px**（角 170.2°）
- 中心孔 **61.664 mm**（1849.6x1926.7）
- 6 螺栓孔 **5.046 / 5.059 / 5.463 / 5.350 / 5.759 / 5.739 mm**
- 椭圆长短轴比 1.088（斜视约 23°），换算系数 0.032658 mm/px

**路径陷阱**：标定照片在【仓库根】的 `data/captures/`，从 `vision-roughness/` 下必须写
`..\data\captures\...`。README「命令行参考」里 `--image data\captures\xxx.jpg` 指向的是
`vision-roughness/data/captures/`（另一个目录），照抄会报「无法读取图片」（已实测）。

**哪个数才是真信号**：检测几何的真回归信号是**椭圆像素轴长**（2599.1x2826.7 等）。
mm 读数 = 像素 × `models/mm_per_pixel.json` 的系数，改这个 json 会让所有 mm 读数整体等比
缩放、而椭圆像素不变。判断「检测有没有被改坏」看像素；判断「标定有没有被动」看系数。

## 4. 不可动 / 不入库 / 明确禁止

- **`models/mm_per_pixel.json` 不要改。** 它是「相机 + 镜头 + 拍摄距离」的函数，只有换相机 /
  镜头 / 机位才用 `--ref-mm` 重建（`measure_ring.py:108-122` 会覆写它）。**不要为了「让读数
  对上标称值」去调它**——那是伪造数据。
- **不入库**（`.gitignore`）：`debug/`、`vision-roughness/data/captures/*`（仅保留
  `capture_single_20260730_203312.jpg`）、`venv/`、`.zcode/`、`**/temp/*`（仅保留
  `hik_debug.py`）、`**/models/*.pth`。新增运行时产物放这些目录里，不要 `git add`。
- `测量.bat` 内容必须 **ASCII + CRLF**（内部走 `pythonw` + `gui_ring.py`）。存成 UTF-8 或 LF
  会让 cmd 报错。
- 不要重新引入 `7375601` 删掉的粗糙度分类线 / 棋盘格标定路线。
- 不要新增第三方依赖（tkinter 是标准库）。

## 5. 领域陷阱（按这些结论行事，别自己推）

- **螺栓孔测的是沉孔口口径**（含沉孔锥度；`measure_ring.py:222`、
  `fit_ellipse_ring.py:293 rim_cross_points` 的 norm<0.72 阈值交叉），**不是螺纹孔径**。
  6 孔读数分散 154.5~176.3px ≈ ±7%（含斜视透视 + 沉孔口定义差异），**不得据此判定尺寸超差**。
- **不传 `--spec-*` 时 `overall_judged=False`**，GUI 显示「已测量」（`gui_ring.py:550-553`），
  但 CLI 仍会打印「综合判定: OK」（`measure_ring.py:227`）。**这个 OK 不是合格判定**，只是
  「未判定」时的默认串。要真判定必须给 `--spec-od` / `--spec-id` / `--spec-bolt`。
- **精度天花板**：单像素 ≈0.0327mm，5 丝(±0.05mm) ≈1.5px；当前机位（斜视 23°、沉孔缓坡
  边缘定位误差 ±3~5px）**达不到**。不要承诺 ±0.05mm，也不要用这套读数做超差判据。
  精度路线见 README，不要靠偷偷改检测算法去「凑精度」。
- **`cv2.imread` 传中文路径返回 None**（已实测：OpenCV 只往 stderr 打一行 WARN，不抛异常），
  而本项目报告文件名（`debug\报告_*.txt`）就是中文。新的读图代码必须对 None 判空并给中文提示。
- 斜视 23° 意味着 mm/px 随画面位置变化，**同一零件上不同位置的读数不可直接互比**。

## 6. 代码地图（改什么去哪改）

- **换相机 / 曝光 / 增益**：只改 `src/config.py` 的 `CAMERA_TYPE` + `CAMERA_KWARGS`——这是
  换相机的唯一改动点，业务代码不许动。
- **相机接入抽象层**：`src/camera_adapter.py`（`webcam` / `file` / `hikvision` 三实现，
  `create_camera()` 在 `:327`），海康 DLL 搜索路径由它自动补。
- **检测管线核心**：`src/fit_ellipse_ring.py` 的 `detect()`（`:408`，返回
  `{"outer","center_hole","bolts"}`）。几何算法都改这里。
- **结果唯一来源**：`src/measure_ring.py` 的 `measure()`（`:125`）产出 `RingResult`（`:49`）。
  CLI 打印它，GUI 显示它（`gui_ring.py:245` 直接调 `measure()`）——**不要在 `gui_ring.py` 里
  自己再算一遍尺寸**。
- **显示层**：`src/gui_ring.py` 只做显示/交互，不改数值。它还负责**实时取景**，所以有一条
  额外硬约束：取帧与测量**只在同一个后台线程**（`gui_ring.py::LiveSource`）里做，**绝不能让
  两个线程同时 `capture()`**（相机是独占资源）；tkinter 只能在主线程碰，所以后台线程只往
  队列塞「编码好的 PNG 字节」和事件，主线程再变成 `tk.PhotoImage`。
- **调参前必须先看数据**，按此顺序跑诊断脚本（cwd 同上，传图片路径）：
  `explore_ring.py`（灰度分布、定阈值）→ `profile_ring.py`（径向直方图找圆心/半径）→
  `ray_ring.py`（8 方向边缘列表）→ `edge_ring.py`（Canny+轮廓 vs HoughCircles 路线对比）。
  **不要凭直觉改阈值。**
- 采集：`capture_single.py`（无窗口连拍，文件名带时间戳）/ `capture_demo.py`（带预览手动采图）。

## 7. 约定与提交风格

- 中文注释 / docstring / 用户可见输出；报告文件写 UTF-8（`measure_ring.py:246`）。
- `MeasureError`（`measure_ring.py:45`）表示**「可预期、需原样提示给用户」的失败**（读图失败、
  外圈拟合失败等），`main()` 会把它直接转成一句中文退出（`:263-264`）。**代码缺陷不要用
  MeasureError 包**，要让真实 traceback 抛出来。
- 提交信息：**中文 + `type:` 前缀**，对标现有 history（`feat` / `fix` / `refactor` / `docs` /
  `chore`，例如 `refactor: 移除粗糙度分类线与旧尺寸测量路线，仅保留圆环尺寸测量`）。
  一次提交只做一件事。
