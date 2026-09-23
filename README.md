# SurfaceIQ（锐感视觉）

基于机器视觉的**金属圆环/法兰零件尺寸测量系统**。

工业相机拍摄零件 → 亚像素射线边缘检测 + 椭圆鲁棒拟合 → 输出 **外径 / 中心孔 /
6 个螺栓孔** 的毫米读数 + OK/NG 判定。

---

## 快速开始

双击 **`vision-roughness\测量.bat`**：

| 操作 | 效果 |
|---|---|
| 直接双击 | 相机现拍 → 弹出结果窗口 |
| 把照片拖到 `测量.bat` 图标上 | 改测该照片，不拍新图 |
| 窗口内 `F5` | 再测一次 |
| 窗口内 `Esc` | 关闭 |
| 「看大图」/「打开报告」 | 用系统看图器 / 记事本打开产物 |

结果窗口用大字给出各尺寸，带 OK/NG 状态、通俗提醒和轮廓缩略图；
出错时把 `0x80000203` 这类错误码翻译成中文提示，不会出现「双击了没反应」。

每次测量自动在 `vision-roughness\debug\` 存两份记录：

| 文件 | 内容 |
|---|---|
| `measure_ring_debug.jpg` | 画出检测轮廓的图（绿=外圈，红=中心孔，蓝=6 螺栓孔） |
| `报告_年月日_时分秒.txt` | 文字报告（时间 / 图像 / 换算系数 / 全部尺寸 / 判定，UTF-8） |

---

## 首次搭建

```bash
cd vision-roughness

# 1. 建虚拟环境（用 Python 3.11，现有 venv 是 3.11.9）
#    必须用 venv 的 python —— 系统 Python 缺 cv2
py -3.11 -m venv venv

# 2. 装依赖（只有两个包，很快）
venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. 验证
venv\Scripts\python.exe -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

**接海康相机还需要**：

1. 装海康机器人 **MVS** 软件（安装时勾选 Python SDK 组件）；
2. 用 MVS 客户端确认能枚举到相机、能出图（确认完**关掉 MVS**，它会占用相机）；
3. 核对 `src/config.py` 的 `sdk_path` 指向实际的 `MvImport` 目录
   （本机为 `E:\海康sadp\MVS\Development\Samples\Python\MvImport`，非默认路径）；
4. MVS 运行时 DLL 目录由 `camera_adapter.py` 自动补进搜索路径，无需手工配 PATH。

---

## 命令行参考

日常用不到命令行（双击 `测量.bat` 即可），标定和排错时才用。
所有命令在 `vision-roughness/` 目录下执行：

```bash
venv\Scripts\python.exe src\measure_ring.py                        # 相机现拍并测量
venv\Scripts\python.exe src\measure_ring.py --image data\captures\xxx.jpg
venv\Scripts\python.exe src\measure_ring.py --ref-mm 88.60          # 用外径真值重建换算系数
```

| 参数 | 说明 |
|---|---|
| `--image <路径>` | 测已有照片；不指定则相机现拍 |
| `--ref-mm <mm>` | 零件**外径真值**，用于建立 mm/px 换算系数并缓存 |
| `--mm-per-px <系数>` | 直接指定换算系数，绕过标定 |
| `--spec-od / --spec-id / --spec-bolt <mm>` | 外径 / 中心孔 / 螺栓孔**标称值**，用于 OK/NG |
| `--tol <mm>` | 公差 ±mm，默认 `0.1` |

> **不传任何 `--spec-*` 时不判定 OK/NG**，窗口显示「已测量」而不是假的 OK。
> 目前没有图纸标称值，等拿到后用上面三个参数开启判定。

---

## 检测管线

核心在 `src/fit_ellipse_ring.py::detect()`：

1. **外圈** —— 72 条射线从外向内找最外侧强下降沿（抛物线插值到亚像素）+ 圆等价
   归一化距离滤离群 + 椭圆鲁棒拟合；再用粗结果做紧窗口精化。
2. **中心孔** —— 同一套射线法找暗→亮的上升沿。
3. **螺栓孔** —— 三步：
   - 粗暗斑挖 250×250 模板 → 1/3 缩比模板匹配（`TM_CCOEFF_NORMED`）取峰；
   - **六孔等角共圆校验**：转到外圈椭圆归一化坐标系，用 RANSAC 思路选相位锚，
     生成 6 个 60° 等角槽位 —— 同时完成「滤假峰」和「补缺失孔」；
   - 逐孔：**螺丝头（暗斑内被包围的亮区）锚定孔心** → 沿射线枚举连续暗游程，
     取第一个**宽度 ≥12px** 的游程末端、按固定亮度阈值交叉定出**沉孔口**
     （跳过 5~15px 的螺丝槽纹/头影，防提前交叉）→ 极化残差**渐缩迭代**椭圆拟合
     （2.5×中位残差 → 1.0×中位，甩开与暗带粘连的远点；轴比 1.0~2.2 防塌陷）。

---

## 实测数据（2026-09-14）

输入 `data/captures/capture_20260908_115557.jpg`（12MP，斜视）；
换算系数 **0.032658 mm/px**（卡尺实测外径 88.60mm 标定，缓存在 `models/mm_per_pixel.json`）。

| 项目 | 像素（椭圆 长×短轴） | mm |
|---|---|---|
| 外径 | 2712.9（2599.1×2826.7） | **88.600** |
| 中心孔 | 1888.1（1849.6×1926.7） | **61.664** |
| 螺栓孔 1 | 154.5 | 5.046 |
| 螺栓孔 2 | 154.9 | 5.059 |
| 螺栓孔 3 | 167.3 | 5.463 |
| 螺栓孔 4 | 163.8 | 5.350 |
| 螺栓孔 5 | 176.3 | 5.759 |
| 螺栓孔 6 | 175.7 | 5.739 |

---

## 已知限制（重要，别误读读数）

- **螺栓孔测的是沉孔口口径**（含沉孔锥度），**不是螺纹孔径**。6 孔读数分散 ±7%
  （154.5~176.3px），含斜视透视（mm/px 随画面位置变化）+ 沉孔口阈值定义差异，
  **不能据此判定尺寸超差**。
- **斜视约 23°**：外圈椭圆长短轴比 1.088，报告会自动打印透视提醒。垂直拍摄后
  mm/px 全视场恒定，标定一次通用。
- **当前机位达不到 5 丝（±0.05mm）**：单像素 ≈ 0.0327mm，5 丝 ≈ 1.5px；而沉孔
  缓坡边缘的实际定位误差 ±3~5px（≈±0.10~0.16mm）。

---

## 下一步：冲 5 丝精度

现状不可达的原因见上。产线**没有底部光照**（背光方案不可用），所以走
**垂直俯视 + 正面光**路线：

1. 先买百元级**低角度环形光**（暗场光），零件平放、相机用气泡水平仪校正到垂直朝下，
   拍样张；
2. 对比样张的边缘梯度锐度，评估亚像素定位能到多少；
3. 不够再依次试同轴光 / 圆顶光；兜底才是远心镜头（上万）或激光轮廓仪（几万）。

**验收判据**：重复装夹拍 10 张，外径重复性 **≥ ±0.05mm** 才算达标。
（用 `src/capture_single.py` 连拍，文件名带时间戳不会互相覆盖。）

---

## 目录结构

```
SurfaceIQ/
├── vision-roughness/
│   ├── 测量.bat                 ★ 日常入口（双击；内部启动 gui_ring.py）
│   ├── src/
│   │   ├── gui_ring.py          结果窗口（tkinter 大字显示 + 缩略图 + 中文报错翻译）
│   │   ├── measure_ring.py      测量主逻辑（CLI 与 GUI 共用同一份结果）
│   │   ├── fit_ellipse_ring.py  检测管线（射线亚像素边缘 + 椭圆鲁棒拟合）
│   │   ├── camera_adapter.py    相机抽象层（webcam / file / hikvision）
│   │   ├── config.py            相机配置 —— 换相机唯一改动点
│   │   ├── capture_demo.py      取景 + 空格手动采图（带预览窗口）
│   │   ├── capture_single.py    单张采图（无窗口，连拍不覆盖）
│   │   └── explore_ring.py / profile_ring.py / ray_ring.py / edge_ring.py
│   │                            开发期图像诊断工具（见文末）
│   ├── models/mm_per_pixel.json mm/px 换算系数缓存（换机位要重建）
│   ├── debug/                   每次测量的调试图 + 文字报告 + GUI 日志（未入库）
│   ├── data/captures/           采集的图片（未入库）
│   ├── temp/hik_debug.py        海康取图故障排查脚本
│   ├── requirements.txt
│   └── venv/                    本地虚拟环境（未入库，约 7.8G，主要体积是 torch）
└── data/captures/               标定/验证用标准照片
```

---

## 环境与相机

- Python 3.11（`vision-roughness/venv/`）。**必须用 venv 的 python**，
  系统 Python 缺 cv2：`vision-roughness\venv\Scripts\python.exe`
- 相机：海康 **MV-CU120-10GM**（12MP 4024×3036，1/1.7" IMX226，黑白，卷帘，GigE，C 口）
  \+ 海康 DH25-10MP-23（25mm FA 定焦）；曝光锁 30000μs、增益 0（锁定亮度）
- 供电：DC 12V/1A（端子 橙=电源+，灰=电源−）；网线直连电脑，相机 IP 192.168.0.1
- MVS 装在 `E:\海康sadp\MVS\`（**SDK 路径非默认**，见 `src/config.py`）

### 踩过的坑

- GigE 包大小必须设 **1500 字节**（`GevSCPSPacketSize`）；网卡虚报巨型帧（8164）会丢包
- 12MP 一帧在 1500 包下约 1 秒，取图超时设 3000ms
- `测量.bat` 内容必须是 **ASCII + CRLF**；存成 UTF-8 或 LF 会导致 cmd 乱码报错

### 排错速查

| 现象 | 原因 | 处理 |
|---|---|---|
| 报错码 `0x80000203` | 相机被 MVS 客户端占用 | 关掉 MVS 软件，再点「再测一次」 |
| 「没找到相机」 | 网线 / 供电 / 网段 | 查网线与相机指示灯，确认网卡与相机同网段 |
| 「能看见但取不到图」 | 占用 / 网线松动 / IP 冲突 | 关 MVS；查是否有两台设备同 IP |
| 取图超时 `0x80000007` | 同上 | 跑 `venv\Scripts\python.exe temp\hik_debug.py` 逐环节定位 |
| 「没找到海康 SDK」 | MVS 未装完整或路径不符 | 重装 MVS（勾选 Python SDK），或改 `config.py` 的 `sdk_path` |
| `No module named cv2` | 用了系统 Python | 改用 `venv\Scripts\python.exe` |
| 窗口里数字都是「—」 | 检测失败 | 看窗口红底提示；检查对焦、光照、零件是否在画面中心 |

---

## 换相机 / 重新标定

1. 改 `src/config.py` 的 `CAMERA_TYPE` / `CAMERA_KWARGS`（业务代码零改动）；
2. 卡尺量出零件外径真值，重建 mm/px 换算系数：

   ```bash
   venv\Scripts\python.exe src\measure_ring.py --ref-mm <外径真值mm>
   ```

3. 换算系数跟着「相机 + 镜头 + 拍摄距离」走，**三者任一变化都必须重建**。

---

## 诊断工具

改检测参数时按这个顺序用，均为 `venv\Scripts\python.exe src\<脚本>.py <图片路径>`：

| 脚本 | 用途 |
|---|---|
| `explore_ring.py` | 摸清外圈/中心孔/螺栓孔/背景的灰度分布，为阈值定参 |
| `profile_ring.py` | 径向距离直方图找半径峰，定位圆心与外圈直径 |
| `ray_ring.py` | 打印指定中心 8 个方向的边缘列表（距离/梯度/灰值） |
| `edge_ring.py` | Canny+轮廓 与 缩图 HoughCircles 的路线对比 |
