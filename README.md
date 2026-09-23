# SurfaceIQ（锐感视觉）

基于机器视觉的**金属圆环/法兰零件尺寸测量系统**。

工业相机拍摄零件 → 亚像素射线边缘检测 + 椭圆鲁棒拟合 → 输出 **外径 / 中心孔 /
6 个螺栓孔** 的毫米读数 + OK/NG 判定。

---

## 快速开始

双击 **`vision-roughness\测量.bat`**：

- **相机已接** → 自动拍照，弹出结果窗口（大字显示各尺寸 + 轮廓缩略图 + 提醒）
- **想测已有照片** → 把照片文件拖到 `测量.bat` 图标上
- 窗口内：`F5` 再测一次，`Esc` 关闭，「看大图」「打开报告」一键打开产物

每次测量自动在 `vision-roughness\debug\` 存两份记录：

| 文件 | 内容 |
|---|---|
| `measure_ring_debug.jpg` | 画出检测轮廓的图（绿=外圈，红=中心孔，蓝=6 螺栓孔） |
| `报告_年月日_时分秒.txt` | 文字报告（时间/图像/换算系数/全部尺寸/判定，UTF-8） |

命令行等价用法（标定、排错时用，在 `vision-roughness/` 目录下）：

```bash
venv\Scripts\python.exe src\measure_ring.py --image data\captures\xxx.jpg
venv\Scripts\python.exe src\measure_ring.py --ref-mm 88.60   # 重建 mm/px 换算系数
```

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
- **精度提升路线**：产线没有底部光照（背光不可用）→ 走 **垂直俯视 + 正面光**：
  先试低角度环形光（暗场光），不够再同轴光/圆顶光，兜底才是远心镜头。
  验收判据：重复装夹拍 10 张，外径重复性 ≥ ±0.05mm。

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
│   ├── models/mm_per_pixel.json mm/px 换算系数缓存
│   ├── debug/                   每次测量的调试图 + 文字报告 + GUI 错误日志
│   ├── data/captures/           采集的图片
│   ├── temp/hik_debug.py        海康取图故障排查脚本
│   └── requirements.txt
└── data/captures/               标定/验证用标准照片
```

---

## 环境与相机

- Python 3.11（`vision-roughness/venv/`）。**必须用 venv 的 python**，
  系统 Python 缺 cv2：`vision-roughness\venv\Scripts\python.exe`
- 相机：海康 **MV-CU120-10GM**（12MP 4024×3036，1/1.7" IMX226，黑白，卷帘，GigE，C 口）
  \+ 海康 DH25-10MP-23（25mm FA 定焦）；曝光锁 30000μs、增益 0（锁定亮度）
- MVS 软件装在 `E:\海康sadp\MVS\`（**SDK 路径非默认**，见 `src/config.py`）
- 取图故障排查：`venv\Scripts\python.exe temp\hik_debug.py`
  （逐环节打印 SDK 返回码，含 PayloadSize）

### 踩过的坑

- GigE 包大小必须设 **1500 字节**（`GevSCPSPacketSize`）；网卡虚报巨型帧（8164）会丢包
- 12MP 一帧在 1500 包下约 1 秒，取图超时设 3000ms
- **相机被 MVS 客户端占用会报 `0x80000203`** → 先关掉 MVS
- `测量.bat` 内容必须是 **ASCII + CRLF**；存成 UTF-8 或 LF 会导致 cmd 乱码报错

---

## 换相机 / 重新标定

1. 改 `src/config.py` 的 `CAMERA_TYPE` / `CAMERA_KWARGS`（业务代码零改动）；
2. 卡尺量出零件外径真值，重建 mm/px 换算系数：

   ```bash
   venv\Scripts\python.exe src\measure_ring.py --ref-mm <外径真值mm>
   ```

3. 换算系数跟着「相机 + 镜头 + 拍摄距离」走，三者任一变化都必须重建。
   拿到图纸标称值后，可用 `--spec-od / --spec-id / --spec-bolt / --tol` 开 OK/NG 判定。

---

## 诊断工具

改检测参数时按这个顺序用：

| 脚本 | 用途 |
|---|---|
| `explore_ring.py` | 摸清外圈/中心孔/螺栓孔/背景的灰度分布，为阈值定参 |
| `profile_ring.py` | 径向距离直方图找半径峰，定位圆心与外圈直径 |
| `ray_ring.py` | 打印指定中心 8 个方向的边缘列表（距离/梯度/灰值） |
| `edge_ring.py` | Canny+轮廓 与 缩图 HoughCircles 的路线对比 |

用法均为 `venv\Scripts\python.exe src\<脚本>.py <图片路径>`。
