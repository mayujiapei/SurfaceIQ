# SurfaceIQ / RoughSense（锐感视觉）

基于机器视觉的 **EDM 工件表面粗糙度分类 + 尺寸检测系统**。

通过工业相机拍摄 EDM（电火花加工）工件表面图像，利用机器学习/深度学习模型完成：

1. **粗糙度分级**：将工件表面纹理分为 16 个等级（对应加工角度 0°~45°，每 3° 一类）
2. **OK/NG 判定**：将 16 分类映射为合格/不合格
3. **尺寸测量**：相机标定 + 边缘检测，测量工件关键尺寸并判定公差

## 目录结构

```
SurfaceIQ/
├── vision-roughness/
│   ├── src/                  # 全部源码
│   │   ├── api.py            # FastAPI 推理服务
│   │   ├── infer.py          # 统一推理入口（单张/批量，ResNet 或 GLCM+SVM）
│   │   ├── train_resnet.py   # ResNet18 迁移学习训练
│   │   ├── train_glcm_svm.py # GLCM+SVM 传统方法训练
│   │   ├── compare_models.py # 模型对比选型
│   │   ├── features.py       # GLCM 纹理特征提取
│   │   ├── calibrate.py      # 棋盘格相机标定
│   │   ├── measure.py        # 尺寸测量（标定→边缘→最小外接矩形→毫米）
│   │   ├── explore.py        # 数据探索与可视化
│   │   └── capture_*.py      # 相机采集适配（单张/演示/标定）
│   ├── models/               # 轻量模型文件（class_to_idx.json、SVM pkl）
│   ├── data/                 # 标定板、示例采集图（训练集不在此仓库）
│   ├── colab_train_resnet.ipynb  # Colab 训练笔记本
│   ├── requirements.txt
│   └── PROJECT_STATUS.md     # 项目状态报告
├── 视觉粗糙度尺寸检测_开发指南.md  # 开发指南
├── 项目能力清单.md                 # 能力清单
└── README.md
```

## 快速开始

```bash
cd vision-roughness
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 单张推理（GLCM+SVM 模型，无需 GPU）
python src/infer.py data/captures/capture_single_20260730_203312.jpg --model glcm

# 启动 FastAPI 服务
python src/api.py
```

## 模型说明

当前选型结果（详见 `vision-roughness/PROJECT_STATUS.md`）：

| 模型 | 验证准确率 | 大小 | 说明 |
|---|---|---|---|
| ResNet18 迁移学习（主模型） | **95.60%** | 42.7 MB | ImageNet 预训练，微调全连接层 |
| GLCM + SVM（备份模型） | 89.93% | 586 KB | 传统纹理特征 + 支持向量机，可解释性好 |

> ⚠️ **大权重文件不在本仓库中**（`models/*.pth` 已由 `.gitignore` 排除）。
> 重新训练方式：
> - ResNet18：`python src/train_resnet.py` 或使用 `colab_train_resnet.ipynb`（Colab GPU 环境）
> - GLCM+SVM：`python src/train_glcm_svm.py`（首次会提取 GLCM 特征并缓存到 `cache/`）

## 数据集说明

> ⚠️ **训练图片不在本仓库中**（`data/EDM/` 与根目录 `Cropped*` 共约 2.4 万张、约 530 MB，已由 `.gitignore` 排除）。

当前训练数据来自 Kaggle 公开数据集 *Surface Roughness Classification*（EDM 表面图像），按放大倍数与类别组织：

```
data/EDM/
├── 50X/class_01/ ... class_16/   # 4320 张（主训练集）
├── 100X/class_01/ ... class_16/  # 3840 张
└── 200X/class_01/ ... class_16/  # 3840 张
```

其中 `class_01` ~ `class_16` 对应加工角度 0°~45°（每 3° 一类）。自建数据集（真实工件 OK/NG 图）待工业相机与标准样块到位后采集，按 `data/my_parts/OK/`、`data/my_parts/NG/` 组织。

## 尺寸测量

- 标定：`python src/calibrate.py`（使用 `data/calibration/checkerboard_9x6.png`，输出内参 `K` 与畸变系数 `dist`）
- 测量：`python src/measure.py`（去畸变 → 高斯模糊 → Canny 边缘 → 轮廓 → 最小外接矩形 → 像素转毫米 → 公差判定）
- 当前状态：代码骨架完成，待实测 `MM_PER_PIXEL` 后投入使用

## 技术栈

Python 3.11 · OpenCV · PyTorch 2.8 · scikit-learn · FastAPI

## 更多文档

- [视觉粗糙度尺寸检测_开发指南.md](视觉粗糙度尺寸检测_开发指南.md)
- [项目能力清单.md](项目能力清单.md)
- [vision-roughness/PROJECT_STATUS.md](vision-roughness/PROJECT_STATUS.md)
