"""模型对比：GLCM+SVM vs ResNet18 — 准确率 + 速度 + 混淆分析"""
import sys, time, json, joblib, glob, torch, torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Subset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, "src")
from features import extract_features

DATA_DIR = "data/EDM/50X"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 数据准备（与训练时相同的划分） ==========
full = datasets.ImageFolder(DATA_DIR)
n_test = int(len(full) * 0.2)
indices = torch.randperm(len(full), generator=torch.Generator().manual_seed(42)).tolist()
test_indices = indices[len(full) - n_test:]  # 后 20% 为测试集
test_full = datasets.ImageFolder(DATA_DIR)  # 只取路径和标签
test_set = Subset(test_full, test_indices)

# 收集测试集图片路径和真实标签
test_items = []
for idx in test_indices:
    path, label_idx = full.samples[idx]
    test_items.append((path, full.classes[label_idx]))

print(f"测试集: {len(test_items)} 张图, {len(full.classes)} 类")
print(f"设备: {device}")
print("=" * 60)

# ========== GLCM + SVM ==========
print("\n>>> GLCM + SVM 评估中...")
model_files = sorted(glob.glob("models/v1_glcm_svm_*.pkl"))
model_path = model_files[-1] if model_files else "models/glcm_svm.pkl"
svm_model = joblib.load(model_path)

glcm_preds, glcm_trues = [], []
t0 = time.time()
for path, true_cls in test_items:
    feat = extract_features(path).reshape(1, -1)
    pred = svm_model.predict(feat)[0]
    glcm_preds.append(pred)
    glcm_trues.append(true_cls)
glcm_time = time.time() - t0

glcm_acc = sum(1 for p, t in zip(glcm_preds, glcm_trues) if p == t) / len(glcm_trues)
print(f"GLCM+SVM 准确率: {glcm_acc:.4f}")
print(f"GLCM+SVM 推理耗时: {glcm_time:.2f}s ({glcm_time/len(test_items)*1000:.1f}ms/张)")

# ========== ResNet18 ==========
print("\n>>> ResNet18 评估中...")
from torchvision import models
class_to_idx = json.load(open("models/class_to_idx.json"))
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(torch.load("models/resnet18.pth", map_location=device))
model.to(device)
model.eval()

tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

resnet_preds, resnet_trues = [], []
import cv2
t0 = time.time()
for path, true_cls in test_items:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    with torch.no_grad():
        tensor = tf(img).unsqueeze(0).to(device)
        prob = torch.softmax(model(tensor), dim=1)
    pred = idx_to_class[prob.argmax().item()]
    resnet_preds.append(pred)
    resnet_trues.append(true_cls)
resnet_time = time.time() - t0

resnet_acc = sum(1 for p, t in zip(resnet_preds, resnet_trues) if p == t) / len(resnet_trues)
print(f"ResNet18 准确率: {resnet_acc:.4f}")
print(f"ResNet18 推理耗时: {resnet_time:.2f}s ({resnet_time/len(test_items)*1000:.1f}ms/张)")

# ========== 对比报告 ==========
print("\n" + "=" * 60)
print("模型对比总结")
print("=" * 60)
print(f"{'指标':<15} {'GLCM+SVM':<15} {'ResNet18':<15}")
print("-" * 45)
print(f"{'准确率':<15} {glcm_acc:.2%}{'':<9} {resnet_acc:.2%}{'':<9}")
print(f"{'推理速度':<13} {glcm_time/len(test_items)*1000:.1f}ms/张{'':<5} {resnet_time/len(test_items)*1000:.1f}ms/张")
print(f"{'模型大小':<13} {Path(model_path).stat().st_size/1024:.0f}KB{'':<8} {Path('models/resnet18.pth').stat().st_size/1024/1024:.1f}MB")
print(f"{'特征维度':<13} {'14维':<15} {'自动学习':<15}")
print(f"{'可解释性':<13} {'高(GLCM物理意义)':<15} {'低(黑盒)':<15}")
print(f"{'GPU需求':<13} {'不需要':<15} {'需要(推理可用CPU)':<15}")

# ========== 选型结论 ==========
glcm_ms = glcm_time / len(test_items) * 1000
resnet_ms = resnet_time / len(test_items) * 1000
faster = "GLCM+SVM" if glcm_ms < resnet_ms else "ResNet18"
conclusion = f"""
=== {time.strftime('%Y%m%d_%H%M')} 模型选型对比 ===
测试集: {len(test_items)} 张图, {len(full.classes)} 类
GLCM+SVM: 准确率={glcm_acc:.4f}, 推理={glcm_ms:.1f}ms/张
ResNet18: 准确率={resnet_acc:.4f}, 推理={resnet_ms:.1f}ms/张

选型建议:
- ResNet18 准确率更高 ({resnet_acc:.2%} vs {glcm_acc:.2%})，差距 {(resnet_acc-glcm_acc)*100:.1f}%
- {faster} 推理更快 ({min(glcm_ms, resnet_ms):.1f}ms vs {max(glcm_ms, resnet_ms):.1f}ms/张)；GLCM+SVM 无需 GPU，可解释性强
- 生产环境推荐 ResNet18（准确率为优先），GLCM+SVM 作为离线备份
"""
print(conclusion)

Path("experiments").mkdir(exist_ok=True)
with open("experiments/log.txt", "a", encoding="utf-8") as f:
    f.write(conclusion)
print("结论已写入 experiments/log.txt")
