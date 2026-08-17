"""ResNet18 迁移学习：ImageFolder 加载 → 微调 → 评估 → 保存"""
import json, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

DATA_DIR = "data/EDM/50X"
EPOCHS, BATCH, LR = 30, 32, 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备:", device)

# 1. 数据变换：训练集做数据增强（随机翻转/旋转/亮度抖动），测试集只做规范化
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # 灰度图转3通道适配预训练模型
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 2. 加载数据集并划分：72% 训练 / 8% 验证 / 20% 测试
# 测试集仍取 randperm(seed=42) 的后 20%，与 compare_models.py 的划分保持一致
full = datasets.ImageFolder(DATA_DIR)
n_total = len(full)
n_test = int(n_total * 0.2)
n_val = int(n_total * 0.08)
indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()
train_indices = indices[:n_total - n_test - n_val]
val_indices = indices[n_total - n_test - n_val:n_total - n_test]
test_indices = indices[n_total - n_test:]

train_full = datasets.ImageFolder(DATA_DIR, transform=train_tf)
test_full = datasets.ImageFolder(DATA_DIR, transform=test_tf)
train_set = Subset(train_full, train_indices)
val_set = Subset(test_full, val_indices)
test_set = Subset(test_full, test_indices)

train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, BATCH, num_workers=0)
test_loader = DataLoader(test_set, BATCH, num_workers=0)
num_classes = len(full.classes)
print(f"{len(full)} 张图，{num_classes} 类（训练 {len(train_indices)} / 验证 {len(val_indices)} / 测试 {len(test_indices)}）")

# 3. 模型：预训练 ResNet18，替换最后的全连接层
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

def evaluate(loader):
    model.eval(); correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            pred = model(imgs.to(device)).argmax(1)
            correct += (pred == labels.to(device)).sum().item()
    return correct / len(loader.dataset)

# 4. 训练循环：每个 epoch 在验证集上评估，按验证集准确率保存最优模型
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        loss = criterion(model(imgs), labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    acc = evaluate(val_loader)
    print(f"epoch {epoch+1}/{EPOCHS}  loss={loss.item():.4f}  val_acc={acc:.4f}")
    if acc > best_acc:                      # 保存验证集上最好的模型
        best_acc = acc
        Path("models").mkdir(exist_ok=True)
        torch.save(model.state_dict(), "models/resnet18.pth")
        json.dump(full.class_to_idx, open("models/class_to_idx.json", "w"))

# 5. 加载最优模型，在测试集上做最终评估（只评估一次，不参与选模）
model.load_state_dict(torch.load("models/resnet18.pth", map_location=device))
test_acc = evaluate(test_loader)

# 6. 记录实验日志
from datetime import datetime
version = datetime.now().strftime("%Y%m%d_%H%M")
Path("experiments").mkdir(exist_ok=True)
with open("experiments/log.txt", "a", encoding="utf-8") as f:
    f.write(f"\n=== {version} ResNet18 ===\n")
    f.write(f"数据: {DATA_DIR}, Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH}\n")
    f.write(f"划分: 训练 {len(train_indices)} / 验证 {len(val_indices)} / 测试 {len(test_indices)}\n")
    f.write(f"最佳验证准确率: {best_acc:.4f}, 测试准确率: {test_acc:.4f}\n")

print(f"最佳验证准确率: {best_acc:.4f}，测试准确率: {test_acc:.4f}，模型已保存")   # 目标 ≥95%
