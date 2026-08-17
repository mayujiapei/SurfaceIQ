"""GLCM + SVM 基线：遍历数据集 → 提特征 → 训练 → 评估 → 保存模型"""
import joblib, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from features import extract_features

DATA_DIR = Path("data/EDM/50X")   # 先用一种放大倍数
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# 1. 遍历数据集，提取特征（首次提取后缓存为 .npy，后续直接加载）
cache_file = CACHE_DIR / "glcm_features_50X.npy"
label_file = CACHE_DIR / "glcm_labels_50X.npy"
if cache_file.exists():
    print("加载缓存特征...")
    X = np.load(cache_file, allow_pickle=False)
    y = np.load(label_file, allow_pickle=False)
else:
    print("首次提取特征（需几分钟）...")
    X, y = [], []
    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.glob("*.jpg"):   # 数据为 .jpg 格式
            X.append(extract_features(img_path))
            y.append(class_dir.name)
    X, y = np.array(X), np.array(y)
    np.save(cache_file, X)
    np.save(label_file, y)
    print(f"特征已缓存到 {cache_file}")
print(f"共 {len(y)} 张图，特征维度 {X.shape[1]}")

# 2. 划分训练/测试集（分层抽样保证每类比例一致）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 3. 标准化 + SVM，串成 Pipeline（保存后推理时不用重复写预处理）
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                class_weight="balanced", probability=True)),
])
model.fit(X_train, y_train)

# 4. 评估：16 分类准确率目标 ≥80% 即算管线跑通
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

# 5. 保存模型（带版本号）+ 记录实验日志
from datetime import datetime
version = datetime.now().strftime("%Y%m%d_%H%M")
Path("models").mkdir(exist_ok=True)
joblib.dump(model, f"models/v1_glcm_svm_{version}.pkl")

Path("experiments").mkdir(exist_ok=True)
with open("experiments/log.txt", "a", encoding="utf-8") as f:
    f.write(f"\n=== {version} GLCM+SVM ===\n")
    f.write(f"数据: {DATA_DIR}\n")
    f.write(f"样本数: {len(y)}, 特征维度: {X.shape[1]}\n")
    f.write(report + "\n")
print(f"模型已保存到 models/v1_glcm_svm_{version}.pkl")
