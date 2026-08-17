# src/infer.py —— 单张/批量推理
# 用法:
#   单张: python src/infer.py data/EDM/50X/class_01/1.jpg --model resnet
#   批量: python src/infer.py data/EDM/50X/ --model resnet --batch --output results.csv
import argparse, json, joblib, cv2, torch, torch.nn as nn
import numpy as np
import time, csv, glob
from pathlib import Path
from features import extract_features

# ========== GLCM 推理 ==========
_glcm_model = None

def predict_glcm(img_path):
    global _glcm_model
    if _glcm_model is None:
        model_files = sorted(glob.glob("models/v1_glcm_svm_*.pkl"))
        model_path = model_files[-1] if model_files else "models/glcm_svm.pkl"
        _glcm_model = joblib.load(model_path)
    feat = extract_features(img_path).reshape(1, -1)
    cls = _glcm_model.predict(feat)[0]
    conf = _glcm_model.predict_proba(feat).max()
    return cls, float(conf)

# ========== ResNet 推理 ==========
_resnet_model = None
_resnet_tf = None
_idx_to_class = None

def _load_resnet():
    global _resnet_model, _resnet_tf, _idx_to_class
    if _resnet_model is not None:
        return
    from torchvision import models, transforms
    class_to_idx = json.load(open("models/class_to_idx.json"))
    _idx_to_class = {v: k for k, v in class_to_idx.items()}
    _resnet_model = models.resnet18()
    _resnet_model.fc = nn.Linear(_resnet_model.fc.in_features, len(class_to_idx))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _resnet_model.load_state_dict(torch.load("models/resnet18.pth", map_location=device))
    _resnet_model.to(device)
    _resnet_model.eval()
    _resnet_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def predict_resnet(img_path):
    _load_resnet()
    device = next(_resnet_model.parameters()).device
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")
    with torch.no_grad():
        tensor = _resnet_tf(img).unsqueeze(0).to(device)
        prob = torch.softmax(_resnet_model(tensor), dim=1)
    idx = prob.argmax().item()
    return _idx_to_class[idx], prob.max().item()

# ========== 统一入口 ==========
def predict(img_path, model_name="resnet"):
    if model_name == "glcm":
        return predict_glcm(img_path)
    else:
        return predict_resnet(img_path)

# ========== 批量推理 ==========
def batch_infer(input_path, model_name="resnet", output_csv=None):
    """input_path 可以是单张图片或文件夹"""
    p = Path(input_path)
    if p.is_dir():
        images = sorted(p.rglob("*.jpg"))
    else:
        images = [p]

    print(f"模型: {model_name} | 共 {len(images)} 张图片")
    results = []
    t0 = time.time()
    for img_path in images:
        try:
            cls, conf = predict(img_path, model_name)
            results.append({"file": str(img_path), "prediction": cls, "confidence": round(conf, 4), "error": ""})
        except Exception as e:
            results.append({"file": str(img_path), "prediction": "ERROR", "confidence": 0, "error": str(e)})

    elapsed = time.time() - t0
    avg_ms = elapsed / len(images) * 1000 if images else 0
    print(f"耗时: {elapsed:.2f}s (平均 {avg_ms:.1f}ms/张)")

    # 输出 CSV
    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "prediction", "confidence", "error"])
            writer.writeheader()
            writer.writerows(results)
        print(f"结果已保存到: {output_csv}")

    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="图片路径或文件夹")
    ap.add_argument("--model", default="resnet", choices=["glcm", "resnet"])
    ap.add_argument("--batch", action="store_true", help="批量模式（传入文件夹时自动启用）")
    ap.add_argument("--output", default=None, help="输出 CSV 路径")
    a = ap.parse_args()

    p = Path(a.input)
    if p.is_dir() or a.batch:
        batch_infer(a.input, a.model, a.output)
    else:
        cls, conf = predict(a.input, a.model)
        print(f"预测等级: {cls}  置信度: {conf:.2%}")
