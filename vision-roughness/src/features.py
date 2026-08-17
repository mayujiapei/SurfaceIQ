"""从一张灰度图提取纹理+统计特征，输出一维特征向量"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy import stats

def extract_features(image_path, distances=(1, 2, 4), angles=(0, 45, 90, 135)):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"读不到图片: {image_path}")

    # --- GLCM 纹理特征 ---
    angles_rad = [np.deg2rad(a) for a in angles]
    glcm = graycomatrix(img, distances=list(distances), angles=angles_rad,
                        levels=256, symmetric=True, normed=True)
    feats = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        vals = graycoprops(glcm, prop)          # shape: (距离数, 方向数)
        feats += [vals.mean(), vals.std()]       # 每个属性2维，5个属性共10维

    # --- 灰度统计特征 ---
    flat = img.ravel().astype(np.float64)
    feats += [flat.mean(), flat.std(),
              float(stats.skew(flat)), float(stats.kurtosis(flat))]
    return np.array(feats, dtype=np.float32)     # 共14维
