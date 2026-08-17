"""数据探索脚本：统计每个 class 的图片数量，随机显示 9 张图"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

DATA_DIR = Path("data/EDM/50X")

def count_images(data_dir):
    """统计每个类别的图片数量"""
    print("=" * 50)
    print(f"数据集路径: {data_dir.resolve()}")
    print("=" * 50)
    
    total = 0
    class_counts = {}
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        count = len(list(class_dir.glob("*.jpg")))
        class_counts[class_dir.name] = count
        total += count
        print(f"  {class_dir.name}: {count} 张")
    
    print("-" * 50)
    print(f"  总计: {len(class_counts)} 个类, {total} 张图片")
    print("=" * 50)
    return class_counts


def show_random_samples(data_dir, n_rows=3, n_cols=3):
    """随机显示 n_rows x n_cols 张图，每类最多取一张"""
    all_images = []
    all_labels = []
    
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        jpgs = list(class_dir.glob("*.jpg"))
        if jpgs:
            # 每类随机选一张
            sample = random.choice(jpgs)
            img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                all_images.append(img)
                all_labels.append(class_dir.name)
    
    # 从所有类的样本中再随机选 n_rows*n_cols 张
    n_show = n_rows * n_cols
    if len(all_images) < n_show:
        n_show = len(all_images)
    
    indices = random.sample(range(len(all_images)), n_show)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    fig.suptitle(f"Random Samples from {data_dir.name}", fontsize=14)
    
    for i, idx in enumerate(indices):
        row, col = divmod(i, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.imshow(all_images[idx], cmap='gray')
        ax.set_title(all_labels[idx])
        ax.axis('off')
    
    # 隐藏多余的子图
    for i in range(n_show, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig("data/sample_grid.png", dpi=100, bbox_inches='tight')
    print(f"\n样本网格图已保存到: data/sample_grid.png")
    plt.show()


def show_image_stats(data_dir):
    """显示每类图片的灰度统计（均值/标准差），观察纹理差异"""
    print("\n各类灰度统计（均值±标准差）:")
    print("-" * 40)
    
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        jpgs = list(class_dir.glob("*.jpg"))[:10]  # 每类取前10张
        means, stds = [], []
        for jpg in jpgs:
            img = cv2.imread(str(jpg), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                means.append(img.mean())
                stds.append(img.std())
        if means:
            print(f"  {class_dir.name}: gray={np.mean(means):.1f}±{np.std(means):.1f}, "
                  f"std={np.mean(stds):.1f}±{np.std(stds):.1f}")


if __name__ == "__main__":
    random.seed(42)
    
    # 1. 统计图片数量
    count_images(DATA_DIR)
    
    # 2. 灰度统计
    show_image_stats(DATA_DIR)
    
    # 3. 随机显示 9 张
    show_random_samples(DATA_DIR, n_rows=3, n_cols=3)
