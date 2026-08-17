"""生成可打印的棋盘格标定板图片。

用法：
    python src/generate_checkerboard.py

默认生成 9x6 内角点（即 10x7 黑白格）的 A4 尺寸棋盘格，保存到 data/calibration/checkerboard_9x6.png。

注意：
    打印时务必选择“实际大小 / 100% 缩放”，不要勾选“适应页面”或“拉伸”。
    打印后测量实际方格边长，填入 calibrate.py 的 SQUARE_MM。
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def generate_checkerboard(inner_corners_x: int, inner_corners_y: int,
                          square_mm: float, dpi: int = 300):
    """生成棋盘格图片。

    Args:
        inner_corners_x: 水平方向内角点数
        inner_corners_y: 垂直方向内角点数
        square_mm: 每个方格的边长（毫米）
        dpi: 输出图片 DPI

    Returns:
        PIL Image
    """
    # 棋盘格实际为 (inner_corners_x + 1) x (inner_corners_y + 1) 个方格
    squares_x = inner_corners_x + 1
    squares_y = inner_corners_y + 1

    # 毫米转像素
    mm_to_px = dpi / 25.4
    square_px = int(round(square_mm * mm_to_px))
    width_px = squares_x * square_px
    height_px = squares_y * square_px

    board = np.zeros((height_px, width_px), dtype=np.uint8)
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                board[y * square_px:(y + 1) * square_px,
                      x * square_px:(x + 1) * square_px] = 255

    img = Image.fromarray(board, mode="L")
    img.info["dpi"] = (dpi, dpi)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=9, help="水平方向内角点数")
    ap.add_argument("--height", type=int, default=6, help="垂直方向内角点数")
    ap.add_argument("--square", type=float, default=20.0, help="每个方格边长，单位 mm")
    ap.add_argument("--dpi", type=int, default=300, help="输出图片 DPI")
    ap.add_argument("--output", default="data/calibration/checkerboard_9x6.png", help="输出路径")
    args = ap.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = generate_checkerboard(args.width, args.height, args.square, args.dpi)
    img.save(output_path)

    squares_x = args.width + 1
    squares_y = args.height + 1
    print(f"已生成棋盘格: {output_path}")
    print(f"  内角点: {args.width} x {args.height}")
    print(f"  方格数: {squares_x} x {squares_y}")
    print(f"  方格边长: {args.square} mm")
    print(f"  图片尺寸: {img.size[0]} x {img.size[1]} px @ {args.dpi} DPI")
    print("\n打印提示：")
    print("  1. 用 A4 纸打印，选择“实际大小 / 100%”，不要适应页面")
    print("  2. 打印后用尺子测量一个方格的实际边长")
    print("  3. 把实测值填入 src/calibrate.py 的 SQUARE_MM")


if __name__ == "__main__":
    main()
