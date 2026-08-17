"""棋盘格标定：求相机内参+畸变系数；再用已知长度物求 mm/像素"""
import cv2, numpy as np, glob, pickle

PATTERN = (9, 6)        # 棋盘格内角点数（列, 行），按你的标定板改
SQUARE_MM = 20.0        # 每格实际边长 mm，打印后务必用尺子实测并修改
                        # 可用 src/generate_checkerboard.py 生成对应标定板

# 1. 准备世界坐标点 (0,0,0)~(8,5,0)
objp = np.zeros((PATTERN[0]*PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM

obj_pts, img_pts = [], []
for f in glob.glob("data/calibration/*.jpg"):   # 拍 10~20 张不同角度的标定板
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, PATTERN)
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        obj_pts.append(objp); img_pts.append(corners)

ret, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None)
print(f"标定重投影误差: {ret:.4f} 像素")     # <0.5 像素为合格
pickle.dump({"K": K, "dist": dist}, open("models/camera_calib.pkl", "wb"))
