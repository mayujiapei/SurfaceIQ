"""全局配置：换相机只需改这个文件，业务代码不用动。

换相机流程：
    1. 改下面的 CAMERA_TYPE / CAMERA_KWARGS；
    2. 用卡尺量出零件外径真值，重建 mm/px 换算系数：
           python src\\measure_ring.py --ref-mm <外径真值mm>
       系数跟着"相机+镜头+拍摄距离"走，三者任一变化都要重建。
"""

# ========== 相机配置 ==========
# 可选类型: "webcam"（USB/笔记本摄像头）| "hikvision"（海康工业相机）| "file"（调试用图片）
CAMERA_TYPE = "hikvision"
CAMERA_KWARGS = {"sdk_path": r"E:\海康sadp\MVS\Development\Samples\Python\MvImport",
                 "exposure_us": 30000.0, "gain": 0}

# 海康相机示例（MVS 装好后切换为下面两行）:
# CAMERA_TYPE = "hikvision"
# CAMERA_KWARGS = {"sdk_path": None, "exposure_us": 20000, "gain": 0}
