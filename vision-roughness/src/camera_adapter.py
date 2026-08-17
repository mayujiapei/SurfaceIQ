"""相机适配器层：把不同相机抽象成统一接口，方便后续替换为海康工业相机。"""
from abc import ABC, abstractmethod
from pathlib import Path
import cv2
import numpy as np


class BaseCamera(ABC):
    """相机抽象基类。所有具体相机都需要实现 capture 和 release。"""

    @abstractmethod
    def capture(self) -> np.ndarray:
        """捕获一帧图像，返回 BGR 格式的 numpy 数组。"""
        pass

    @abstractmethod
    def release(self):
        """释放相机资源。"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class WebcamCamera(BaseCamera):
    """电脑摄像头实现（基于 OpenCV VideoCapture）。

    用法：
        with WebcamCamera(device_id=0) as cam:
            frame = cam.capture()
    """

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._cap = None
        self._open()

    def _open(self):
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 device_id={self.device_id}，请检查驱动或换一个编号")
        # 等待几帧，让摄像头自动曝光稳定
        for _ in range(3):
            self._cap.read()

    def capture(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("摄像头采图失败，请检查连接")
        return frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class FileCamera(BaseCamera):
    """文件相机：每次 capture 都返回同一张本地图片。

    用于无摄像头时的单元测试或服务调试。
    """

    def __init__(self, image_path: str):
        self.image_path = Path(image_path)
        self._frame = cv2.imread(str(self.image_path))
        if self._frame is None:
            raise ValueError(f"无法读取图片: {self.image_path}")

    def capture(self) -> np.ndarray:
        return self._frame.copy()

    def release(self):
        pass


class HikvisionCamera(BaseCamera):
    """海康工业相机占位实现。

    硬件到位后在这里接入 MvImport / MVS Python SDK，
    业务代码无需改动，只改配置即可。
    """

    def __init__(self, **kwargs):
        raise NotImplementedError("海康相机尚未接入，请等待硬件到位后实现")

    def capture(self) -> np.ndarray:
        pass

    def release(self):
        pass


def create_camera(camera_type: str = "webcam", **kwargs) -> BaseCamera:
    """工厂函数：根据配置创建对应相机实例。"""
    camera_type = camera_type.lower()
    if camera_type == "webcam":
        return WebcamCamera(**kwargs)
    if camera_type == "file":
        return FileCamera(**kwargs)
    if camera_type == "hikvision":
        return HikvisionCamera(**kwargs)
    raise ValueError(f"不支持的相机类型: {camera_type}")
