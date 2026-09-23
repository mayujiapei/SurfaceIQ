"""相机适配器层：把不同相机抽象成统一接口，方便替换相机品牌。

契约（所有相机必须遵守）：
    capture() 必须返回 BGR 三通道 uint8 图像。黑白相机（如 MV-CU120-10GM）
    必须在适配器内部把单通道灰度图转成 BGR，否则下游 measure.py 的
    COLOR_BGR2GRAY 会崩溃。统一用 BaseCamera._ensure_bgr() 处理。

切换相机：只需修改 src/config.py 的 CAMERA_TYPE / CAMERA_KWARGS。
"""
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np


def _add_mvs_runtime_to_dll_path():
    """把 MVS 运行时 DLL 目录补进 DLL 搜索路径。

    MVS 5.x 把 MvCameraControl.dll 装到 Common Files 并写入系统 PATH，
    但安装前已打开的终端/进程拿不到新 PATH，这里显式补一份，保证
    任何时刻启动的进程都能加载 SDK。
    """
    candidates = [
        os.environ.get("MVCAM_COMMON_RUNENV"),
        r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
        r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64",
    ]
    for d in candidates:
        if not d or not os.path.isdir(d):
            continue
        if d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(d)
        except (OSError, AttributeError):
            pass  # Python<3.8 无此接口，PATH 方案已足够


class BaseCamera(ABC):
    """相机抽象基类。所有具体相机都需要实现 capture 和 release。"""

    @abstractmethod
    def capture(self) -> np.ndarray:
        """捕获一帧图像，返回 BGR 三通道 uint8 格式的 numpy 数组。"""
        pass

    @abstractmethod
    def release(self):
        """释放相机资源。必须可重复调用（幂等）。"""
        pass

    @staticmethod
    def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
        """把灰度图/BGRA 图统一转换为 BGR 三通道，保证下游代码无需关心相机类型。"""
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

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
        return self._ensure_bgr(frame)

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
        return self._ensure_bgr(self._frame.copy())

    def release(self):
        pass


class HikvisionCamera(BaseCamera):
    """海康机器人 GigE 工业相机（基于 MVS Python SDK，如 MV-CU120-10GM）。

    前置条件：
        1. 安装海康 MVS 软件（含 Python SDK 组件）；
        2. 相机已通过网线与电脑连接并上电，MVS 客户端能枚举到相机。

    用法（config.py 中配置）：
        CAMERA_TYPE = "hikvision"
        CAMERA_KWARGS = {"sdk_path": None, "exposure_us": 20000, "gain": 0}

    注意：
        - MV-CU120-10GM 是黑白相机，SDK 输出 Mono8 单通道，capture() 内部
          已转换为 BGR 三通道，下游代码无需改动；
        - 现场使用建议传入固定的 exposure_us / gain，避免自动曝光导致
          图像亮度漂移（纹理特征对亮度敏感）。
    """

    DEFAULT_SDK_PATH = r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"

    def __init__(self, sdk_path: str = None, exposure_us: float = None,
                 gain: float = None, timeout_ms: int = 3000, packet_size=1500):
        self._timeout_ms = timeout_ms
        self._packet_size = packet_size
        self._cam = None
        self._grabbing = False
        self._mv = self._import_sdk(sdk_path or self.DEFAULT_SDK_PATH)
        self._open(exposure_us, gain)

    @staticmethod
    def _import_sdk(sdk_path: str):
        """导入海康 MVS Python SDK；失败时给出操作指引。"""
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        _add_mvs_runtime_to_dll_path()
        try:
            import MvCameraControl_class as mv
        except ImportError as e:
            raise RuntimeError(
                f"找不到海康 MVS Python SDK（{e}）\n"
                "请按以下步骤处理：\n"
                "  1. 安装海康机器人 MVS 软件（安装时勾选 Python SDK）；\n"
                "  2. 确认 MvCameraControl_class.py 的实际位置（通常在\n"
                f"     {HikvisionCamera.DEFAULT_SDK_PATH}）；\n"
                "  3. 若路径不同，在 config.py 的 CAMERA_KWARGS 中设置 sdk_path。"
            )
        return mv

    def _check(self, ret: int, action: str):
        """检查 SDK 返回码，非 0 抛出带中文说明的异常。"""
        if ret != 0:
            raise RuntimeError(f"海康相机操作失败: {action} (错误码 0x{ret:08x})")

    def _open(self, exposure_us, gain):
        """枚举 GigE 相机，逐个尝试打开并试取一帧，直到成功。

        同一台相机经多个网卡（或与其他设备 IP 冲突）会被枚举成多项，
        其中可能只有部分枚举项能真正收到视频流——"打开成功"不算数，
        必须试取一帧验证。
        """
        from ctypes import POINTER, cast

        mv = self._mv
        device_list = mv.MV_CC_DEVICE_INFO_LIST()
        ret = mv.MvCamera.MV_CC_EnumDevices(mv.MV_GIGE_DEVICE, device_list)
        self._check(ret, "枚举设备")
        if device_list.nDeviceNum == 0:
            raise RuntimeError(
                "找不到海康相机，请检查：\n"
                "  1. 网线是否连接相机与电脑，相机是否上电（指示灯）；\n"
                "  2. 电脑网卡 IP 是否与相机同一网段（用 MVS 的 IP 配置工具检查）；\n"
                "  3. MVS 客户端里能否枚举到相机。"
            )

        errors = []
        for i in range(device_list.nDeviceNum):
            gige = cast(device_list.pDeviceInfo[i], POINTER(mv.MV_GIGE_DEVICE_INFO)).contents
            model = bytes(gige.chModelName).split(b"\x00")[0].decode("ascii", "ignore")
            ip = gige.nCurrentIp
            ip_str = ".".join(str((ip >> s) & 0xFF) for s in (24, 16, 8, 0))
            print(f"[hikvision] 尝试相机{i}: {model} @ {ip_str}")
            try:
                self._try_open_one(device_list, i, exposure_us, gain)
                print(f"[hikvision] 相机{i} 取图验证通过，已就绪")
                return
            except Exception as e:
                errors.append(f"  相机{i}({model} @ {ip_str}): {e}")

        raise RuntimeError(
            "枚举到的海康相机全部无法取图：\n" + "\n".join(errors) +
            "\n请检查：网线是否插在有相机的网口、相机是否被 MVS 客户端占用、"
            "是否存在 IP 冲突（两台设备同 IP）。"
        )

    def _try_open_one(self, device_list, idx, exposure_us, gain):
        """打开指定枚举项、配置参数、开始采集并试取一帧；失败抛异常并清理。"""
        from ctypes import POINTER, byref, cast, memset, sizeof

        mv = self._mv
        cam = mv.MvCamera()
        device_info = cast(device_list.pDeviceInfo[idx], POINTER(mv.MV_CC_DEVICE_INFO)).contents
        self._check(cam.MV_CC_CreateHandle(device_info), "创建设备句柄")
        try:
            self._check(cam.MV_CC_OpenDevice(mv.MV_ACCESS_Exclusive, 0), "打开设备")

            # GigE 网口包大小：默认 1500（标准以太网，任何网卡都稳定）；
            # 现场网卡开启巨型帧(Jumbo Frame)后可调大（如 8164）提升帧率，
            # 或传 packet_size="auto" 用 SDK 推荐值（部分网卡虚报巨型帧支持，会丢包）
            pkt_size = self._packet_size
            if pkt_size == "auto":
                pkt = cam.MV_CC_GetOptimalPacketSize()
                pkt_size = int(pkt) if 0 < pkt < 0x80000000 else 1500
            self._check(cam.MV_CC_SetIntValue("GevSCPSPacketSize", int(pkt_size)),
                        "设置网口包大小")

            # 连续采集模式（关闭触发；现场若用光电触发，在这里改 TriggerMode/TriggerSource）
            self._check(cam.MV_CC_SetEnumValue("TriggerMode", mv.MV_TRIGGER_MODE_OFF),
                        "设置触发模式")

            # 手动曝光/增益：锁定成像参数，避免亮度漂移影响纹理特征
            if exposure_us is not None:
                self._check(cam.MV_CC_SetEnumValue("ExposureAuto", 0), "关闭自动曝光")
                self._check(cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us)),
                            "设置曝光时间")
            if gain is not None:
                self._check(cam.MV_CC_SetEnumValue("GainAuto", 0), "关闭自动增益")
                self._check(cam.MV_CC_SetFloatValue("Gain", float(gain)), "设置增益")

            self._check(cam.MV_CC_StartGrabbing(), "开始采集")

            # 试取一帧：枚举项可能对应收不到视频流的网卡，打开成功≠能出图
            frame = mv.MV_FRAME_OUT()
            memset(byref(frame), 0, sizeof(frame))
            try:
                self._check(cam.MV_CC_GetImageBuffer(frame, self._timeout_ms),
                            "试取一帧")
            finally:
                if frame.pBufAddr:
                    cam.MV_CC_FreeImageBuffer(frame)
        except Exception:
            try:
                cam.MV_CC_StopGrabbing()
            except Exception:
                pass
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            raise
        self._cam = cam
        self._grabbing = True

    def capture(self) -> np.ndarray:
        """取一帧图像，返回 BGR 三通道数组；超时抛出带排查指引的异常。"""
        from ctypes import byref, memset, sizeof, string_at

        if self._cam is None or not self._grabbing:
            raise RuntimeError("海康相机未在采集状态，请重新创建相机实例")

        mv = self._mv
        frame_out = mv.MV_FRAME_OUT()
        memset(byref(frame_out), 0, sizeof(frame_out))

        ret = self._cam.MV_CC_GetImageBuffer(frame_out, self._timeout_ms)
        if ret != 0:
            raise RuntimeError(
                f"海康相机取图超时/失败 (错误码 0x{ret:08x})，请检查：\n"
                "  1. 网线是否松动（GigE 链路灯是否亮）；\n"
                "  2. 相机是否被其他程序（如 MVS 客户端）占用。"
            )

        try:
            info = frame_out.stFrameInfo
            data = string_at(frame_out.pBufAddr, info.nFrameLen)

            if info.enPixelType == mv.PixelType_Gvsp_Mono8:
                # 黑白相机：单通道灰度图
                img = np.frombuffer(data, dtype=np.uint8).reshape(info.nHeight, info.nWidth)
            elif info.enPixelType == mv.PixelType_Gvsp_BGR8_Packed:
                img = np.frombuffer(data, dtype=np.uint8).reshape(info.nHeight, info.nWidth, 3)
            elif info.enPixelType == mv.PixelType_Gvsp_RGB8_Packed:
                img = np.frombuffer(data, dtype=np.uint8).reshape(info.nHeight, info.nWidth, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                raise RuntimeError(
                    f"暂不支持的海康相机像素格式: 0x{info.enPixelType:08x}\n"
                    "请在 MVS 客户端把 Pixel Format 改为 Mono8（黑白）或 BGR8/RGB8（彩色）。"
                )
            return self._ensure_bgr(img.copy())
        finally:
            self._cam.MV_CC_FreeImageBuffer(frame_out)

    def release(self):
        """停止采集并释放设备，可重复调用。"""
        cam, self._cam = self._cam, None
        if cam is None:
            return
        try:
            if self._grabbing:
                cam.MV_CC_StopGrabbing()
        finally:
            self._grabbing = False
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()


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
