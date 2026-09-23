"""海康取图超时(0x80000007)排查脚本：手动控制每个环节，定位问题。

用法（venv 下）：
    python temp/hik_debug.py
"""
import sys
from ctypes import POINTER, byref, cast, memset, sizeof

sys.path.insert(0, "src")
import config
from camera_adapter import _add_mvs_runtime_to_dll_path

_add_mvs_runtime_to_dll_path()
sys.path.insert(0, config.CAMERA_KWARGS["sdk_path"])
import MvCameraControl_class as mv

FORCED_PACKET_SIZE = 1500   # 强制标准包，排除巨型帧问题
TIMEOUT_MS = 3000


def check(ret, action):
    if ret != 0:
        print(f"[FAIL] {action}: 0x{ret:08x}")
    else:
        print(f"[ OK ] {action}")
    return ret


def main():
    device_list = mv.MV_CC_DEVICE_INFO_LIST()
    if check(mv.MvCamera.MV_CC_EnumDevices(mv.MV_GIGE_DEVICE, device_list), "枚举设备") != 0:
        return
    print(f"       发现 {device_list.nDeviceNum} 台相机")
    if device_list.nDeviceNum == 0:
        return

    cam = mv.MvCamera()
    info = cast(device_list.pDeviceInfo[0], POINTER(mv.MV_CC_DEVICE_INFO)).contents
    if check(cam.MV_CC_CreateHandle(info), "创建句柄") != 0:
        return
    if check(cam.MV_CC_OpenDevice(mv.MV_ACCESS_Exclusive, 0), "打开设备") != 0:
        return

    # 强制标准包 1500
    pkt = cam.MV_CC_GetOptimalPacketSize()
    print(f"       SDK 推荐包大小: {pkt}")
    check(cam.MV_CC_SetIntValue("GevSCPSPacketSize", FORCED_PACKET_SIZE), "强制包大小=1500")

    check(cam.MV_CC_SetEnumValue("TriggerMode", 0), "关闭触发")
    check(cam.MV_CC_SetEnumValue("ExposureAuto", 0), "关闭自动曝光")
    check(cam.MV_CC_SetFloatValue("ExposureTime", 5000.0), "曝光=5000us")

    # 打印 PayloadSize（一帧需要多少字节）
    payload = mv.MVCC_INTVALUE()
    memset(byref(payload), 0, sizeof(payload))
    if cam.MV_CC_GetIntValue("PayloadSize", payload) == 0:
        print(f"       PayloadSize = {payload.nCurValue} 字节")

    check(cam.MV_CC_StartGrabbing(), "开始采集")

    frame = mv.MV_FRAME_OUT()
    memset(byref(frame), 0, sizeof(frame))
    ret = cam.MV_CC_GetImageBuffer(frame, TIMEOUT_MS)
    if ret == 0:
        print(f"[ OK ] 取图成功! {frame.stFrameInfo.nWidth}x{frame.stFrameInfo.nHeight}, "
              f"{frame.stFrameInfo.nFrameLen} 字节, 像素格式 0x{frame.stFrameInfo.enPixelType:08x}")
        cam.MV_CC_FreeImageBuffer(frame)
    else:
        print(f"[FAIL] 取图仍然失败: 0x{ret:08x}（{TIMEOUT_MS}ms 超时）")
        print("       -> 包大小已排除，嫌疑指向 Windows 防火墙拦截 UDP 视频流")
        print("          或 MVS 网卡过滤驱动未绑定到这个网卡")

    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()


if __name__ == "__main__":
    main()
