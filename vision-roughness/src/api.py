"""FastAPI 推理服务。

启动：
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

或：
    python src/api.py

接口：
    GET  /health            健康检查
    POST /predict           上传图片推理
    POST /predict/camera    调用摄像头采图并推理
"""
import shutil
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

sys.path.insert(0, "src")
from camera_adapter import BaseCamera, WebcamCamera
from infer import predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期管理：启动时无需操作，关闭时释放摄像头。"""
    yield
    global _camera
    if _camera is not None:
        _camera.release()
        _camera = None


app = FastAPI(title="RoughSense API", version="0.1.0", lifespan=lifespan)

# 临时文件目录
TEMP_DIR = Path("temp/uploads")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# 全局相机实例（懒加载）
_camera: Optional[BaseCamera] = None


def get_camera() -> BaseCamera:
    """获取或初始化摄像头。"""
    global _camera
    if _camera is None:
        _camera = WebcamCamera(device_id=0)
    return _camera


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    model: str
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    camera_ready: bool


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查：验证模型文件是否存在、摄像头是否可用。"""
    model_ready = Path("models/resnet18.pth").exists() and Path("models/class_to_idx.json").exists()
    camera_ready = _camera is not None
    return HealthResponse(status="ok", model_ready=model_ready, camera_ready=camera_ready)


@app.post("/predict", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...), model: str = "resnet"):
    """上传单张图片进行推理。"""
    if model not in ("glcm", "resnet"):
        raise HTTPException(status_code=400, detail="model 必须是 glcm 或 resnet")

    save_path = TEMP_DIR / f"upload_{int(time.time() * 1000)}_{file.filename}"
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        t0 = time.time()
        cls, conf = predict(str(save_path), model_name=model)
        elapsed_ms = (time.time() - t0) * 1000

        return PredictResponse(
            prediction=cls,
            confidence=round(conf, 4),
            model=model,
            elapsed_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {e}")
    finally:
        # 清理临时文件
        if save_path.exists():
            save_path.unlink()


@app.post("/predict/camera", response_model=PredictResponse)
async def predict_camera(model: str = "resnet"):
    """调用摄像头采图并推理。"""
    if model not in ("glcm", "resnet"):
        raise HTTPException(status_code=400, detail="model 必须是 glcm 或 resnet")

    save_path = TEMP_DIR / f"camera_{int(time.time() * 1000)}.jpg"
    try:
        cam = get_camera()
        frame = cam.capture()
        cv2.imwrite(str(save_path), frame)

        t0 = time.time()
        cls, conf = predict(str(save_path), model_name=model)
        elapsed_ms = (time.time() - t0) * 1000

        return PredictResponse(
            prediction=cls,
            confidence=round(conf, 4),
            model=model,
            elapsed_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"采图或推理失败: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
