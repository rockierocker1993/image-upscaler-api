from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
import os
import numpy as np
import cv2

app = FastAPI(title="Image Upscaler Service")

def _parse_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


DEVICE = os.getenv("DEVICE", "auto").strip().lower()
GPU_ID = int(os.getenv("GPU_ID", "0"))
HALF_PRECISION = _parse_bool(os.getenv("HALF_PRECISION"), default=True)
CPU_THREADS = int(os.getenv("CPU_THREADS", "0"))
CPU_INTEROP_THREADS = int(os.getenv("CPU_INTEROP_THREADS", "0"))
CV_THREADS = int(os.getenv("CV_THREADS", "0"))

if DEVICE == "cuda":
    use_cuda = torch.cuda.is_available()
elif DEVICE == "cpu":
    use_cuda = False
else:
    use_cuda = torch.cuda.is_available()

if not use_cuda:
    if CPU_THREADS > 0:
        torch.set_num_threads(CPU_THREADS)
    if CPU_INTEROP_THREADS > 0:
        torch.set_num_interop_threads(CPU_INTEROP_THREADS)
    if CV_THREADS > 0:
        cv2.setNumThreads(CV_THREADS)
else:
    torch.backends.cudnn.benchmark = True

MODEL = os.getenv("MODEL", "RealESRGAN_x4plus.pth")
DEFAULT_TILE = int(os.getenv("DEFAULT_TILE", "0"))
TILE_PAD = int(os.getenv("TILE_PAD", "10"))
PRE_PAD = int(os.getenv("PRE_PAD", "0"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "95"))
WEBP_QUALITY = int(os.getenv("WEBP_QUALITY", "95"))

rrdbnet = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4,
)

upsampler = RealESRGANer(
    scale=4,
    model_path=f"model/{MODEL}",
    model=rrdbnet,
    tile=DEFAULT_TILE,
    tile_pad=TILE_PAD,
    pre_pad=PRE_PAD,
    half=HALF_PRECISION and use_cuda,
    gpu_id=GPU_ID if use_cuda else None,
)

@app.post("/upscale")
async def upscale(
    file: UploadFile = File(...),
    scale: int = Form(4),
    tile: int = Form(0)
):
    # detect media type
    content_type = file.content_type or "image/png"

    ext_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
    }

    ext = ext_map.get(content_type, ".png")
    encode_params = []
    if ext == ".jpg":
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    elif ext == ".webp":
        encode_params = [int(cv2.IMWRITE_WEBP_QUALITY), WEBP_QUALITY]

    file_bytes = await file.read()
    np_bytes = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return Response(
            content=b"Invalid image file",
            media_type="text/plain",
            status_code=400,
        )

    requested_tile = tile if tile > 0 else DEFAULT_TILE

    with torch.inference_mode():
        original_tile = upsampler.tile_size
        upsampler.tile_size = requested_tile
        try:
            sr_bgr, _ = upsampler.enhance(
                image_bgr,
                outscale=scale,
            )
        finally:
            upsampler.tile_size = original_tile

    ok, encoded = cv2.imencode(ext, sr_bgr, encode_params)
    if not ok:
        return Response(
            content=b"Failed to encode image",
            media_type="text/plain",
            status_code=500,
        )

    final_bytes = encoded.tobytes()

    return Response(
        content=final_bytes,
        media_type=content_type if content_type in ext_map else "image/png",
        headers={
            "Content-Disposition": f"inline; filename=result{ext}",
            "Content-Length": str(len(final_bytes))
        }
    )

@app.get("/health")
def health():
    return {"status": "ok"}