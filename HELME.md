# HELME — Helm-style Environment Reference

All environment variables accepted by the `image-upscaler-api` service. Use these in your `.env` file, `docker-compose.yml` override, Kubernetes ConfigMap/Secret, or any orchestration layer.

---

## Variable Reference

### Runtime & Device

| Variable | Type | Default | Accepted Values | Description |
|----------|------|---------|-----------------|-------------|
| `DOCKER_RUNTIME` | string | `runc` | `runc`, `nvidia` | Docker container runtime. Set to `nvidia` when using GPU. |
| `DEVICE` | string | `cpu` | `cpu`, `cuda`, `auto` | Inference device. `auto` picks CUDA if available, else CPU. |
| `GPU_ID` | int | `0` | `0`, `1`, … | GPU device index. Only used when `DEVICE=cuda`. |
| `HALF_PRECISION` | bool | `true` | `true`, `false`, `1`, `0` | Enable FP16 half-precision inference. Only used when `DEVICE=cuda`. Has no effect on CPU. |

### CPU Parallelism

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CPU_THREADS` | int | `0` | PyTorch intra-op parallelism threads. `0` = let PyTorch decide (usually all cores). Tune to number of logical CPUs. |
| `CPU_INTEROP_THREADS` | int | `0` | PyTorch inter-op parallelism threads. `0` = let PyTorch decide. Typically set to `2`–`4`. |
| `CV_THREADS` | int | `0` | OpenCV thread pool size. `0` = let OpenCV decide. Set to number of physical cores for best results. |

### Model

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MODEL` | string | `RealESRGAN_x4plus.pth` | Filename of the model weights inside the `model/` directory. Must be a Real-ESRGAN compatible `.pth` file. |

### Inference Tiling

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEFAULT_TILE` | int | `0` | Default tile size in pixels for inference. `0` = process the full image in one pass (fastest for small images, high memory usage for large ones). Recommended: `256`–`512` for large images. |
| `TILE_PAD` | int | `10` | Overlap padding between tiles in pixels. Prevents visible seams at tile boundaries. Increase to `20`–`32` if seams are visible. |
| `PRE_PAD` | int | `0` | Padding added to the image before inference and removed afterward. Helps avoid edge artifacts. |

### Output Quality

| Variable | Type | Default | Range | Description |
|----------|------|---------|-------|-------------|
| `JPEG_QUALITY` | int | `95` | `1`–`100` | JPEG output compression quality. Higher = better quality, larger file. |
| `WEBP_QUALITY` | int | `95` | `1`–`100` | WebP output compression quality. Higher = better quality, larger file. |

---

## Profiles

### Minimal CPU (low-spec machine)

```env
DOCKER_RUNTIME=runc
DEVICE=cpu
HALF_PRECISION=false
CPU_THREADS=4
CPU_INTEROP_THREADS=1
CV_THREADS=2
DEFAULT_TILE=256
TILE_PAD=10
PRE_PAD=0
JPEG_QUALITY=90
WEBP_QUALITY=90
MODEL=RealESRGAN_x4plus.pth
```

### Optimal CPU (8-core / 16-thread)

```env
DOCKER_RUNTIME=runc
DEVICE=cpu
HALF_PRECISION=false
CPU_THREADS=16
CPU_INTEROP_THREADS=2
CV_THREADS=8
DEFAULT_TILE=0
TILE_PAD=10
PRE_PAD=0
JPEG_QUALITY=95
WEBP_QUALITY=95
MODEL=RealESRGAN_x4plus.pth
```

### GPU (NVIDIA, FP16)

```env
DOCKER_RUNTIME=nvidia
DEVICE=cuda
GPU_ID=0
HALF_PRECISION=true
CPU_THREADS=0
CPU_INTEROP_THREADS=0
CV_THREADS=0
DEFAULT_TILE=0
TILE_PAD=10
PRE_PAD=0
JPEG_QUALITY=95
WEBP_QUALITY=95
MODEL=RealESRGAN_x4plus.pth
```

---

## Boolean Parsing

Bool variables accept any of the following truthy values (case-insensitive):

```
1  true  yes  y  on
```

Any other value is treated as `false`.

---

## Per-Request Override

The `tile` form field in a `POST /upscale` request overrides `DEFAULT_TILE` for that individual request:

```bash
# Use tile=256 for this request only, regardless of DEFAULT_TILE
curl -X POST http://localhost:8124/upscale \
  -F "file=@large_photo.jpg" \
  -F "scale=4" \
  -F "tile=256" \
  --output result.jpg
```

Setting `tile=0` in the request falls back to `DEFAULT_TILE`.
