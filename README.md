# Image Upscaler API

A FastAPI service that upscales images using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). Runs on CPU by default and can be switched to GPU via environment variables without any code changes.

## Requirements

- Docker and Docker Compose
- (Optional) NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU acceleration

## Project Structure

```
image-upscaler-api/
├── app/
│   └── main.py                  # FastAPI application
├── model/
│   └── RealESRGAN_x4plus.pth    # Model weights
├── data/                        # Volume-mounted directory
├── Dockerfile                   # x86-64 (CPU / discrete GPU)
├── Dockerfile.jetson            # aarch64 (Jetson Orin / JetPack 6.x)
├── docker-compose.yml           # Standard (x86-64) compose
├── docker-compose.jetson.yml    # Jetson compose
├── requirements.txt             # x86-64 deps (CPU-only PyTorch)
├── requirements.jetson.txt      # Jetson deps (torch excluded — in base image)
├── .env.example                 # Environment variable reference
├── README.md
└── HELME.md
```

## Quick Start

```bash
# 1. Copy the example environment file
cp .env.example .env

# 2. Edit .env to match your setup (optional, defaults work out of the box)

# 3. Build and start the service
docker compose up -d --build

# 4. Verify the service is running
curl http://localhost:8124/health
```

## API Endpoints

### `POST /upscale`

Upscales an uploaded image.

**Form fields:**

| Field  | Type | Default | Description |
|--------|------|---------|-------------|
| `file` | file | required | Image file (PNG, JPEG, WEBP) |
| `scale` | int | `4` | Upscale factor |
| `tile`  | int | `0` | Tile size for inference (0 = full image) |

**Supported input formats:** PNG, JPEG, WEBP

**Response:** The upscaled image in the same format as the input.

**Example:**

```bash
curl -X POST http://localhost:8124/upscale \
  -F "file=@photo.jpg" \
  -F "scale=4" \
  -F "tile=0" \
  --output result.jpg
```

### `GET /health`

Returns service health status.

```bash
curl http://localhost:8124/health
# {"status":"ok"}
```

## Configuration

All settings are controlled via environment variables. The easiest way is to create a `.env` file from `.env.example`.

### Device

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cpu` | `cpu`, `cuda`, or `auto` |
| `GPU_ID` | `0` | GPU device index (CUDA only) |
| `HALF_PRECISION` | `true` | FP16 inference (CUDA only) |
| `DOCKER_RUNTIME` | `runc` | Docker runtime (`runc` or `nvidia`) |

### CPU Performance Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `CPU_THREADS` | `0` | PyTorch intra-op threads (0 = auto) |
| `CPU_INTEROP_THREADS` | `0` | PyTorch inter-op threads (0 = auto) |
| `CV_THREADS` | `0` | OpenCV threads (0 = auto) |

**Recommended values for an 8-core / 16-thread CPU:**
```
CPU_THREADS=16
CPU_INTEROP_THREADS=2
CV_THREADS=8
```

### Inference

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `RealESRGAN_x4plus.pth` | Model filename inside `model/` |
| `DEFAULT_TILE` | `0` | Default tile size (0 = full image) |
| `TILE_PAD` | `10` | Padding between tiles |
| `PRE_PAD` | `0` | Padding added before inference |

### Output Quality

| Variable | Default | Description |
|----------|---------|-------------|
| `JPEG_QUALITY` | `95` | JPEG output quality (1–100) |
| `WEBP_QUALITY` | `95` | WebP output quality (1–100) |

## Switching to GPU

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
2. Update your `.env` file:

```env
DOCKER_RUNTIME=nvidia
DEVICE=cuda
GPU_ID=0
HALF_PRECISION=true
```

3. Rebuild and restart:

```bash
docker compose up -d --build
```

> **Note:** The default `requirements.txt` installs CPU-only PyTorch wheels. For GPU support on x86-64, replace the torch/torchvision lines with the appropriate CUDA wheels from [pytorch.org](https://pytorch.org/get-started/locally/).

## Jetson Orin Nano Super (JetPack 6.x)

The Jetson build uses a separate Dockerfile and compose file. No code changes are needed — the same `app/main.py` runs on Jetson.

### Prerequisites on the Jetson host

1. JetPack 6.x (tested on JetPack 6.0, r36.2.x).
2. NVIDIA Container Runtime installed:

```bash
sudo apt install nvidia-container-runtime
sudo systemctl restart docker
```

3. Verify Docker can access the GPU:

```bash
docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r36.2.0 nvidia-smi
```

### Quick Start (Jetson)

```bash
# 1. Copy the example environment file
cp .env.example .env
# (Jetson defaults are pre-configured in docker-compose.jetson.yml — .env is optional)

# 2. Build and start with the Jetson compose file
docker compose -f docker-compose.jetson.yml up -d --build

# 3. Verify
curl http://localhost:8124/health
```

If your JetPack release uses a different tag, override the base image:

```bash
JETSON_BASE_IMAGE=nvcr.io/nvidia/l4t-ml:r36.4.0-py3 \
docker compose -f docker-compose.jetson.yml up -d --build
```

### Jetson vs x86 file map

| Concern | x86-64 | Jetson |
|---------|--------|--------|
| Compose file | `docker-compose.yml` | `docker-compose.jetson.yml` |
| Dockerfile | `Dockerfile` | `Dockerfile.jetson` |
| Requirements | `requirements.txt` | `requirements.jetson.txt` |
| Default device | `cpu` | `cuda` |
| PyTorch source | PyPI CPU wheels | Pre-installed in `l4t-pytorch` base image |

### Notes

- `Dockerfile.jetson` defaults to `nvcr.io/nvidia/l4t-ml:r36.2.0-py3` and supports override via `JETSON_BASE_IMAGE` for JetPack tag compatibility.
- Do **not** install `torch` or `torchvision` via pip on Jetson — this can overwrite the JetPack-matched CUDA build with a CPU wheel.
- The Jetson Orin Nano Super has 8 GB of unified RAM/VRAM. Set `DEFAULT_TILE=256` if you encounter out-of-memory errors with very large images.
- `HALF_PRECISION=true` is safe and recommended on the Ampere architecture.

## Performance Notes

- **CPU (8-core):** ~2 minutes per 4× upscale on a typical photo. Use `DEFAULT_TILE=256` for large images to reduce memory usage at the cost of slightly more processing time.
- **GPU:** Typical inference is 3–10× faster than CPU depending on GPU model. Enable `HALF_PRECISION=true` for further speed gains on modern GPUs.
- **Tile size:** Setting `DEFAULT_TILE=0` processes the full image in one pass (fastest for small images). For images larger than ~1 MP on CPU, experiment with `DEFAULT_TILE=256` or `DEFAULT_TILE=384`.
