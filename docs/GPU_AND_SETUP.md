# GPU Training & Setup Guide

## 🎯 About GPU

**Yes, you can use GPU!** But there are important setup requirements.

### GPU Compatibility

✅ **GPU-Accelerated**:
- PyTorch (neural network training) - Uses CUDA/GPU
- Stable-Baselines3 - Automatically uses GPU if available

⚠️ **NOT GPU-Accelerated**:
- PyBullet (physics simulation) - CPU only
- Environment stepping - CPU only
- Goal checking - CPU only

**Result**: ~50-60% speedup overall (not 10x because physics is CPU-bound)

---

## 🔧 Prerequisites for GPU Training

### 1. Check if You Have GPU

```bash
# Check if NVIDIA GPU is available
nvidia-smi

# Check if PyTorch can use GPU
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

### 2. Install GPU Support (if not already installed)

```bash
# NVIDIA CUDA (12.1 recommended)
# Download from: https://developer.nvidia.com/cuda-downloads

# cuDNN (NVIDIA Deep Neural Network library)
# Download from: https://developer.nvidia.com/cudnn

# Or use conda/mamba (easier):
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3. Update Requirements (GPU version)

```bash
# Current requirements.txt already includes torch>=2.0.0
# This automatically installs GPU support if CUDA is available

# To verify GPU support:
uv pip install -r requirements.txt
python -c "from torch.utils.cpp_extension import CppExtension; import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 📊 Training Speed Comparison

### Training Configurations

| Setup | Envs | Device | Est. Time per Stage | Total (4 Stages) |
|-------|------|--------|-------------------|-----------------|
| 1 Env, CPU | 1 | CPU | ~1.5 hours | 6 hours |
| 4 Envs, CPU | 4 | CPU | ~30 min | 2 hours |
| 4 Envs, GPU | 4 | CUDA | ~20-25 min | ~1.5 hours |
| 8 Envs, GPU | 8 | CUDA | ~15-20 min | ~1 hour |
| 16 Envs, GPU | 16 | CUDA | ~12-15 min | ~50 min |

**Notes**:
- Physics simulation (PyBullet) can't use GPU → limits speedup
- Larger `--num-envs` helps more with GPU than CPU
- Recommended: `--num-envs 8` on GPU, `--num-envs 4` on CPU

---

## 🚀 Training Commands

### CPU Training (Recommended for beginners)

```bash
# Quick test (5 minutes)
uv run train_stages.py --stage 1 --steps 10000 --num-envs 1

# Normal training (with parallel envs)
uv run train_stages.py --stage 1 --num-envs 4
uv run train_stages.py --stage 2 --load saves/foosball_stage_1_completed.zip --num-envs 4
uv run train_stages.py --stage 3 --load saves/foosball_stage_2_completed.zip --num-envs 4
uv run train_stages.py --stage 4 --load saves/foosball_stage_3_completed.zip --num-envs 4
```

### GPU Training (Fast)

```bash
# With GPU (automatically detected by PyTorch)
uv run train_stages.py --stage 1 --num-envs 8
uv run train_stages.py --stage 2 --load saves/foosball_stage_1_completed.zip --num-envs 8
uv run train_stages.py --stage 3 --load saves/foosball_stage_2_completed.zip --num-envs 8
uv run train_stages.py --stage 4 --load saves/foosball_stage_3_completed.zip --num-envs 8

# With custom learning rate (if GPU training is unstable)
uv run train_stages.py --stage 2 --load saves/foosball_stage_1_completed.zip --num-envs 8 --lr 5e-4
```

**Note**: PyTorch automatically detects and uses GPU. No special flags needed!

---

## ✅ Verify GPU Setup

```bash
# Check PyTorch GPU support
python << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
else:
    print("GPU not available - will use CPU")
PYEOF

# Check if Stable-Baselines3 can detect GPU
python << 'PYEOF'
from stable_baselines3.common.utils import get_device
device = get_device("auto")
print(f"Stable-Baselines3 device: {device}")
PYEOF
```

---

## 🎓 Why Parallel Environments Matter More on GPU

**CPU (Limited by single physics thread)**:
- 1 env: Full speed (1x)
- 4 envs: ~3.5x speedup (overhead)
- 8 envs: ~6x speedup (diminishing returns)

**GPU (Can parallelize network updates)**:
- 1 env: Full speed (1x)
- 4 envs: ~3.8x speedup
- 8 envs: ~7x speedup (better scaling)
- 16 envs: ~12x speedup (optimal for most GPUs)

**Recommendation**:
- If GPU available: Use `--num-envs 8` or higher
- If CPU only: Use `--num-envs 4-8` (balance between speed and memory)
- Don't exceed 16 unless you have 8+ GB VRAM

---

## 🐛 Troubleshooting

### Problem: "CUDA out of memory"

Solution:
```bash
# Reduce parallel environments
uv run train_stages.py --stage 1 --num-envs 4

# Or reduce batch size
uv run train_stages.py --stage 1 --num-envs 8 --batch-size 32

# Check GPU memory
nvidia-smi
```

### Problem: "No GPU found even though I have one"

Solution:
```bash
# Verify GPU drivers
nvidia-smi

# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Test again
python -c "import torch; print(torch.cuda.is_available())"
```

### Problem: "GPU very slow / CPU faster"

Possible causes:
- Physics simulation is CPU-bound
- Not enough parallelism (`--num-envs` too low)
- GPU not fully utilized
- Memory transfers bottleneck

Solution:
```bash
# Try with more parallel envs
uv run train_stages.py --stage 1 --num-envs 16

# Check GPU utilization
# In another terminal: watch nvidia-smi
```

---

## 📊 Monitoring Training

### Real-time GPU Monitoring

```bash
# Terminal 1: Start training
uv run train_stages.py --stage 1 --num-envs 8

# Terminal 2: Monitor GPU
watch nvidia-smi

# Terminal 3: Monitor training
tensorboard --logdir logs/
# Open http://localhost:6006
```

### What to Watch

**Good GPU utilization**:
- GPU Memory: 60-90% used
- GPU Utilization: 80-100%
- GPU Power: 100-200W

**Bad GPU utilization**:
- GPU Memory: <50%
- GPU Utilization: <50%
- → Increase `--num-envs`

---

## ⚙️ Advanced Optimization

### For Maximum Speed (GPU)

```bash
# Aggressive settings
uv run train_stages.py --stage 1 \
  --num-envs 16 \
  --steps 250000 \
  --batch-size 256 \
  --n-steps 2048
```

### For Stability (GPU)

```bash
# Conservative settings
uv run train_stages.py --stage 1 \
  --num-envs 4 \
  --steps 250000 \
  --batch-size 32 \
  --n-steps 1024 \
  --lr 1e-4
```

### Balanced (Recommended)

```bash
# Default (good balance)
uv run train_stages.py --stage 1 --num-envs 8
```

---

## 🎯 Recommended Setup by Hardware

### CPU Only (e.g., Laptop)
```bash
uv run train_stages.py --stage 1 --num-envs 2 --steps 250000
# Time per stage: ~2 hours
# Total: ~8 hours
```

### CPU + GPU (e.g., Mid-range Laptop)
```bash
uv run train_stages.py --stage 1 --num-envs 8 --steps 250000
# Time per stage: ~20 min
# Total: ~1.5 hours
```

### High-end GPU (e.g., RTX 4090)
```bash
uv run train_stages.py --stage 1 --num-envs 16 --steps 250000
# Time per stage: ~10 min
# Total: ~40 min
```

---

## ✅ Checklist

- [ ] GPU available: `nvidia-smi` shows device
- [ ] PyTorch GPU support: `torch.cuda.is_available()` returns True
- [ ] Enough VRAM: At least 4GB (6GB+ recommended)
- [ ] Drivers updated: Latest NVIDIA drivers
- [ ] Training script ready: `uv run train_stages.py --stage 1`

---

**Note**: PyTorch automatically uses GPU if available. No manual device switching needed!

Start training:
```bash
uv run train_stages.py --stage 1 --num-envs 8
```

Good luck! 🚀
