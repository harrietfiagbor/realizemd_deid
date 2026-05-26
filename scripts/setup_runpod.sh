#!/bin/bash
# scripts/setup_runpod.sh
# One-shot environment setup for RunPod GPU instance
# Run ONCE after spinning up the instance:
#   bash scripts/setup_runpod.sh

set -e
echo "=== RealizeMD De-identification Pipeline — RunPod Setup ==="

# ── System deps ───────────────────────────────────────────────────────────────
apt-get update -q
apt-get install -q -y git libgl1-mesa-glx libglib2.0-0 p7zip-full unzip

# ── Python deps ───────────────────────────────────────────────────────────────
pip install -q \
    torch torchvision \
    tensorflow \
    opencv-python-headless \
    scikit-image \
    scikit-learn \
    matplotlib \
    Pillow \
    tqdm \
    pandas \
    numpy \
    omegaconf \
    huggingface_hub \
    transformers \
    lpips \
    pytorch-fid \
    lama-cleaner \
    pyyaml \
    einops \
    timm \
    kaggle

echo "✅ Python packages installed"

# ── Download EyePACS test images ──────────────────────────────────────────────
if ls /workspace/data/eyepacs/images/*.jpeg 1> /dev/null 2>&1; then
    echo "✅ EyePACS test images already present at /workspace/data/eyepacs/images/, skipping download."
else
    export KAGGLE_USERNAME="adjoadede33"
    export KAGGLE_KEY="KGAT_8abb244b54efbba9afc7f3a802af4408"

    mkdir -p /workspace/data/eyepacs/images
    kaggle competitions download -c diabetic-retinopathy-detection \
        -f test.zip.001 -p /workspace/data/

    # This handles the nested zip wrapper you found
    unzip -o /workspace/data/test.zip.001.zip -d /workspace/data/eyepacs/

    # This extracts the JPEGs flat into /workspace/data/eyepacs/images/
    7z e /workspace/data/eyepacs/test.zip.001 -o/workspace/data/eyepacs/images/ "*.jpeg" -r -y || true

    # Cleanup to keep your workspace from getting full
    rm -f /workspace/data/test.zip.001.zip /workspace/data/eyepacs/test.zip.001
fi
echo "Done: $(ls /workspace/data/eyepacs/images/ | wc -l) images"

# ── Download Model A weights (arkanivasarkar Attention U-Net) ─────────────────
mkdir -p /workspace/models
ln -sfn /workspace/models /workspace/realizemd_deid/models

if [ -f "/workspace/models/attention_unet/AttentionUNet.h5" ]; then
    echo "✅ Model A weights already exist at /workspace/models/attention_unet/, skipping download."
else
    if [ ! -d "/workspace/arkan_unet" ]; then
        git clone --quiet \
            https://github.com/arkanivasarkar/Retinal-Vessel-Segmentation-using-variants-of-UNET \
            /workspace/arkan_unet
    fi
    cp -r "/workspace/arkan_unet/Trained models/" /workspace/models/attention_unet/
    echo "✅ Model A weights ready at /workspace/models/attention_unet/"
fi
echo "   Available: $(ls '/workspace/models/attention_unet/')"

# ── Download RETFound ─────────────────────────────────────────────────────────
if [ -d "/workspace/RETFound" ]; then
    echo "✅ RETFound repo already present, skipping clone."
else
    git clone --quiet https://github.com/rmaphoh/RETFound /workspace/RETFound
    echo "✅ RETFound repo ready"
fi

if [ -f "/workspace/models/RETFound_mae_natureCFP.pth" ]; then
    echo "✅ RETFound weights already exist, skipping download."
else
    if [ -z "$HF_TOKEN" ]; then
        echo "❌ Error: RETFound is a gated model. To download the weights:"
        echo "   1. Accept the model terms at: https://huggingface.co/YukunZhou/RETFound_mae_natureCFP"
        echo "   2. Create a Read token at: https://huggingface.co/settings/tokens"
        echo "   3. Export your token in the shell before running the script:"
        echo "      export HF_TOKEN=\"your_huggingface_token\""
        echo "      bash scripts/setup_runpod.sh"
        exit 1
    fi

    python -c "
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download(
    repo_id='YukunZhou/RETFound_mae_natureCFP',
    filename='RETFound_mae_natureCFP.pth',
    local_dir='/workspace/models/',
    token=os.environ.get('HF_TOKEN')
)
print('RETFound weights:', path)
"
    echo "✅ RETFound weights ready"
fi

# ── Download LaMa weights ─────────────────────────────────────────────────────
# lama-cleaner auto-downloads on first use — pre-warm here to avoid cold start
python -c "
from lama_cleaner.model import LaMa
m = LaMa(device='cuda')
print('LaMa weights ready')
"
echo "✅ LaMa weights ready"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete — everything is ready ==="
echo ""
echo "De-identify:"
echo "  python scripts/run_pipeline.py \\"
echo "      --input   /workspace/data/eyepacs/images/ \\"
echo "      --output  /workspace/data/deid/ \\"
echo "      --weights '/workspace/models/attention_unet/AttentionUNet.h5' \\"
echo "      --device  cuda"