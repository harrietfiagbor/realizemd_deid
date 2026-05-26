# RealizeMD — Retinal De-identification Pipeline

**Stage:** POC — Vessel Segmentation + Pathology Protection + LaMa Inpainting  
**Model:** Attention U-Net (Model A) + Rule-based pathology detection + LaMa  
**Eval:** RETFound re-id metric + DR AUC preservation + FID/SSIM/LPIPS  

---

## Structure

```
realizemd_deid/
├── pipeline/
│   ├── preprocessing.py    — FOV masking, CLAHE, normalisation
│   ├── segmentation.py     — Model A vessel segmentation
│   ├── pathology.py        — Rule-based lesion detection
│   ├── masking.py          — Combined inpaint mask builder
│   ├── inpainting.py       — LaMa inpainting wrapper
│   └── deidentify.py       — Single deidentify(image) -> image interface
├── eval/
│   ├── reid.py             — RETFound re-id rate (privacy metric)
│   ├── utility.py          — DR AUC preservation (utility metric)
│   ├── realism.py          — FID / SSIM / LPIPS (realism metric)
│   └── scorecard.py        — Generates full scorecard per model
├── scripts/
│   ├── run_pipeline.py     — Batch de-identification on a folder
│   ├── run_eval.py         — Full eval harness on holdout set
│   └── setup_runpod.sh     — RunPod environment setup
├── configs/
│   └── default.yaml        — All hyperparameters in one place
└── requirements.txt
```

## Quickstart (RunPod)

```bash
# 1. Setup environment
bash scripts/setup_runpod.sh

# 2. Run pipeline on a folder of images
python scripts/run_pipeline.py \
    --input /data/eyepacs/images/ \
    --output /data/deid_output/ \
    --config configs/default.yaml

# 3. Run eval on holdout
python scripts/run_eval.py \
    --original /data/eyepacs/images/ \
    --deid /data/deid_output/ \
    --output /data/evals/scorecards/ \
    --config configs/default.yaml
```

## Key design decisions

- `deidentify()` is a pure function: `np.ndarray (H,W,3) -> np.ndarray (H,W,3)`
- All hyperparameters live in `configs/default.yaml` — no magic numbers in code
- Eval harness is completely separate from pipeline — can plug in any future model
- Vessel mask dilation: 5px ellipse kernel (coverage ~10%, per supervisor's 8-12% target)
