"""healthy_fp_tests.py -- where do detections on healthy-labelled eyes come from? (interim; to be folded into
notebooks/detector_evaluation.ipynb section 8b). Reads the notebook's inference cache on the pod; no detection rerun.

Production configuration (seed 42, threshold 0.35, EX disc-excluded, any-pixel to 512, clip_region erode 6).
Hard check first: per-eye false-positive px must reproduce the notebook's claims (test: EX 93.79, HE 513.95;
val: EX 160.98, HE 305.42 px per eye), otherwise stop.
T1 fellow eye: FP px per eye (Mann-Whitney) and any-detection (Fisher), fellow eye DR (grade >= 1) vs grade 0.
T2 photo quality: Spearman of FP px with brightness, contrast, dark fraction (green < 30), glare fraction (max >= 240).
T3 tile seams: share of FP outline px within 1 px of an interior sliding-window seam vs share of fundus px (chance).
Holm across all tests. Bootstrap CIs (10,000, seed 0).

Usage (pod):  python scripts/healthy_fp_tests.py --root /workspace/detector_eval_pkg
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, cv2
from scipy import stats

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--skip-reproduction", action="store_true")
a = ap.parse_args(); ROOT = Path(a.root)
INPUTS, OUT = ROOT / "notebooks" / "inputs", ROOT / "notebooks" / "outputs" / "detector_evaluation"
CACHE = OUT / "cache"; THR = [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]; TI = THR.index(0.35)
WINDOW = {"ex": (512, 384), "he": (768, 640)}; N_BOOT = 10_000; rng = np.random.default_rng(0)

def tissue_mask(img_rgb, thr=20):                    # verbatim from scratch/vessel_clip_proto.py
    t = (img_rgb.max(axis=2) > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(t)
    if n <= 1:
        return t > 0
    t = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    cs, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(t); cv2.drawContours(filled, cs, -1, 1, -1)
    return filled > 0
def clip_region(img_rgb, erode=6):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
    return cv2.erode(tissue_mask(img_rgb).astype(np.uint8), k) > 0
def starts(L, patch, stride):
    return sorted({max(0, min(s, L - patch)) for s in list(range(0, L - patch, stride)) + [L - patch]})

man = pd.read_csv(INPUTS / "detector_eval_manifest.csv", dtype=str).fillna("")
fellow = pd.read_csv(INPUTS / "healthy_fellow_eye.csv", dtype={"name": str})
H = man[man.group == "healthy"].merge(fellow[["name", "fellow_level"]], on="name", how="left")
rows, seam_rows = [], []
for r in H.itertuples():
    rgb = cv2.cvtColor(cv2.imread(str(CACHE / "rgb512" / f"{r.name}.png")), cv2.COLOR_BGR2RGB)
    clip, tis = clip_region(rgb), tissue_mask(rgb)
    full = cv2.imread(str(ROOT / r.image)); hgt, wid = full.shape[:2]
    g, mx = rgb[..., 1].astype(float), rgb.max(2)
    q = dict(brightness=g[tis].mean(), contrast=g[tis].std(), dark_frac=float((g[tis] < 30).mean()), glare_frac=float((mx[tis] >= 240).mean()))
    z = np.load(CACHE / "grid" / f"{r.name}__ex.npz"), np.load(CACHE / "grid" / f"{r.name}__he.npz")
    for det, zz in zip(("ex", "he"), z):
        m = np.unpackbits(zz["s42"][TI])[:512 * 512].reshape(512, 512).astype(bool) & clip
        rows.append(dict(name=r.name, role=r.role, det=det, fellow_level=r.fellow_level, fp_px=int(m.sum()), **q))
        if m.any():
            patch, stride = WINDOW[det]; sm = np.zeros((512, 512), bool)
            for L, axis, full_len in ((max(hgt, patch), 0, hgt), (max(wid, patch), 1, wid)):
                for s0 in starts(L, patch, stride):
                    for edge in (s0, s0 + patch):
                        if 0 < edge < full_len:
                            k = int(round(edge * 512 / full_len))
                            if 0 <= k < 512:
                                if axis == 0: sm[k, :] = True
                                else: sm[:, k] = True
            sm = cv2.dilate(sm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            bd = m & ~(cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
            seam_rows.append(dict(name=r.name, det=det, outline_px=int(bd.sum()), on_seam=int((bd & sm).sum()), chance=float((sm & tis).sum() / tis.sum())))
E = pd.DataFrame(rows); S = pd.DataFrame(seam_rows)

rep = E.groupby(["role", "det"]).fp_px.mean().round(2); print("per-eye FP px (reproduction check):"); print(rep.to_string())
if not a.skip_reproduction:
    want = {("test", "ex"): 93.79, ("test", "he"): 513.95, ("val", "ex"): 160.98, ("val", "he"): 305.42}
    bad = {k: (rep[k], v) for k, v in want.items() if abs(rep[k] - v) > 0.01}
    assert not bad, f"does not reproduce the notebook: {bad}"
    print("reproduces the notebook's healthy-eye numbers exactly")

T = []
for det in ("ex", "he"):
    d = E[(E.det == det) & E.fellow_level.notna()]; A, B = d[d.fellow_level == 0], d[d.fellow_level >= 1]
    ka, kb = int((A.fp_px > 0).sum()), int((B.fp_px > 0).sum())
    diff = B.fp_px.to_numpy()[rng.integers(0, len(B), (N_BOOT, len(B)))].mean(1) - A.fp_px.to_numpy()[rng.integers(0, len(A), (N_BOOT, len(A)))].mean(1)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    T.append(dict(det=det, test="T1 fellow eye: FP px per eye", detail=f"fellow DR n={len(B)} mean {B.fp_px.mean():.1f} median {B.fp_px.median():.0f} | fellow grade 0 n={len(A)} mean {A.fp_px.mean():.1f} median {A.fp_px.median():.0f} | diff in means 95% CI {lo:.1f} to {hi:.1f}",
                  effect=B.fp_px.mean() - A.fp_px.mean(), p=stats.mannwhitneyu(B.fp_px, A.fp_px, alternative="two-sided").pvalue))
    T.append(dict(det=det, test="T1 fellow eye: any detection", detail=f"fellow DR {kb}/{len(B)} ({100 * kb / len(B):.0f}%) | fellow grade 0 {ka}/{len(A)} ({100 * ka / len(A):.0f}%)",
                  effect=100 * kb / len(B) - 100 * ka / len(A), p=stats.fisher_exact([[kb, len(B) - kb], [ka, len(A) - ka]]).pvalue))
    e = E[E.det == det]
    for v in ("brightness", "contrast", "dark_frac", "glare_frac"):
        rr = stats.spearmanr(e.fp_px, e[v]); T.append(dict(det=det, test=f"T2 photo quality: {v}", detail=f"Spearman over {len(e)} eyes", effect=rr.statistic, p=rr.pvalue))
    s_ = S[S.det == det]
    if len(s_):
        exp_, obs_ = (s_.outline_px * s_.chance).to_numpy(), s_.on_seam.to_numpy(); idx = rng.integers(0, len(s_), (N_BOOT, len(s_)))
        lo, hi = np.percentile(obs_[idx].sum(1) / exp_[idx].sum(1), [2.5, 97.5]); dd = s_.on_seam / s_.outline_px - s_.chance
        T.append(dict(det=det, test="T3 tile seams: outline on seam vs chance", detail=f"{len(s_)} eyes | {100 * obs_.sum() / s_.outline_px.sum():.1f}% of outline px on a seam vs {100 * exp_.sum() / s_.outline_px.sum():.1f}% by chance | ratio 95% CI {lo:.2f}-{hi:.2f}",
                      effect=obs_.sum() / exp_.sum(), p=stats.wilcoxon(dd).pvalue if (dd != 0).any() else 1.0))
T = pd.DataFrame(T); p = T.p.to_numpy(); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
for rank, i in enumerate(o): run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1.0, run)
T["p_holm"] = adj; T["significant"] = T.p_holm < 0.05
OUT.mkdir(parents=True, exist_ok=True)
T.to_csv(OUT / "healthy_fp_tests.csv", index=False); E.to_csv(OUT / "healthy_fp_per_eye.csv", index=False); S.to_csv(OUT / "healthy_fp_seams.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", None)
print(T[["det", "test", "detail", "effect", "p_holm", "significant"]].round(4).to_string(index=False))
