"""
eval/scorecard.py
Generates the one-page scorecard per model candidate.
Format matches the supervisor's framework exactly.
"""

import json
from datetime import date
from pathlib import Path


def generate(model_name: str,
             reid_results: dict,
             utility_results: dict,
             realism_results: dict,
             n_patients: int,
             n_images: int,
             output_dir: str = None) -> str:
    """
    Generate a formatted scorecard string + optionally save to file.

    Args:
        model_name:       e.g. 'AttentionUNet_ModelA + LaMa'
        reid_results:     output of eval.reid.compute_reid_rate()
        utility_results:  output of eval.utility.compute_dr_preservation()
        realism_results:  output of eval.realism.compute_ssim_lpips() + fid
        n_patients:       number of patients in holdout
        n_images:         number of images in holdout
        output_dir:       if set, saves scorecard as .txt and .json

    Returns:
        Formatted scorecard string
    """

    def _status(condition: bool) -> str:
        return '✅ PASS' if condition else '❌ FAIL'

    def _fmt(val, decimals=3):
        if val is None:
            return 'N/A'
        return f'{val:.{decimals}f}'

    # ── Privacy ───────────────────────────────────────────────────────────────
    rank1        = reid_results.get('rank1_rate', None)
    random_base  = reid_results.get('random_baseline', None)
    ratio        = reid_results.get('ratio_vs_random', None)
    sp_auc       = reid_results.get('same_patient_auc', None)
    reid_pass    = reid_results.get('pass', False)

    # ── Utility ───────────────────────────────────────────────────────────────
    auc_orig     = utility_results.get('dr_auc_original', None)
    auc_deid     = utility_results.get('dr_auc_deid', None)
    pres_ratio   = utility_results.get('preservation_ratio', None)
    util_pass    = utility_results.get('pass', False)
    auc_low      = utility_results.get('dr_auc_deid_low_grade', None)
    auc_high     = utility_results.get('dr_auc_deid_high_grade', None)

    # ── Realism ───────────────────────────────────────────────────────────────
    fid          = realism_results.get('fid', None)
    ssim_mean    = realism_results.get('ssim_mean', None)
    lpips_mean   = realism_results.get('lpips_mean', None)
    real_pass    = realism_results.get('overall_pass', False)

    fid_pass     = fid is not None and fid < 30
    ssim_pass    = ssim_mean is not None and (0.65 <= ssim_mean <= 0.85)
    lpips_pass   = lpips_mean is not None and (0.15 <= lpips_mean <= 0.35)

    overall_pass = reid_pass and util_pass and fid_pass and ssim_pass

    # ── Format ────────────────────────────────────────────────────────────────
    card = f"""
{'='*60}
REALIZEMED — DE-IDENTIFICATION SCORECARD
{'='*60}
Model:     {model_name}
Date:      {date.today().isoformat()}
Holdout N: {n_patients} patients, {n_images} images

PRIVACY
  Rank-1 re-id rate:      {_fmt(rank1, 4)}  (target ≤ {_fmt(random_base*2 if random_base else None, 4)})  {_status(reid_pass)}
  vs. random baseline:    {_fmt(ratio, 2)}×  (target ≤ 2.0×)
  Same-patient AUC:       {_fmt(sp_auc, 3)}  (target ≤ 0.55)  {_status(sp_auc is not None and sp_auc <= 0.55)}

UTILITY
  DR grading AUC (orig):  {_fmt(auc_orig, 3)}
  DR grading AUC (deid):  {_fmt(auc_deid, 3)}
  Preservation ratio:     {_fmt(pres_ratio, 3)}  (target ≥ 0.95)  {_status(util_pass)}
  AUC grade 0–1 (deid):   {_fmt(auc_low, 3)}
  AUC grade 2–4 (deid):   {_fmt(auc_high, 3)}

REALISM
  FID:                    {_fmt(fid, 1)}   (target < 30)    {_status(fid_pass)}
  Mean SSIM:              {_fmt(ssim_mean, 3)}  (target 0.65–0.85) {_status(ssim_pass)}
  LPIPS:                  {_fmt(lpips_mean, 3)}               {_status(lpips_pass)}

{'='*60}
OVERALL: {'✅ PASS' if overall_pass else '❌ FAIL'}
{'='*60}
"""

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        slug = model_name.replace(' ', '_').replace('+', 'plus')[:40]
        txt_path = output_dir / f'scorecard_{slug}_{date.today().isoformat()}.txt'
        json_path = output_dir / f'scorecard_{slug}_{date.today().isoformat()}.json'

        txt_path.write_text(card)
        json_path.write_text(json.dumps({
            'model':    model_name,
            'date':     date.today().isoformat(),
            'n_patients': n_patients,
            'n_images': n_images,
            'privacy':  reid_results,
            'utility':  utility_results,
            'realism':  realism_results,
            'overall_pass': overall_pass,
        }, indent=2))

        print(f'Scorecard saved: {txt_path}')

    return card
