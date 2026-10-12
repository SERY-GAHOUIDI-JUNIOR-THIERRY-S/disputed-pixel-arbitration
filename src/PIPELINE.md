# Pipeline — the files actually used, in order

This folder contains the curated set of scripts that produce the paper's results,
selected from the full working tree and arranged in execution order. Each file keeps
its own detailed `USAGE` header; the commands below are the representative entry points.

> **Complete.** All code the pipeline needs is included. The only external artefacts you
> supply yourself are the public datasets (CDnet2014, LIMU) and the MMSegmentation guidance
> checkpoints — the DeepLabV3+ and PSPNet ADE20K `.pth`/config files named in
> `00_core/opus_guidance_config.py`, which you download from MMSeg into your `models/`
> folder. These are third-party weights and large, so they are not committed to the repo.

---

## 00_core — the engine (imported by everything else)

- **`opus_final_hod_framework.py`** — the CPB + HoD + 3-stage model-communication
  framework (the faithful Liang et al. reproduction: `k_stage1/2/3`, `HoDPipeline`,
  metrics, guidance factory). Imported everywhere as `F`.
- **`opus_drl_env_junior_enhanced_with_patch_cnn.py`** — the RL environment: `FrameEnv`,
  the 23-D disputed-pixel state, the patch extractor, the dispute set, `f1_from_counts`.
  Imported everywhere as `ENV`.
- **`opus_final_project_metrics.py`** — the consolidated DRL trainer + cross-validation +
  evaluation engine ("junior v4": 3-way gate, REINFORCE + behaviour-cloning, per-frame
  baseline, `build_scenes`, `make_kfold`, `train_one`, `evaluate`, `build_gate`,
  `_segment_gate3`). Imported as `T` by the CV / figure / oracle scripts. This is the
  module that produced the reported numbers.
- **`opus_guidance_config.py`** — the corrected `ADE20KGuidance` wrapper (DeepLabV3+/PSPNet
  via MMSegmentation) with the right 12 ADE20K foreground class ids. Run its precompute use
  (step 2a) **after** the CPB models are trained; it caches guidance so later stages run
  from the `.npz` cache. Needs the MMSeg checkpoints in `models/` (see the note at top).

You do not run the core files directly (except `opus_final_project_metrics.py --stage
precompute`); the stage scripts below import them.

## 01_train_cpb — train the per-scene CPB background models (make the `.pkl`s)

Run first. Produces `models/cpb_numba_rgb/cpb_paper_<cat>_<scene>.pkl` (and a grayscale
set with the gray flag).

- **`opus_final_hod_training.py`** — the Numba CPB trainer (Pearson support selection,
  b = mean, sigma = std; the paper's plain-Gaussian fit). This is the channels-aware core
  that produced the RGB models.
- **`opus_cpb_train_extra_limu_cdnet.py`** — thin CDnet wrapper around that core, used to
  build the controlled grayscale-vs-RGB model sets from one trainer.

```bash
python opus_final_hod_training.py --output_dir models/cpb_numba_rgb   # see file header for flags
```

## 02_main_experiment — the CDnet results (Tables 3, 5, 6)

Order: precompute guidance → run the 5 CV folds → aggregate → oracle → ablations.

```bash
# (2a) cache guidance once (uses 00_core; needs opus_guidance_config.py)
python ../00_core/opus_final_project_metrics.py --stage precompute \
    --models_dir models/cpb_numba_rgb --guidance deeplab --cache_dir drl_cache

# (2b) agent: one process per fold, then the capacity-matched MLP (--no_patch)
for k in 0 1 2 3 4; do python cv_fold.py --fold $k --models_dir models/cpb_numba_rgb \
    --cache_dir drl_cache --steps 20000 --win_size 10 --bc_coef 1.0 \
    --save_dir models/drl_gate_rgb_cv; done

# (2c) aggregate the folds -> the main category/scene/pooled numbers
python cv_aggregate.py --save_dir models/drl_gate_rgb_cv --out cv_rgb_agent.json
```

- **`cv_fold.py`** — trains + evaluates one CV fold (memory-safe, one process each).
- **`cv_aggregate.py`** — combines the folds into the final CV summary (Table 3 / per-cat).
- **`oracle_ceiling.py`** — the oracle upper bound on disputed pixels (the 0.825 row).
- **`opus_backbone_ablation.py`** — the 2×2 representation × backbone grid (Table 6).
- **`opus_final_project_ablation_study_supervised_vs_drl.py`** — the single-harness
  head-to-head that establishes RL-agent vs capacity-matched MLP (Table 5).

## 03_transfer_limu — cross-dataset transfer (Section 4.5)

```bash
python opus_convert_limu.py            # LIMU -> CDnet layout
python opus_cpb_train_extra_limu.py    # per-scene CPB on LIMU
python opus_train_transfer_arbiters.py # train agent + MLP on CDnet (fair transfer)
python opus_transfer_limu_evaluation.py --stage eval   # zero-shot LIMU scores
```

- **`opus_convert_limu.py`** — converts the LIMU dataset into the CDnet folder layout.
- **`opus_cpb_train_extra_limu.py`** — trains CPB per-scene on LIMU.
- **`opus_train_transfer_arbiters.py`** — trains the DRL agent and the MLP on identical
  CDnet scenes so the target-domain difference is the method, not the data.
- **`opus_transfer_limu_evaluation.py`** — applies a CDnet-trained arbiter zero-shot to LIMU.
- **`opus_transfer_5cv_folds_training_eval_against_cv_agents.py`** — repeats the transfer
  across all 5 folds for a mean ± std claim.
- **`opus_transfer_mlp_eval.py`** — MLP-only zero-shot transfer (the PSPNet generalisation
  estimate).
- **`opus_limu_indomain_cv.py`** — LIMU in-domain leave-one-scene-out CV.

## 04_diagnostics — reproduction-gap analysis (Table 2 and its figures)

- **`opus_paper_metrics_table.py`** — CPB-alone / guidance / framework in all three
  aggregations (the baseline rows).
- **`diag_resolution.py`** — native-vs-fit scoring diagnostic (why 0.513 vs 0.428).
- **`validate_protocol.py`** — shows uniform-N frame sampling converges to the full-ROI
  score (licenses the 60-frame protocol).
- **`cpb_comm_diagnose.py`** — stage-by-stage where F1 is lost (under-detection diagnosis).

## 05_stats — significance and intervals

- **`wilcoxon_gray_fw.py`** — paired Wilcoxon (agent vs framework, agent vs MLP).
- **`bootstrap_ci.py`** — bootstrap confidence intervals for the main gains.

## 06_figures — regenerate the manuscript figures

- **`regen_charts.py`** — the per-category and robustness charts.
- **`opus_make_decomp.py`** — the per-stage decomposition panel (Fig. per-stage).
- **`opus_make_errormaps.py`** — the CPB under-detection error maps.
- **`opus_make_qualitative_agent.py`** / **`make_qualitative.py`** — qualitative panels.
- **`opus_export_figure_panels.py`** — exports the raw figure panels.

## 07_checks

- **`verify_pipeline.py`** — asserts the run will use RGB + all-64 + paper Stage-2
  (a GO / NO-GO pre-flight check). Run it before 02.

---

## Notes on what was left out (and why)

From the full tree, these were **superseded earlier versions** of files kept above and
were deliberately excluded to keep the pipeline unambiguous. Say the word if you want any
added back:

- Earlier RL environments: `opus_drl_env.py`, `opus_drl_env_junior.py`
  (superseded by `..._enhanced_with_patch_cnn`).
- Earlier DRL trainers: `opus_final_drl_train.py`, `opus_final_drl_train_junior.py`,
  `opus_final_drl_train_junior_enhanced_with_patch_cnn.py` (the last is a same-code twin of
  `opus_final_project_metrics.py`, which is the name the final scripts import).
- Earlier framework versions: `liang_d.py` (v5), `liang_v7.py` (v7), `cpb_d.py`,
  `oracle_framework.py`, `opus_liang_hod_implemented.py`,
  `opus_liang_framework_three_stages_with_guidance_with_single_chanel.py`,
  `opus_liang_hod_framework_searching_increase_baseline.py`.
- Alternative / older CPB trainers: `opus_final_hod_training_v2.py`,
  `opus_cpb_training_with_numba_on_first_400_frame_with_plain_std.py` (v5),
  `opus_cpb_training_with_numba_on_first_400_frames_single_chanel.py` (v6, standalone with
  built-in viz), `opus_cpb_train_robust.py`, `opus_liang_hod_training_cpb.py`.
- Exploration / sanity one-offs: `opus_cpb_viz.py`, `opus_cpb_blockrule_sanity.py`,
  `opus_eval_cpbfix.py`, `opus_eval_gatesweep.py`, `opus_eval_naegate.py`,
  `opus_faithful_eval_check_up.py`, `compare_block_frac.py`,
  `opus_check_limu_number_frames.py`, `opus_final_guidance_index_check.py`,
  `diag_iom_transfer.py`, `k_stage2_paper_patch.py` (a one-off patch snippet).

## Status: resolved

1. **`opus_guidance_config.py`** — included in `00_core/` (the corrected version).
2. **CPB trainer** — confirmed: `opus_final_hod_training.py` is the core that produced the
   RGB models (`opus_cpb_train_extra_limu_cdnet.py` imports it as its CPB core).

The only things not in this folder are the public datasets and the MMSeg guidance
checkpoints, which you download separately (see the note at the top).
