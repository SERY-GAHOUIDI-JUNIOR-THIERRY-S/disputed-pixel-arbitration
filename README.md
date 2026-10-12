# Adaptive Cross-Scene Foreground Segmentation via Learned Disputed-Pixel Arbitration

This repository accompanies the paper *"Adaptive Cross-Scene Foreground Segmentation via
Learned Disputed-Pixel Arbitration"* and contains the manuscript, its LaTeX source, the
analysis and figure-generation code, and the raw result files behind the reported tables
and figures.

**Authors:** Sery Gahouidi Junior Thierry, Zhe Chen (corresponding author)
**Affiliation:** College of Computer Science and Technology, Nanjing University of
Aeronautics and Astronautics (NUAA), Nanjing 211106, China
**Contact:** zhechen@nuaa.edu.cn · juniorsery00@gmail.com

<!-- After you create the Zenodo release, paste the DOI badge here, e.g.:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->

## Overview

Cross-scene video foreground segmentation couples two complementary but imperfect sources
of evidence: an unsupervised Co-occurrence Pixel-Block (CPB) background model that adapts
per scene but drifts on complex dynamics, and a supervised segmentation network that is
accurate on its training distribution but weaker on unseen scenes. The model-communication
framework of Liang et al. combines the two, but resolves the pixels on which they disagree
with a single fixed, hand-designed rule.

This work replaces that fixed resolution with a lightweight, context-aware **arbiter** that
learns, per disputed pixel, whether to trust the unsupervised model, trust the guidance, or
keep the framework's own decision. The arbiter is trained with a policy-gradient objective
(REINFORCE with a behaviour-cloning auxiliary) over a 23-dimensional state built entirely
from quantities the framework already computes. Only disputed pixels are ever changed; the
underlying models and the majority of agreed pixels are left untouched.

## Key results (CDnet2014, 5-fold scene-disjoint cross-validation)

| System (primary RGB / DeepLabV3+ configuration) | Category-mean F-measure |
|---|---|
| Re-implemented static framework (faithful base) | 0.540 |
| + capacity-matched supervised MLP control | 0.625 |
| + learned arbiter (DRL agent) | **0.660** |
| Oracle ceiling on disputed pixels | 0.825 |

The learned arbiter improves the static framework by **+0.120** category-mean, and beats the
capacity-matched supervised control by a stable **+0.037 ± 0.002** across a full 2×2 of two
CPB representations (grayscale, RGB) and two guidance backbones (DeepLabV3+, PSPNet), with
the advantage statistically significant (paired Wilcoxon) in every configuration. The policy
also transfers zero-shot to the LIMU dataset.

## Repository structure

```
.
├── paper/
│   ├── PAA_submission_SERY.pdf      # compiled manuscript
│   └── latex/                       # full LaTeX source that produces the PDF
│       ├── PAA_submission_SERY.tex  # main manuscript source
│       ├── PAA_submission_SERY.bbl  # compiled bibliography
│       ├── sn-jnl.cls, sn-*.bst     # Springer Nature journal class + styles
│       ├── palette.tex              # shared colour palette for the diagrams
│       ├── fig_ours_flow.tex        # TikZ source: our method flow diagram
│       ├── fig_liang_flow.tex       # TikZ source: base-framework flow diagram
│       └── fig_*.png / fig_*.pdf    # figures included by the manuscript
├── src/                             # the full pipeline, numbered in execution order
│   ├── PIPELINE.md                  # ordered walkthrough + run commands for every stage
│   ├── 00_core/                     # the engine, imported by every stage
│   │   ├── opus_final_hod_framework.py              # CPB + HoD + 3-stage framework (F)
│   │   ├── opus_drl_env_junior_enhanced_with_patch_cnn.py  # RL environment, 23-D state (ENV)
│   │   ├── opus_final_project_metrics.py            # DRL trainer + CV + eval engine (T)
│   │   └── opus_guidance_config.py                  # DeepLabV3+/PSPNet ADE20K guidance
│   ├── 01_train_cpb/                # train the per-scene CPB background models (.pkl)
│   ├── 02_main_experiment/          # CDnet CV, oracle, head-to-head, 2×2 backbone (Tables 3/5/6)
│   ├── 03_transfer_limu/            # cross-dataset transfer to LIMU (Section 4.5)
│   ├── 04_diagnostics/              # reproduction-gap analysis (Table 2)
│   ├── 05_stats/                    # paired Wilcoxon, bootstrap CIs
│   ├── 06_figures/                  # regenerate the manuscript figures
│   └── 07_checks/                   # pre-flight pipeline verification
├── results/                         # raw result dumps behind the tables/figures
│   ├── headtohead.json              # agent vs. MLP vs. framework head-to-head
│   ├── cv_rgb_agent.json            # per-fold CV results, RGB agent
│   ├── cv_rgb_mlp.json              # per-fold CV results, RGB supervised control
│   ├── cpb_all_dec_all64.json       # CPB diagnosis sweep
│   ├── cpb_all_dec_blockmean.json   # CPB diagnosis (block-mean variant)
│   ├── full_gate_deeplab_fl6.json   # guidance-reliability gate results
│   └── history.json                 # training history
├── README.md
├── CITATION.cff
├── requirements.txt
├── LICENSE                          # MIT, for the code
└── LICENSE-paper                    # CC-BY-4.0, for the manuscript and figures
```

## Data

Experiments use two public benchmarks, which are **not redistributed here**:

- **CDnet2014** — http://changedetection.net
- **LIMU** — the LIMU moving-object dataset (Kyushu University).

Download them from their original sources, subject to their own terms.

## Reproducing the paper

The full pipeline is under `src/`, with folders numbered in execution order
(`00_core` → `07_checks`). **Start with [`src/PIPELINE.md`](src/PIPELINE.md)**, which walks
through each stage and gives the exact run commands: train the CPB models, cache the guidance,
run the 5-fold cross-validation, then the ablations, transfer, diagnostics, statistics and
figures. Algorithm 1 in the paper specifies the training procedure the engine implements
(REINFORCE with the behaviour-cloning auxiliary, per-frame baseline, annealed BC weight
1.0 → 0.2).

The core engine (`src/00_core`) is imported by every stage and is not run directly, except
`opus_final_project_metrics.py --stage precompute`, which caches the guidance. Paths in the
scripts assume the standard layout (`data/cdnet2014/...`, `models/...`, caches under
`drl_cache*`); adjust them to your setup. A representative set of raw result files is provided
under `results/`. Besides the datasets above, you also supply the MMSegmentation guidance
checkpoints named in `src/00_core/opus_guidance_config.py` (third-party weights, not committed
here).

To rebuild the manuscript PDF:

```bash
cd paper/latex
pdflatex PAA_submission_SERY
bibtex   PAA_submission_SERY
pdflatex PAA_submission_SERY
pdflatex PAA_submission_SERY
```

## Citation

If you use this work, please cite the paper. A machine-readable `CITATION.cff` is included
(GitHub will show a "Cite this repository" button). BibTeX:

```bibtex
@article{sery2026arbitration,
  title   = {Adaptive Cross-Scene Foreground Segmentation via Learned Disputed-Pixel Arbitration},
  author  = {Gahouidi Junior Thierry, Sery and Chen, Zhe},
  journal = {Pattern Analysis and Applications (under review)},
  year    = {2026}
}
```

## Acknowledgements

This work was supported by the National Natural Science Foundation of China under Grant No.
62172217. We build on the model-communication framework of Liang et al. (*Pattern
Recognition*, 2021).

## License

- **Code** (`src/`, scripts): MIT License — see [`LICENSE`](LICENSE).
- **Manuscript and figures** (`paper/`): Creative Commons Attribution 4.0 International
  (CC-BY-4.0) — see [`LICENSE-paper`](LICENSE-paper).
