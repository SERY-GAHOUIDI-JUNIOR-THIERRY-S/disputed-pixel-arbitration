# Disputed-pixel arbitration for cross-scene foreground segmentation

Code, frozen models and per-frame results for a learned arbiter that settles the pixels on which an
unsupervised background model (CPB, co-occurrence pixel-block) and a semantic segmentation network
(DeepLabV3+ or PSPNet, ADE20K) disagree. The arbiter is a small numpy MLP (25 → 64 → 64 → 1, 5,889 parameters)
trained on CDnet2014 only and applied frozen to LIMU, WallFlower and LASIESTA.

The accompanying manuscript is under review; a citation will be added here after publication.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | The full pipeline: CPB re-implementation (R6), the base communication framework (re-implementation), feature extraction and training of the arbiter, frozen evaluation on the target datasets, SemanticBGS, classical BGSLibrary baselines, result tables and statistics |
| `models/` | Frozen arbiters with their selected thresholds: `deeplabv3plus/` (τ = 0.675), `pspnet/` (τ = 0.725), `fusion_control/` (minimal learned fusion) |
| `results/` | Per-frame counts (TP, FP, FN per scored frame) and summaries for every system and sequence, the comparison tables (`comparison/COMPARISON.md`) and the ZBS detections |
| `guidance/` | Generators of the guidance caches (mmsegmentation, ADE20K) and the Kaggle notebook used for PSPNet |
| `external/zbs/` | Scripts used to run ZBS (CVPR 2023) under the same protocol |

The datasets and the guidance caches are not stored in the repository (see below).

## Installation

Python 3.10 or newer.

```
pip install -r requirements.txt
```

Optional: `pip install pybgs` (BGSLibrary 3.3.1) for `bgs_baselines.py`; mmsegmentation 1.2.2 with PyTorch only to
regenerate the guidance caches; the ZBS repository and its environment only for `external/zbs/`.

## Quick check without any dataset

The tables and the statistical tests are rebuilt from the per-frame counts in `results/`:

```
cd code
python compare_tables.py
python stats_tests.py --dataset lasiesta --ref "Arbiter (PSPNet)" --comps "CPB|PSPNet alone|Liang framework (PSPNet)|SemanticBGS on CPB (PSPNet)|SemanticBGS on SuBSENSE (PSPNet)|PAWCS|SuBSENSE|ZBS|Arbiter (DeepLab)"
python stats_tests.py --dataset limu --ref "Arbiter (DeepLab)" --comps "CPB|DeepLabV3+ alone|Liang framework (DeepLab)|Minimal fusion (DeepLab)|SemanticBGS on CPB (DeepLab)|SuBSENSE|PAWCS|ZBS"
```

Output: `results/comparison/COMPARISON.md` and `results/stats/`.

## Data

Download the datasets from their providers and place them as follows (or set `DPA_DATA`, see `code/paths.py`):

```
data/
  cdnet2014/    CDnet2014 (changedetection.net)                 source domain only
  limu/         LIMU, Kyushu University (limu.ait.kyushu-u.ac.jp/dataset/en/)
  wallflower/   WallFlower (Toyama et al., ICCV 1999)
  lasiesta/     LASIESTA (Cuevas et al., CVIU 2016)
```

Guidance caches (per-frame ADE20K foreground probabilities) go to `caches/` (or set `DPA_CACHE`):

```
caches/
  limu_deeplab/  limu_pspnet/  wallflower_deeplab/  wallflower_pspnet/  lasiesta_deeplab/  lasiesta_pspnet/
```

They are attached to the GitHub release of this repository, or can be regenerated with the scripts in `guidance/`.

## Reproducing the results

All commands run from `code/`. Nothing is tuned on the target datasets: each target command trains the CPB on the
sequence's own label-free training window, then applies the frozen guidance, the frozen arbiter and its threshold.

**1. Source domain (CDnet2014): CPB, features, split, ablation, training**

```
python -u cdnet_cpb_eval.py --root ../data/cdnet2014 --mode frozen
python -u arbiter_mlp.py extract --root ../data/cdnet2014 --cache <cdnet_guidance_cache> --cpb_out cdnet_runs --work arbiter_runs --n_frames 150
python -u arbiter_mlp.py split   --work arbiter_runs
python -u arbiter_mlp.py ablate  --work arbiter_runs
python -u arbiter_mlp.py train   --work arbiter_runs --features "ALL,-A"
```

The split uses seed 2026 (39 training and 10 validation videos, PTZ excluded). The frozen models obtained this way
are the ones in `models/`.

**2. Frozen evaluation on the target datasets**

```
python -u apply_arbiter.py limu       --limu ../data/limu             --guid ../caches/limu_deeplab       --model ../models/deeplabv3plus/mlp_ALL+noA_final.npz --json ../models/deeplabv3plus/mlp_ALL+noA_final.json --out ../runs/arbiter_deeplab/limu --breakdown
python -u apply_arbiter.py wallflower --wf ../data/wallflower         --guid ../caches/wallflower_deeplab --model ../models/deeplabv3plus/mlp_ALL+noA_final.npz --json ../models/deeplabv3plus/mlp_ALL+noA_final.json --out ../runs/arbiter_deeplab/wallflower
python -u lasiesta_cpb_eval.py --root ../data/lasiesta --rule A --out ../runs/lasiesta_cpb
python -u apply_arbiter.py lasiesta   --lasiesta ../data/lasiesta     --guid ../caches/lasiesta_deeplab   --model ../models/deeplabv3plus/mlp_ALL+noA_final.npz --json ../models/deeplabv3plus/mlp_ALL+noA_final.json --cpb_out ../runs/lasiesta_cpb --out ../runs/lasiesta_arbiters/deeplab
```

For PSPNet guidance, use the `*_pspnet` caches and `../models/pspnet/mlp_pspnet_ALL_noA_final.npz` (with its `.json`).
WallFlower takes a few minutes; LIMU about 20 minutes; LASIESTA longer (20 sequences).

**3. Compared methods under the same protocol**

```
python -u semantic_bgs.py --dataset limu --guid deeplab=../caches/limu_deeplab,pspnet=../caches/limu_pspnet --out ../runs/semantic_bgs
python -u bgs_baselines.py --datasets limu,wallflower,lasiesta --out ../runs/classical_bgs
```

ZBS: see the header of `external/zbs/zbs_detect.py` (detection, GPU) and `external/zbs/zbs_post.py`
(background model, selection and scoring). The authors' released code of the base framework is available at
github.com/MeteoorLiu/Venus (branch MeteoorLiu-SUMC); its results under our protocol are in `results/authors_code/`.

**4. Tables and statistics from your own runs**

Point `compare_tables.py` and `stats_tests.py` at your run folders with their command-line options, or copy the
run outputs into `results/` with the same folder names.

## Metrics

Main metric: per-frame F-measure (precision and recall averaged over the scored frames that contain foreground),
reported with the false-positive rate on frames whose ground truth is empty. Pooled F is reported separately.
WallFlower: F on the official evaluated frame of each sequence; MovedObject by specificity.

## Licence

Code: MIT licence (see `LICENSE`). The datasets, the guidance networks and the third-party methods keep their own
licences and are not redistributed here.

## Contact

Zhe Chen (corresponding author), zhechen@nuaa.edu.cn, College of Computer Science and Technology,
Nanjing University of Aeronautics and Astronautics.
