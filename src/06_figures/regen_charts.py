#!/usr/bin/env python3
"""
Regenerate the two RESULTS charts in the manuscript from the LOCKED numbers:
  fig_percat_rgb_deeplab.png  (Fig. 1 -- per-category, primary RGB-DeepLabV3+)
  fig_robustness_2x2.png      (Fig. 2 -- base x backbone robustness, all 4 cells)

Numbers are hard-coded from the locked 2x2 results (master status table). Edit the
dicts below if a cell is re-run. Colour-blind-safe palette, light background.

  python regen_charts.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_FW, C_MLP, C_AG = '#8C8C8C', '#4C78A8', '#E45756'   # framework / MLP / agent

# ---- Fig 1: per-category, RGB-DeepLabV3+ ----
CATS = ['baseline', 'cameraJitter', 'badWeather', 'dynamicBackground',
        'intermittentObjectMotion', 'lowFramerate', 'nightVideos', 'PTZ',
        'shadow', 'thermal', 'turbulence']
SHORT = ['BSL', 'CJT', 'BDW', 'DBG', 'IOM', 'LFR', 'NVD', 'PTZ', 'SHD', 'THM', 'TBL']
FW  = [0.650, 0.727, 0.607, 0.612, 0.526, 0.494, 0.323, 0.135, 0.719, 0.665, 0.484]
MLP = [0.744, 0.681, 0.720, 0.693, 0.626, 0.538, 0.324, 0.403, 0.911, 0.664, 0.567]
AG  = [0.801, 0.763, 0.742, 0.842, 0.683, 0.568, 0.356, 0.359, 0.926, 0.678, 0.540]


def fig_percat(out='fig_percat_rgb_deeplab.png'):
    x = np.arange(len(CATS)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.bar(x - w, FW,  w, label='framework (fixed rule)', color=C_FW)
    ax.bar(x,     MLP, w, label='+ supervised MLP',       color=C_MLP)
    ax.bar(x + w, AG,  w, label='+ DRL agent (ours)',     color=C_AG)
    ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=8)
    ax.set_ylabel('category $F$-measure'); ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=8, ncol=3, frameon=False)
    ax.grid(axis='y', alpha=0.3, lw=0.5); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight')
    print('saved ->', out)


# ---- Fig 2: base x backbone robustness (category-mean) ----
CELLS = ['RGB\nDeepLabV3+', 'RGB\nPSPNet', 'Gray\nDeepLabV3+', 'Gray\nPSPNet']
R_FW  = [0.540, 0.532, 0.527, 0.524]
R_MLP = [0.625, 0.632, 0.599, 0.615]
R_AG  = [0.660, 0.668, 0.639, 0.652]


def fig_robust(out='fig_robustness_2x2.png'):
    x = np.arange(len(CELLS)); w = 0.25
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.bar(x - w, R_FW,  w, label='framework (fixed rule)', color=C_FW)
    ax.bar(x,     R_MLP, w, label='+ supervised MLP',       color=C_MLP)
    ax.bar(x + w, R_AG,  w, label='+ DRL agent (ours)',     color=C_AG)
    for xi, v in zip(x + w, R_AG):
        ax.text(xi, v + 0.012, f'{v:.3f}', ha='center', fontsize=8, color=C_AG)
    ax.set_xticks(x); ax.set_xticklabels(CELLS, fontsize=9)
    ax.set_ylabel('category-mean $F$-measure'); ax.set_ylim(0, 0.8)
    ax.legend(loc='upper center', fontsize=8, ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis='y', alpha=0.3, lw=0.5); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight')
    print('saved ->', out)


if __name__ == '__main__':
    fig_percat()
    fig_robust()
