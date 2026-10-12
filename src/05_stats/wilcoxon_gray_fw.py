#!/usr/bin/env python3
"""
Exact paired Wilcoxon (agent vs. framework AND agent vs. MLP) over the 53 CDnet
held-out scenes, for the two GRAYSCALE cells. Fills the two "<1e-5" bounds in
Table 5 with exact p-values.

Run on the machine that has the head-to-head result JSONs:

  python wilcoxon_gray_fw.py --files headtohead_gray.json headtohead_gray_psp.json

It is schema-tolerant: it locates the per-scene dict and the framework / agent /
mlp F1 under any of the common key spellings we used. Paste the printout back.
"""
import argparse, json, sys
import numpy as np
from scipy.stats import wilcoxon

FW_KEYS   = ['framework', 'fw', 'comm', 'stage3', 'base_fw', 'liang']
AG_KEYS   = ['framework+agent', 'agent', 'f1', 'drl', 'rl']
MLP_KEYS  = ['framework+mlp', 'mlp', 'supervised', 'sup', 'bc']


def find_per_scene(d):
    for k in ('per_scene', 'scene_f1', 'scenes', 'per_scene_f1'):
        if isinstance(d.get(k), dict):
            return d[k]
    # fall back: the top-level dict itself looks per-scene if values are dicts
    if all(isinstance(v, dict) for v in d.values()):
        return d
    raise SystemExit('could not locate a per-scene dict in this JSON')


def pick(scene_rec, keys):
    for k in keys:
        if k in scene_rec and isinstance(scene_rec[k], (int, float)):
            return float(scene_rec[k])
    return None


def paired(per_scene, a_keys, b_keys):
    A, B = [], []
    for rec in per_scene.values():
        a, b = pick(rec, a_keys), pick(rec, b_keys)
        if a is not None and b is not None:
            A.append(a); B.append(b)
    return np.array(A), np.array(B)


def report(tag, agent, other, name):
    if len(agent) == 0:
        print(f'  [{tag}] agent vs {name}: keys not found in this file'); return
    wins = int(np.sum(agent > other)); losses = int(np.sum(agent < other))
    diff = agent - other
    try:
        stat, p = wilcoxon(agent, other, zero_method='wilcox', alternative='two-sided')
    except ValueError as e:
        p = float('nan'); stat = float('nan')
        print(f'  [{tag}] wilcoxon note: {e}')
    print(f'  [{tag}] agent vs {name:9s}: n={len(agent)}  '
          f'agent>{name} {wins}/{len(agent)} (losses {losses})  '
          f'mean d={diff.mean():+.4f}  W={stat:.1f}  p={p:.3e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True,
                    help='head-to-head JSONs, e.g. headtohead_gray.json headtohead_gray_psp.json')
    args = ap.parse_args()
    for fp in args.files:
        print(f'\n=== {fp} ===')
        d = json.load(open(fp))
        ps = find_per_scene(d)
        print(f'  {len(ps)} scenes')
        ag, fw = paired(ps, AG_KEYS, FW_KEYS)
        report('FW ', ag, fw, 'framework')
        ag2, mlp = paired(ps, AG_KEYS, MLP_KEYS)
        report('MLP', ag2, mlp, 'mlp')


if __name__ == '__main__':
    main()
