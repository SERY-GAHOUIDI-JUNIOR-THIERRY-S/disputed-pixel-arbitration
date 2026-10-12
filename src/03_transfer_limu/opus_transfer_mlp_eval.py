#!/usr/bin/env python3
"""
================================================================================
MLP-ONLY TRANSFER EVAL  (no agent needed)
================================================================================
Scores a pre-trained supervised MLP arbiter zero-shot on a target dataset's cache,
reporting base(framework) / guidance / framework+MLP / oracle. Use it for the
LIMU-under-PSPNet generalisation test: train the MLP on CDnet-PSPNet disputes, then
apply it zero-shot to LIMU with PSPNet guidance. The MLP stands in for the agent
(agent ~ MLP, established in the ablation), so this estimates how a PSPNet-guided
agent would generalise -- without training one.

USAGE
  python opus_transfer_mlp_eval.py --dataset limu \
     --models_dir models/cpb_limu --cache_dir drl_cache_limu_pspnet \
     --sup_ckpt models/sup_gate_all_pspnet/sup.pth
================================================================================
"""
import argparse, json, os, sys, importlib
import numpy as np

T = None
for _n in ('opus_final_project_metrics', 'opus_final_project_metrics_ablation',
           'opus_final_drl_train_junior_enhanced_with_patch_cnn', 'opus_final_drl_train_junior'):
    try:
        T = importlib.import_module(_n); break
    except Exception:
        continue
if T is None:
    sys.exit('Could not import trainer module. Run from the project folder.')
F = T.F; ENV = T.ENV
SYS = ['guidance', 'framework', 'framework+mlp', 'oracle']

DATASETS = {
    'limu': ['limu/CameraParameter', 'limu/Intersection', 'limu/LightSwitch'],
    'cdnet': [f'{c}/{s}' for c, s in
              [(c, s) for c, ss in F.CDNET2014.items() for s in ss]],
}


def load_sup(path):
    import torch, torch.nn as nn
    s = torch.load(path, map_location='cpu'); dim = s['state_dim']
    net = (nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                         nn.Linear(64, 2)) if s.get('sup_model', 'mlp') != 'linear'
           else nn.Linear(dim, 2))
    net.load_state_dict(s['sup_state']); net.eval()
    return net, np.asarray(s['mu'], np.float32), np.asarray(s['sd'], np.float32)


def score_scene(sc, sup):
    import torch
    net, mu, sd = sup
    acc = {s: [0, 0, 0, 0] for s in SYS}
    for i in range(len(sc.nums)):
        e = sc.env(i); r, c = e.dispute_ij()
        base = e.baseline_mask()
        guid = e._to_native((e.guid_fg > 0).astype(np.uint8) * 255)
        if len(r) == 0:
            mm = base
        else:
            Xn = ((e.state(r, c).astype(np.float32) - mu) / sd).astype(np.float32)
            with torch.no_grad():
                sact = net(torch.from_numpy(Xn)).argmax(-1).numpy()
            mm = T._segment_gate3(e, sact, r, c)
        of = e.s1_default.copy()
        if len(r):
            of[r, c] = (e.gtfg_fit[r, c].astype(np.uint8) * 255)
        om = e._to_native(of)
        for name, mask in (('guidance', guid), ('framework', base),
                           ('framework+mlp', mm), ('oracle', om)):
            for k, v in enumerate(e.counts(mask)):
                acc[name][k] += v
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='limu', choices=list(DATASETS))
    ap.add_argument('--models_dir', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--sup_ckpt', required=True)
    ap.add_argument('--save_dir', default='models/transfer_mlp')
    ap.add_argument('--win_size', type=int, default=10)
    ap.add_argument('--af', type=float, default=2.5); ap.add_argument('--floor_min', type=float, default=6.0)
    ap.add_argument('--bth_pixels', type=float, default=7.0); ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--sigema', type=float, default=0.8)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    root = T.find_root()
    envargs = dict(af=args.af, floor=args.floor_min, bth=args.bth_pixels, ref=args.ref, sigema=args.sigema)
    sup = load_sup(args.sup_ckpt)
    scenes = T.build_scenes(DATASETS[args.dataset], args, root, envargs)
    if not scenes:
        sys.exit('No scenes built — check --cache_dir / --models_dir.')

    pool = {s: [0, 0, 0, 0] for s in SYS}; scene_f1 = {}
    for sc in scenes:
        acc = score_scene(sc, sup)
        scene_f1[sc.sid] = {s: ENV.f1_from_counts(*acc[s]) for s in SYS}
        for s in SYS:
            for k in range(4):
                pool[s][k] += acc[s][k]
        print(f'  {sc.sid:<26} ' + '  '.join(f'{s.split("+")[-1][:4]}={scene_f1[sc.sid][s]:.3f}' for s in SYS))

    pooled = {s: ENV.f1_from_counts(*pool[s]) for s in SYS}
    scene_mean = {s: float(np.mean([scene_f1[x][s] for x in scene_f1])) for s in SYS}
    print('\n' + '=' * 66)
    print(f'  MLP TRANSFER  dataset={args.dataset}  cache={args.cache_dir}')
    print('=' * 66)
    print(f"  {'system':<16}{'pooled':>10}{'scene-mean':>13}")
    for s in SYS:
        print(f"  {s:<16}{pooled[s]:>10.4f}{scene_mean[s]:>13.4f}")
    print('  ' + '-' * 46)
    print(f"  framework+mlp - framework : pooled {pooled['framework+mlp']-pooled['framework']:+.4f}")
    print(f"  framework+mlp - guidance  : pooled {pooled['framework+mlp']-pooled['guidance']:+.4f}")
    print('=' * 66)
    json.dump({'dataset': args.dataset, 'pooled': pooled, 'scene_mean': scene_mean,
               'scene_f1': scene_f1}, open(os.path.join(args.save_dir, f'transfer_mlp_{args.dataset}.json'), 'w'), indent=2)
    print('  saved -> ' + os.path.join(args.save_dir, f'transfer_mlp_{args.dataset}.json'))


if __name__ == '__main__':
    main()