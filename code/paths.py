#!/usr/bin/env python3
"""Default locations. Override with environment variables (or with the command-line options of each script).

DPA_DATA     folder holding limu/, wallflower/, lasiesta/ (and cdnet2014/)   default: <repo>/data
DPA_CACHE    folder holding the guidance caches                                default: <repo>/caches
DPA_RESULTS  folder holding the result files (per-frame counts)                default: <repo>/results
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('DPA_DATA', os.path.join(REPO, 'data'))
CACHE = os.environ.get('DPA_CACHE', os.path.join(REPO, 'caches'))
RESULTS = os.environ.get('DPA_RESULTS', os.path.join(REPO, 'results'))
LIMU = os.path.join(DATA, 'limu')
WALLFLOWER = os.path.join(DATA, 'wallflower')
LASIESTA = os.path.join(DATA, 'lasiesta')
CDNET = os.path.join(DATA, 'cdnet2014')
MODELS = {'DL': os.path.join(REPO, 'models', 'deeplabv3plus', 'mlp_ALL+noA_final.npz'),
          'PSP': os.path.join(REPO, 'models', 'pspnet', 'mlp_pspnet_ALL_noA_final.npz')}
CACHES = {'limu': {'DL': os.path.join(CACHE, 'limu_deeplab'), 'PSP': os.path.join(CACHE, 'limu_pspnet')},
          'wallflower': {'DL': os.path.join(CACHE, 'wallflower_deeplab'), 'PSP': os.path.join(CACHE, 'wallflower_pspnet')},
          'lasiesta': {'DL': os.path.join(CACHE, 'lasiesta_deeplab'), 'PSP': os.path.join(CACHE, 'lasiesta_pspnet')}}
CPB_LASIESTA = os.path.join(REPO, 'runs', 'lasiesta_cpb')      # trained LASIESTA CPB models (lasiesta_cpb_eval.py --out)
