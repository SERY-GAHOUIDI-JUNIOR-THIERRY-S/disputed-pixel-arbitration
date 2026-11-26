## LIMU (per-frame F; same frames, guidance and metric for every row)

| System | CameraParameter | Intersection | LightSwitch | **Mean** | Pooled F (mean) | FPR empty-GT (mean) |
|---|---|---|---|---|---|---|
| CPB (R6, frozen) | 0.674 | 0.688 | 0.580 | **0.647** | 0.512 | 0.044 |
| DeepLabV3+ alone | 0.800 | 0.566 | 0.621 | **0.662** | 0.671 | 0.003 |
| CPB + DeepLabV3+: Liang framework, our re-impl. | 0.818 | 0.725 | 0.589 | **0.711** | 0.577 | 0.045 |
| CPB + DeepLabV3+: Liang framework (block repl.), our re-impl. | 0.801 | 0.577 | 0.624 | **0.667** | 0.667 | 0.004 |
| Liang framework, authors' released code | 0.776 | 0.565 | 0.660 | **0.667** | 0.680 | – |
| **CPB + DeepLabV3+: framework + arbiter (ours)** | 0.824 | 0.733 | 0.745 | **0.767** | 0.652 | 0.008 |
| CPB + DeepLabV3+: oracle arbiter (ceiling) | 0.909 | 0.789 | 0.882 | **0.860** | 0.869 | 0.001 |
| PSPNet alone | 0.752 | 0.515 | 0.613 | **0.626** | 0.675 | 0.003 |
| CPB + PSPNet: Liang framework, our re-impl. | 0.800 | 0.721 | 0.602 | **0.708** | 0.565 | 0.045 |
| CPB + PSPNet: Liang framework (block repl.), our re-impl. | 0.767 | 0.540 | 0.609 | **0.639** | 0.669 | 0.004 |
| **CPB + PSPNet: framework + arbiter (ours)** | 0.839 | 0.721 | 0.702 | **0.754** | 0.586 | 0.040 |
| CPB + PSPNet: oracle arbiter (ceiling) | 0.915 | 0.783 | 0.885 | **0.861** | 0.893 | 0.000 |
| SemanticBGS rules on CPB + DeepLabV3+ | 0.820 | 0.684 | 0.655 | **0.720** | 0.609 | 0.012 |
| SemanticBGS rules on SuBSENSE + DeepLabV3+ | 0.852 | 0.444 | 0.822 | **0.706** | 0.542 | 0.009 |
| SemanticBGS rules on CPB + PSPNet | 0.826 | 0.687 | 0.655 | **0.723** | 0.604 | 0.013 |
| SemanticBGS rules on SuBSENSE + PSPNet | 0.852 | 0.443 | 0.828 | **0.708** | 0.547 | 0.008 |
| CPB + DeepLabV3+: minimal learned fusion (logistic on CPB vote, CPB score, guidance prob.) | 0.839 | 0.713 | 0.659 | **0.737** | 0.627 | 0.011 |
| SuBSENSE (BGSLibrary) | 0.766 | 0.439 | 0.811 | **0.672** | 0.441 | 0.018 |
| PAWCS (BGSLibrary) | 0.526 | 0.261 | 0.716 | **0.501** | 0.437 | 0.003 |
| LOBSTER (BGSLibrary) | 0.385 | 0.092 | 0.334 | **0.271** | 0.106 | 0.167 |
| ViBe (BGSLibrary) | 0.345 | 0.353 | 0.316 | **0.338** | 0.198 | 0.103 |
| GMM (MOG2) (BGSLibrary) | 0.382 | 0.242 | 0.697 | **0.440** | 0.360 | 0.004 |
| KDE (BGSLibrary) | 0.203 | 0.393 | 0.181 | **0.259** | 0.192 | 0.291 |
| PBAS (BGSLibrary) | 0.443 | 0.611 | 0.413 | **0.489** | 0.290 | 0.137 |
| ZBS (CVPR 2023), paper parameters | 0.808 | 0.731 | 0.793 | **0.777** | 0.742 | 0.004 |
| ZBS, CDnet-script group 1 (sensitivity) | 0.862 | 0.763 | 0.800 | **0.808** | 0.798 | 0.002 |
| ZBS, CDnet-script group 2 (sensitivity) | 0.833 | 0.763 | 0.774 | **0.790** | 0.765 | 0.004 |
| ZBS, CDnet-script group 3 (sensitivity) | 0.406 | 0.634 | 0.254 | **0.431** | 0.361 | 0.125 |

*As reported (other protocols / aggregation not specified; reference only):* Liang et al. ICASSP/PR 2021: CPB + DeepLabV3+ 0.842 / 0.771 / 0.769 (0.793); CPB 0.655 / 0.678 / 0.663 (0.665); SuBSENSE 0.674 / 0.653 / 0.693 (0.675); KDE 0.646 / 0.648 / 0.675 (0.656); GMM 0.637 / 0.642 / 0.674 (0.652); PBAS 0.658 / 0.641 / 0.459 (0.586).

## WallFlower (F on the evaluated frame; MovedObject = specificity)

| System | Bootstrap | Camouflage | FgAperture | LightSwitch | TimeOfDay | WavingTrees | **Mean F** | MovedObject |
|---|---|---|---|---|---|---|---|---|
| CPB (R6, frozen) | 0.451 | 0.745 | 0.740 | 0.841 | 0.308 | 0.681 | **0.628** | 0.920 |
| DeepLabV3+ alone | 0.819 | 0.986 | 0.975 | 0.614 | 0.375 | 0.980 | **0.791** | 1.000 |
| CPB + DeepLabV3+: Liang framework, our re-impl. | 0.752 | 0.924 | 0.940 | 0.840 | 0.347 | 0.975 | **0.796** | 0.921 |
| CPB + DeepLabV3+: Liang framework (block repl.), our re-impl. | 0.778 | 0.985 | 0.975 | 0.619 | 0.372 | 0.980 | **0.785** | 0.996 |
| Liang framework, authors' released code (af 3.5) | 0.562 | 0.980 | 0.849 | 0.290 | 0.244 | 0.980 | **0.651** | 0.927 |
| **CPB + DeepLabV3+: framework + arbiter (ours)** | 0.779 | 0.958 | 0.976 | 0.777 | 0.534 | 0.982 | **0.834** | 0.997 |
| CPB + DeepLabV3+: oracle arbiter (ceiling) | 0.862 | 0.997 | 0.994 | 0.951 | 0.600 | 0.993 | **0.900** | 1.000 |
| PSPNet alone | 0.711 | 0.977 | 0.958 | 0.671 | 0.233 | 0.977 | **0.755** | 1.000 |
| CPB + PSPNet: Liang framework, our re-impl. | 0.721 | 0.918 | 0.928 | 0.835 | 0.322 | 0.973 | **0.783** | 0.921 |
| CPB + PSPNet: Liang framework (block repl.), our re-impl. | 0.675 | 0.978 | 0.960 | 0.674 | 0.234 | 0.977 | **0.750** | 0.996 |
| **CPB + PSPNet: framework + arbiter (ours)** | 0.685 | 0.960 | 0.965 | 0.815 | 0.431 | 0.978 | **0.806** | 0.998 |
| CPB + PSPNet: oracle arbiter (ceiling) | 0.825 | 0.991 | 0.993 | 0.954 | 0.544 | 0.992 | **0.883** | 1.000 |
| SemanticBGS rules on CPB + DeepLabV3+ | 0.506 | 0.955 | 0.776 | 0.758 | 0.506 | 0.978 | **0.747** | 0.998 |
| SemanticBGS rules on SuBSENSE + DeepLabV3+ | 0.465 | 0.975 | 0.708 | 0.494 | 0.852 | 0.980 | **0.746** | 1.000 |
| SemanticBGS rules on CPB + PSPNet | 0.493 | 0.837 | 0.741 | 0.816 | 0.487 | 0.975 | **0.725** | 0.998 |
| SemanticBGS rules on SuBSENSE + PSPNet | 0.447 | 0.968 | 0.673 | 0.454 | 0.797 | 0.980 | **0.720** | 1.000 |
| CPB + DeepLabV3+: minimal learned fusion (logistic on CPB vote, CPB score, guidance prob.) | 0.603 | 0.931 | 0.943 | 0.736 | 0.489 | 0.864 | **0.761** | 0.943 |
| SuBSENSE (BGSLibrary) | 0.428 | 0.960 | 0.670 | 0.290 | 0.868 | 0.958 | **0.696** | 0.947 |
| PAWCS (BGSLibrary) | 0.611 | 0.968 | 0.623 | 0.307 | 0.384 | 0.958 | **0.642** | 0.957 |
| LOBSTER (BGSLibrary) | 0.756 | 0.968 | 0.784 | 0.247 | 0.651 | 0.909 | **0.719** | 0.971 |
| ViBe (BGSLibrary) | 0.430 | 0.910 | 0.608 | 0.117 | 0.109 | 0.927 | **0.517** | 0.979 |
| GMM (MOG2) (BGSLibrary) | 0.448 | 0.103 | 0.327 | 0.273 | 0.359 | 0.808 | **0.386** | 1.000 |
| KDE (BGSLibrary) | 0.621 | 0.974 | 0.822 | 0.276 | 0.334 | 0.881 | **0.651** | 0.892 |
| PBAS (BGSLibrary) | 0.347 | 0.844 | 0.608 | 0.244 | 0.591 | 0.936 | **0.595** | 1.000 |
| ZBS (CVPR 2023), paper parameters | 0.706 | 0.984 | 0.849 | 0.500 | 0.289 | 0.980 | **0.718** | 0.922 |
| ZBS, CDnet-script group 1 (sensitivity) | 0.426 | 0.948 | 0.667 | 0.560 | 0.462 | 0.980 | **0.674** | 0.965 |
| ZBS, CDnet-script group 2 (sensitivity) | 0.404 | 0.973 | 0.013 | 0.523 | 0.473 | 0.980 | **0.561** | 0.984 |
| ZBS, CDnet-script group 3 (sensitivity) | 0.694 | 0.954 | 0.576 | 0.461 | 0.234 | 0.980 | **0.650** | 0.489 |

*As reported (reference only):* PR 2021: CPB 0.671; CPB + DeepLabV3+ 0.720; DeepLabV3+ alone 0.788; CPB + PSPNet 0.805; PSPNet alone 0.973; SuBSENSE 0.671; KDE 0.683; GMM 0.644; PBAS 0.562. ICPR 2020: CPB + STAM 0.793.

## LASIESTA (rule A; per-frame F per category = mean of its 2 sequences; 20 sequences)

| System | SI | CA | OC | IL | MB | BS | CL | RA | SN | SU | **Mean** | Pooled F | FPR empty-GT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CPB (R6, frozen, rule A) | 0.867 | 0.891 | 0.787 | 0.370 | 0.871 | 0.601 | 0.888 | 0.881 | 0.749 | 0.779 | **0.769** | 0.734 | 0.026 |
| DeepLabV3+ alone | 0.929 | 0.878 | 0.659 | 0.938 | 0.761 | 0.732 | 0.624 | 0.604 | 0.269 | 0.187 | **0.658** | 0.648 | 0.038 |
| CPB + DeepLabV3+: Liang framework, our re-impl. | 0.891 | 0.855 | 0.727 | 0.385 | 0.882 | 0.587 | 0.734 | 0.725 | 0.421 | 0.398 | **0.660** | 0.596 | 0.034 |
| CPB + DeepLabV3+: Liang framework (block repl.), our re-impl. | 0.921 | 0.887 | 0.651 | 0.903 | 0.758 | 0.688 | 0.659 | 0.656 | 0.297 | 0.219 | **0.664** | 0.655 | 0.025 |
| **CPB + DeepLabV3+: framework + arbiter (ours)** | 0.944 | 0.967 | 0.910 | 0.874 | 0.947 | 0.863 | 0.869 | 0.888 | 0.872 | 0.519 | **0.865** | 0.864 | 0.013 |
| CPB + DeepLabV3+: oracle arbiter (ceiling) | 0.959 | 0.975 | 0.936 | 0.952 | 0.968 | 0.913 | 0.951 | 0.930 | 0.917 | 0.915 | **0.942** | 0.953 | 0.000 |
| PSPNet alone | 0.905 | 0.939 | 0.649 | 0.913 | 0.757 | 0.771 | 0.604 | 0.575 | 0.331 | 0.188 | **0.663** | 0.668 | 0.038 |
| CPB + PSPNet: Liang framework, our re-impl. | 0.890 | 0.888 | 0.716 | 0.384 | 0.877 | 0.652 | 0.732 | 0.728 | 0.433 | 0.401 | **0.670** | 0.610 | 0.034 |
| CPB + PSPNet: Liang framework (block repl.), our re-impl. | 0.919 | 0.947 | 0.661 | 0.902 | 0.764 | 0.766 | 0.652 | 0.636 | 0.359 | 0.222 | **0.683** | 0.685 | 0.025 |
| **CPB + PSPNet: framework + arbiter (ours)** | 0.945 | 0.970 | 0.913 | 0.869 | 0.951 | 0.842 | 0.924 | 0.899 | 0.873 | 0.875 | **0.906** | 0.920 | 0.000 |
| CPB + PSPNet: oracle arbiter (ceiling) | 0.960 | 0.976 | 0.936 | 0.950 | 0.969 | 0.912 | 0.953 | 0.941 | 0.921 | 0.915 | **0.944** | 0.954 | 0.000 |
| CPB + DeepLabV3+: minimal learned fusion (logistic on CPB vote, CPB score, guidance prob.) | 0.923 | 0.946 | 0.845 | 0.614 | 0.893 | 0.736 | 0.856 | 0.874 | 0.723 | 0.612 | **0.802** | 0.769 | 0.011 |
| SemanticBGS rules on CPB + DeepLabV3+ | 0.944 | 0.971 | 0.924 | 0.806 | 0.932 | 0.881 | 0.928 | 0.904 | 0.834 | 0.876 | **0.900** | 0.903 | 0.002 |
| SemanticBGS rules on SuBSENSE + DeepLabV3+ | 0.944 | 0.974 | 0.908 | 0.824 | 0.941 | 0.775 | 0.945 | 0.899 | 0.835 | 0.916 | **0.896** | 0.923 | 0.001 |
| SemanticBGS rules on CPB + PSPNet | 0.944 | 0.963 | 0.913 | 0.740 | 0.936 | 0.885 | 0.923 | 0.900 | 0.824 | 0.877 | **0.890** | 0.890 | 0.003 |
| SemanticBGS rules on SuBSENSE + PSPNet | 0.940 | 0.973 | 0.911 | 0.781 | 0.940 | 0.762 | 0.940 | 0.902 | 0.814 | 0.918 | **0.888** | 0.916 | 0.001 |
| SuBSENSE (BGSLibrary) | 0.923 | 0.868 | 0.897 | 0.602 | 0.923 | 0.634 | 0.919 | 0.877 | 0.731 | 0.860 | **0.823** | 0.829 | 0.007 |
| PAWCS (BGSLibrary) | 0.929 | 0.844 | 0.909 | 0.678 | 0.903 | 0.870 | 0.967 | 0.905 | 0.595 | 0.838 | **0.844** | 0.864 | 0.001 |
| LOBSTER (BGSLibrary) | 0.910 | 0.840 | 0.905 | 0.415 | 0.885 | 0.475 | 0.861 | 0.858 | 0.630 | 0.838 | **0.762** | 0.752 | 0.011 |
| ViBe (BGSLibrary) | 0.821 | 0.671 | 0.898 | 0.376 | 0.706 | 0.398 | 0.859 | 0.849 | 0.629 | 0.763 | **0.697** | 0.673 | 0.010 |
| GMM (MOG2) (BGSLibrary) | 0.475 | 0.161 | 0.573 | 0.420 | 0.185 | 0.580 | 0.466 | 0.629 | 0.191 | 0.656 | **0.434** | 0.430 | 0.001 |
| KDE (BGSLibrary) | 0.827 | 0.765 | 0.776 | 0.406 | 0.862 | 0.373 | 0.874 | 0.791 | 0.552 | 0.612 | **0.684** | 0.644 | 0.025 |
| PBAS (BGSLibrary) | 0.894 | 0.635 | 0.847 | 0.383 | 0.629 | 0.513 | 0.901 | 0.887 | 0.695 | 0.810 | **0.719** | 0.723 | 0.010 |
| ZBS (CVPR 2023), paper parameters | 0.951 | 0.861 | 0.878 | 0.918 | 0.745 | 0.946 | 0.968 | 0.920 | 0.904 | 0.956 | **0.905** | 0.917 | 0.005 |
| ZBS, CDnet-script group 1 (sensitivity) | 0.872 | 0.972 | 0.874 | 0.926 | 0.934 | 0.934 | 0.972 | 0.921 | 0.948 | 0.968 | **0.932** | 0.946 | 0.002 |
| ZBS, CDnet-script group 2 (sensitivity) | 0.872 | 0.972 | 0.875 | 0.922 | 0.932 | 0.935 | 0.970 | 0.921 | 0.946 | 0.965 | **0.931** | 0.945 | 0.002 |
| ZBS, CDnet-script group 3 (sensitivity) | 0.702 | 0.653 | 0.651 | 0.319 | 0.444 | 0.605 | 0.434 | 0.582 | 0.345 | 0.223 | **0.496** | 0.430 | 0.110 |

*As reported (other protocols; reference only), per category SI CA OC IL MB BS CL RA SN SU, overall:* BSUV-Net 2.0 (CDnet-trained, supervised) 0.92 0.68 0.96 0.88 0.81 0.77 0.93 0.94 0.84 0.79 (0.85); Cuevas 0.88 0.84 0.78 0.65 0.93 0.66 0.93 0.87 0.78 0.72 (0.81); Haines 0.89 0.89 0.92 0.85 0.84 0.68 0.83 0.89 0.17 0.86 (0.78); Maddalena 0.95 0.86 0.95 0.21 0.91 0.40 0.97 0.90 0.81 0.88 (0.78).
