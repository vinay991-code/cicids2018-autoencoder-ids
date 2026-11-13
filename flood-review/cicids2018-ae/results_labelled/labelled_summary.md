# F1 at a labelled threshold (Phase 15, reporting only)

Test attack share 0.1193; flag-everything F1 = 0.213. Mean over 20 random splits per seed, then over seeds [42, 1, 2]; 95% t-interval of the mean (n = 3, wide). The threshold maximises F1 on a labelled 20% and is applied to the other 80%.

| Model | F1 (rows) | F1 (blocks) | Recall (blocks) | Precision (blocks) | False-positive rate (blocks) |
|---|---|---|---|---|---|
| unsup_mixed (ordinary loss) | 0.229 [0.189, 0.268] | 0.196 [0.171, 0.221] | 0.728 | 0.117 | 0.706 |
| normal_only (benign-only, ordinary loss) | 0.713 [0.576, 0.849] | 0.644 [0.388, 0.901] | 0.697 | 0.613 | 0.057 |
| trimmed 10% | 0.233 [0.198, 0.268] | 0.206 [0.146, 0.266] | 0.812 | 0.120 | 0.750 |
| trimmed 20% | 0.222 [0.201, 0.243] | 0.205 [0.165, 0.245] | 0.844 | 0.117 | 0.817 |
| trimmed 30% | 0.227 [0.213, 0.240] | 0.206 [0.201, 0.212] | 0.810 | 0.119 | 0.770 |
| DAE-LR mixed, lambda=0.01 | 0.229 [0.212, 0.247] | 0.206 [0.190, 0.222] | 0.842 | 0.121 | 0.786 |
| DAE-LR mixed, lambda=0.1 | 0.235 [0.214, 0.255] | 0.214 [0.160, 0.268] | 0.839 | 0.125 | 0.750 |
| DAE-LR mixed, lambda=1 | 0.228 [0.204, 0.251] | 0.193 [0.166, 0.220] | 0.736 | 0.115 | 0.714 |
| DAE-LR mixed, lambda=10 | 0.226 [0.197, 0.255] | 0.194 [0.170, 0.218] | 0.722 | 0.116 | 0.695 |
| DAE-LR benign-only, lambda=0.01 | 0.692 [0.626, 0.759] | 0.631 [0.454, 0.809] | 0.688 | 0.601 | 0.060 |
| DAE-LR benign-only, lambda=0.1 | 0.630 [0.527, 0.733] | 0.587 [0.479, 0.694] | 0.694 | 0.525 | 0.086 |
| DAE-LR benign-only, lambda=1 | 0.623 [0.595, 0.652] | 0.565 [0.484, 0.646] | 0.720 | 0.483 | 0.107 |
| DAE-LR benign-only, lambda=10 | 0.564 [0.466, 0.662] | 0.507 [0.416, 0.598] | 0.703 | 0.418 | 0.146 |
