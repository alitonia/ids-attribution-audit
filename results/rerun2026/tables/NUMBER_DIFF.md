# NUMBER DIFF — old paper → rerun2026 generation

| quantity | old | new |
|---|---|---|
| median damage (pp) | 0.0 | 0.027 |
| worst damage (pp / cell) | 69.1 | 69.10 / ('nf', 'label_flip', 0.01, 1) |
| ciciot worst (pp) | 0.23 | 2.15 |
| lf range (seed-mean) | 0.94–0.99 | 0.957–0.990 |
| fp range (seed-mean) | 0.84–0.98 | 0.835–0.975 |
| trigger CIC | 0.32–0.37 | 0.302–0.365 |
| trigger NF/CICIoT | 0.84–0.95 | 0.841–0.946 |
| dF1 median (pp) | +0.2 (gated) | +0.20 (pure TracInCP) |
| dF1 rho<=5 range (pp) | -3.6..+3.2 | -22.2..2.3 |
| collapse (10%) | -32.6, flag 77.6% | 40.0 at ('cic', 'label_flip', 0.1, 1), flag 82.9% |
| runs | 108 (3 seeds) | 180 (5 seeds) |

two_sided_trigger: {'one_sided_mean_precision': 0.010188145512647908, 'two_sided_mean_precision': 0.0038074369451556547, 'lo_tail_mean_precision': 1.5503875968992248e-05, 'hi_tail_mean_precision': 0.017295972555380682}
heldout_fpr: {'mean': 0.009699408033893442, 'p95': 0.010427059803272059, 'target': 0.01}
nf_f1_max_shift: 1.48
df1 rho<=5 cells < -5pp: [-22.2, -11.7, -6.3, -4.4]
