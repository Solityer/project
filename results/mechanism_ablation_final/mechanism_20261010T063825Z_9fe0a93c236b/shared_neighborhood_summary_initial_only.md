# Shared Neighborhood Proof Reuse — initial-verification-only real-data CUDA results

Session: mechanism_20261010T063825Z_9fe0a93c236b; status: complete; backend: CUDA; accepted initial proof receipts: 66/66.
Measurements: three paired repetitions per configuration. CPU receipt/hash audit passed.
Full independent re-verification was explicitly cancelled by the user after 13/66 cells; this report does not claim a full re-verification archive or publication-integrity PASS.
Shared Neighborhood compares the shared PSQ/neighborhood proof with separate denominator and weighted-aggregate proofs.
Layer/head source: formal hidden layer 0, attention head 0, width 8. Feature Folding D<8 uses the first D coordinates of that same real width-8 hidden feature as a controlled mechanism width.
Commitment and Prove are disjoint native proof partitions; Verify is the recorded independent-verifier measurement; Proof Size is the serialized mechanism proof file length.
Per-pair effects are computed first: Saving (%) = (Control - Original) / Control × 100; Improvement Factor = Control / Original. Means and sample SDs summarize the three values; no confidence interval or significance claim is made.

| Dataset | N | E | D | G | PPI components | Metric | Original mean | Original sample SD | Control mean | Control sample SD | Paired O−C mean | Paired O−C sample SD | Saving % mean | Saving % sample SD | Control/Original mean | Control/Original sample SD | Repetitions |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cora | 2708 | 13264 | 8 | 1 | 0 | Commitment Time (ms) | 206.702699 | 7.120460 | 353.634640 | 24.946647 | -146.931942 | 21.817175 | 41.410275 | 3.336515 | 1.710607 | 0.100709 | 3/3 |
| cora | 2708 | 13264 | 8 | 1 | 0 | Prove Time (ms) | 345.534078 | 10.972564 | 677.117608 | 50.813144 | -331.583530 | 40.008304 | 48.860268 | 2.215929 | 1.957891 | 0.085334 | 3/3 |
| cora | 2708 | 13264 | 8 | 1 | 0 | Verify Time (ms) | 44.467823 | 1.055529 | 68.511897 | 0.781149 | -24.044074 | 1.024862 | 35.093915 | 1.401223 | 1.541166 | 0.033278 | 3/3 |
| cora | 2708 | 13264 | 8 | 1 | 0 | Proof Size (Bytes) | 6429.00 | 0.00 | 10512.00 | 0.00 | -4083.00 | 0.00 | 38.841324 | 0.000000 | 1.635091 | 0.000000 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Commitment Time (ms) | 212.143170 | 19.517307 | 349.323467 | 20.621233 | -137.180297 | 10.696460 | 39.317027 | 3.004326 | 1.650571 | 0.080686 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Prove Time (ms) | 351.772467 | 8.066595 | 653.836772 | 23.809918 | -302.064305 | 15.749700 | 46.181228 | 0.721516 | 1.858311 | 0.025001 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Verify Time (ms) | 44.449377 | 0.156974 | 67.252298 | 3.114517 | -22.802920 | 3.046451 | 33.812658 | 3.048005 | 1.512947 | 0.067872 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Proof Size (Bytes) | 6433.00 | 0.00 | 10516.00 | 0.00 | -4083.00 | 0.00 | 38.826550 | 0.000000 | 1.634696 | 0.000000 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Commitment Time (ms) | 7102.556514 | 402.214694 | 12725.005279 | 842.770615 | -5622.448765 | 441.443197 | 44.159988 | 0.565220 | 1.790953 | 0.018136 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Prove Time (ms) | 45863.157337 | 1714.352951 | 86174.995114 | 3064.169239 | -40311.837777 | 1413.986678 | 46.780454 | 0.362589 | 1.879067 | 0.012814 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Verify Time (ms) | 1439.559956 | 71.144898 | 2883.187509 | 175.350953 | -1443.627553 | 121.731069 | 50.029139 | 1.690104 | 2.002696 | 0.067845 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Proof Size (Bytes) | 6428.00 | 0.00 | 10511.00 | 0.00 | -4083.00 | 0.00 | 38.845020 | 0.000000 | 1.635190 | 0.000000 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Commitment Time (ms) | 7209.160186 | 414.030242 | 12787.075050 | 766.681156 | -5577.914864 | 352.939877 | 43.615525 | 0.170283 | 1.773549 | 0.005352 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Prove Time (ms) | 46051.176317 | 1091.990995 | 87017.482740 | 2819.556230 | -40966.306422 | 1745.086203 | 47.067963 | 0.504654 | 1.889330 | 0.018042 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Verify Time (ms) | 1440.269591 | 25.401856 | 2832.875936 | 179.566569 | -1392.606345 | 166.596748 | 49.047690 | 2.737668 | 1.966488 | 0.108012 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Proof Size (Bytes) | 6428.00 | 0.00 | 10511.00 | 0.00 | -4083.00 | 0.00 | 38.845020 | 0.000000 | 1.635190 | 0.000000 | 3/3 |
