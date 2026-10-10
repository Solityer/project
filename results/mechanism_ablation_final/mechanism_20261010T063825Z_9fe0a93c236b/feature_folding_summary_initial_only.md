# Sparse Node-to-Edge Feature Verification — initial-verification-only real-data CUDA results

Session: mechanism_20261010T063825Z_9fe0a93c236b; status: complete; backend: CUDA; accepted initial proof receipts: 66/66.
Measurements: three paired repetitions per configuration. CPU receipt/hash audit passed.
Full independent re-verification was explicitly cancelled by the user after 13/66 cells; this report does not claim a full re-verification archive or publication-integrity PASS.
Feature Folding D<8 is a controlled-width experiment on the first D coordinates of a real 8-wide hidden head.
Layer/head source: formal hidden layer 0, attention head 0, width 8. Feature Folding D<8 uses the first D coordinates of that same real width-8 hidden feature as a controlled mechanism width.
Commitment and Prove are disjoint native proof partitions; Verify is the recorded independent-verifier measurement; Proof Size is the serialized mechanism proof file length.
Per-pair effects are computed first: Saving (%) = (Control - Original) / Control × 100; Improvement Factor = Control / Original. Means and sample SDs summarize the three values; no confidence interval or significance claim is made.

| Dataset | N | E | D | G | PPI components | Metric | Original mean | Original sample SD | Control mean | Control sample SD | Paired O−C mean | Paired O−C sample SD | Saving % mean | Saving % sample SD | Control/Original mean | Control/Original sample SD | Repetitions |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cora | 2708 | 13264 | 1 | 1 | 0 | Commitment Time (ms) | 132.196016 | 9.003068 | 129.553191 | 6.699432 | 2.642825 | 2.304019 | -1.982220 | 1.676955 | 0.980740 | 0.016142 | 3/3 |
| cora | 2708 | 13264 | 1 | 1 | 0 | Prove Time (ms) | 214.143930 | 8.569871 | 220.837092 | 12.584712 | -6.693162 | 5.052354 | 2.964141 | 2.075519 | 1.030864 | 0.022254 | 3/3 |
| cora | 2708 | 13264 | 1 | 1 | 0 | Verify Time (ms) | 35.071085 | 0.553868 | 35.768942 | 0.924423 | -0.697857 | 0.642892 | 1.927220 | 1.731698 | 1.019864 | 0.018141 | 3/3 |
| cora | 2708 | 13264 | 1 | 1 | 0 | Proof Size (Bytes) | 3848.00 | 0.00 | 3743.00 | 0.00 | 105.00 | 0.00 | -2.805236 | 0.000000 | 0.972713 | 0.000000 | 3/3 |
| cora | 2708 | 13264 | 2 | 1 | 0 | Commitment Time (ms) | 131.103915 | 5.701119 | 238.891992 | 15.519220 | -107.788078 | 9.871084 | 45.067103 | 1.257460 | 1.821032 | 0.041216 | 3/3 |
| cora | 2708 | 13264 | 2 | 1 | 0 | Prove Time (ms) | 219.425529 | 4.594433 | 407.190561 | 22.725257 | -187.765032 | 18.739912 | 46.035397 | 2.122985 | 1.854953 | 0.071949 | 3/3 |
| cora | 2708 | 13264 | 2 | 1 | 0 | Verify Time (ms) | 36.183942 | 0.117266 | 49.843894 | 0.592779 | -13.659953 | 0.674149 | 27.397477 | 1.023571 | 1.377547 | 0.019580 | 3/3 |
| cora | 2708 | 13264 | 2 | 1 | 0 | Proof Size (Bytes) | 3982.00 | 0.00 | 7105.00 | 0.00 | -3123.00 | 0.00 | 43.954961 | 0.000000 | 1.784279 | 0.000000 | 3/3 |
| cora | 2708 | 13264 | 4 | 1 | 0 | Commitment Time (ms) | 131.693751 | 6.181249 | 450.025080 | 24.400005 | -318.331330 | 18.439424 | 70.727015 | 0.400626 | 3.416549 | 0.047084 | 3/3 |
| cora | 2708 | 13264 | 4 | 1 | 0 | Prove Time (ms) | 219.192538 | 9.187472 | 781.056467 | 29.611887 | -561.863929 | 20.430255 | 71.939295 | 0.119834 | 3.563746 | 0.015255 | 3/3 |
| cora | 2708 | 13264 | 4 | 1 | 0 | Verify Time (ms) | 36.443253 | 0.235660 | 77.345754 | 2.032159 | -40.902500 | 1.796818 | 52.866492 | 0.923996 | 2.122181 | 0.041941 | 3/3 |
| cora | 2708 | 13264 | 4 | 1 | 0 | Proof Size (Bytes) | 4250.00 | 0.00 | 13829.00 | 0.00 | -9579.00 | 0.00 | 69.267481 | 0.000000 | 3.253882 | 0.000000 | 3/3 |
| cora | 2708 | 13264 | 8 | 1 | 0 | Commitment Time (ms) | 144.341810 | 12.020545 | 896.298752 | 94.413616 | -751.956942 | 95.838469 | 83.779615 | 2.124320 | 6.238243 | 0.840091 | 3/3 |
| cora | 2708 | 13264 | 8 | 1 | 0 | Prove Time (ms) | 225.572148 | 5.226329 | 1578.782958 | 141.472482 | -1353.210809 | 138.421237 | 85.649809 | 1.070840 | 6.995653 | 0.544863 | 3/3 |
| cora | 2708 | 13264 | 8 | 1 | 0 | Verify Time (ms) | 37.300290 | 0.212180 | 135.910515 | 2.916121 | -98.610224 | 2.936429 | 72.546599 | 0.624225 | 3.643782 | 0.082232 | 3/3 |
| cora | 2708 | 13264 | 8 | 1 | 0 | Proof Size (Bytes) | 4786.00 | 0.00 | 27277.00 | 0.00 | -22491.00 | 0.00 | 82.454082 | 0.000000 | 5.699331 | 0.000000 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Commitment Time (ms) | 143.777858 | 8.727506 | 939.062666 | 105.133799 | -795.284807 | 100.698738 | 84.592176 | 1.488114 | 6.531188 | 0.637403 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Prove Time (ms) | 230.110673 | 8.586204 | 1639.147056 | 153.798069 | -1409.036382 | 148.606379 | 85.895367 | 1.132014 | 7.119578 | 0.555609 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Verify Time (ms) | 35.661619 | 1.396813 | 131.528486 | 9.996152 | -95.866866 | 8.925349 | 72.821972 | 1.421730 | 3.686114 | 0.191388 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Proof Size (Bytes) | 4790.00 | 0.00 | 27281.00 | 0.00 | -22491.00 | 0.00 | 82.441993 | 0.000000 | 5.695407 | 0.000000 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Commitment Time (ms) | 4841.685374 | 283.546066 | 29139.920117 | 283.265661 | -24298.234743 | 472.855325 | 83.381154 | 1.045684 | 6.033759 | 0.393231 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Prove Time (ms) | 24803.083967 | 1616.233341 | 169974.396391 | 1270.812674 | -145171.312425 | 2811.463833 | 85.402984 | 1.045324 | 6.875222 | 0.513341 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Verify Time (ms) | 732.656534 | 50.927608 | 5554.053955 | 145.496373 | -4821.397421 | 101.560590 | 86.817036 | 0.622216 | 7.597069 | 0.366231 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Proof Size (Bytes) | 4785.00 | 0.00 | 27276.00 | 0.00 | -22491.00 | 0.00 | 82.457105 | 0.000000 | 5.700313 | 0.000000 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Commitment Time (ms) | 4810.942343 | 292.279827 | 28847.553516 | 398.054226 | -24036.611173 | 330.047281 | 83.326213 | 0.906150 | 6.009286 | 0.327189 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Prove Time (ms) | 24924.818230 | 1392.639262 | 169608.465302 | 3804.602370 | -144683.647071 | 3038.083970 | 85.307805 | 0.648493 | 6.815417 | 0.308651 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Verify Time (ms) | 750.784711 | 49.952939 | 5284.680328 | 197.820629 | -4533.895617 | 211.996520 | 85.776355 | 1.165816 | 7.060846 | 0.554592 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Proof Size (Bytes) | 4785.00 | 0.00 | 27276.00 | 0.00 | -22491.00 | 0.00 | 82.457105 | 0.000000 | 5.700313 | 0.000000 | 3/3 |
