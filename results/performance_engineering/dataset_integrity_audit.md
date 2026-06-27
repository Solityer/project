# Dataset Integrity Audit

Status: passed for local file integrity; OGBN has a separate formal LReLU table-domain conflict recorded below.

| dataset | full | graphs | nodes | normalized_edges | feature_dim | classes | domain_cfg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| cora | true | 1 | 2708 | 13264 | 1433 | 7 | `FH=FH:4194304,edge=edge:131072,in=in:2048,d_h=d:16,cat=cat:128,C=C:16,N=N:4096` |
| citeseer | true | 1 | 3327 | 12431 | 3703 | 6 | `FH=FH:16777216,edge=edge:131072,in=in:4096,d_h=d:16,cat=cat:128,C=C:8,N=N:4096` |
| pubmed | true | 1 | 19717 | 108365 | 500 | 3 | `FH=FH:16777216,edge=edge:262144,in=in:512,d_h=d:16,cat=cat:128,C=C:8,N=N:32768` |
| ppi | true | 295 | 56944 | 850576 | 50 | 121 | `FH=FH:4194304,edge=edge:1048576,in=in:64,d_h=d:16,cat=cat:16,C=C:128,N=N:65536` |
| ogbn-arxiv | true | 1 | 169343 | 1335586 | 128 | 40 | `FH=FH:33554432,edge=edge:8388608,in=in:256,d_h=d:16,cat=cat:16,C=C:64,N=N:262144` |

## Hashes

| dataset | graph_source_hash | features_hash | labels_hash | edge_index_hash | preprocessing_config_hash | quant_config_hash | domain_cfg_hash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cora | `d0f89715305732390c0e0a220d7e9569d8c92112062880a6c934af0c0db8737a` | `5050ecc4784f04ab4ce1d2fc2c277bf354f328d3eacc0acab7458cc461965954` | `8f929c6f3a938980341b2bcfa82a96e69e8d494d0c741396fd4bd87f460385c4` | `c41cf0e8321332492dd345ba54bee94817f217c19ca90a697050700235b12999` | `fce4ed3a4f8789c899dfcdf5ffc9481981b899c827fa58eadac8510526e09618` | `7432664551971964393` | `2cfe9e460a03270d72c209c742c8e233a703a9a18c4d2188744ddecb2a878d9c` |
| citeseer | `a369656f21ea7f0c5333a5626ef5aa818b739cd77cc0068bcbf940236ea01904` | `c2773bda52e719c5c2c5f007a152ee2a356106440cdf670af7752ab75b300daa` | `bb47cb2abf66935d45a9e09ff2cde1874560d24e791838b01edd44a2076419eb` | `7a22a7fa347c89d013a87f5e475be356ad48714987afab990f0ec64978b9d0bd` | `1a55d71bc1792a17455f7d04a5ae4ddbb2467c91f1b18364230ae11e78a4f01f` | `7432664551971964393` | `31604218890a4cd8fa9641d3fa747e5b78cdfc4786492844dbb38c098758c9b8` |
| pubmed | `a26e1f9b1cdc9f10b5688c6841e42b2318e76a332bde7f619edc396b52f84fba` | `f05e1c053284b3f47726693645deef159eb4abcec06ce50344769c516e11a037` | `d48a582e64f106deb746656dc18a446f13c11b421d1f7de19301c4415a6b354d` | `1eb9632dede5e3f3b392e5849fd779554d0c556d07c9073de829145bc3d72bf1` | `742b7827e3bf258685f66a4e09002b481f7218c21458ae98a8c621cfde229a25` | `7432664551971964393` | `a2df905faa434d70475e269f8198a6c5e4f4c51c93a8cddf63c299e5cc01524a` |
| ppi | `2e0d1475056cc6ae8631b5fae69784a7c2253db82fa035dfe69e003d1d8d85c8` | `b7528351c39f33759aa25411047696dc02306d2a55eec0a8cefe4ec32ce90efa` | `656e5d2edea92dae1ee3c4b5aef179d568f6e8c756c991d53f8befaa08581543` | `b28ac7c1724769e59bb1941ae0cb3a7e9da0f8fc6d5567ec13be161f1e122507` | `3854444ea51c813a7e1637eaaf0abe38988ce4f1e90535a19984726946dcf309` | `6437678090238948388` | `57d40f511d31fd4623c073d4a895551c9c46b1be8bbe34d7b2a5c85882a92847` |
| ogbn-arxiv | `4c9ccdb22fcf9447f3175b5dd9b0df623878819a60003b86bb81f7e23b2ae86b` | `e2e4dda733eff8130a8e3ff0167ce091ee145eb6f0e6eedb8e04fef36100e40b` | `7d16a4e399d6f89d2204735ca01e8e4667efae291ab154b56c2bd9110c47767c` | `b847b26c612b78a52fd785590bc4fbb6d6fc607337cda1b6d5ba5b6524151bf5` | `5eb038d432ee367f171d012928f6d7a939d1a4cf2451e08f6fbc39357699060a` | `6437678090238948388` | `7c6a5f10ccb9c1ca4ad3a3e3279625db690662105463ad193962129bf6c366fc` |

## PPI Multi-Graph Metadata

- graph_count: `295`
- batch_graphs: `295`
- node_ptr_hash: `b8c715ff62744f2f0859d3f310236e660278c72433cafd80a1849814cd7ed969`
- node_ptr_entries: `296`
- all_graphs_selected: `True`

## OGBN Large-Graph Metadata

- normalized_edge_count: `1335586`
- feature_table_size: `21675904`
- fixed_edge_domain_size: `8388608`
- lrelu_required_min_raw: `-44524211`
- lrelu_required_max_raw: `108122576`
- lrelu_required_span_raw: `152646788`
- next_power_of_two_for_required_lrelu_table: `268435456`
- diagnostic_path: `results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_formal_raw_20260627.csv`
- current_raw_diagnostic_path: `results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_current_raw_20260627T095943Z.csv`
- rescale_probe_diagnostic_path: `results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_rescale_products_20260627T095943Z.csv`
- rescale_probe_hidden_span_raw: `440240`
- rescale_probe_output_span_raw: `255080`

The OGBN dataset files and preprocessing hashes are stable. Under the current raw fixed-point proof relation, the required contiguous LReLU raw span exceeds the fixed edge-domain/SRS capacity. A diagnostic-only product-rescale probe fits within the existing table capacity, but retaining that route would require changing verifier equations and proof relation, so it is excluded by the main invariants.
