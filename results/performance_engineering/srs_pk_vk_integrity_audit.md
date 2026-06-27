# SRS / PK / VK Integrity Audit

Status: passed

| field | value |
| --- | --- |
| srs_type | `trusted_srs` |
| srs_id | `zkgat_secure_main_srs_v1` |
| srs_degree_bound | `67108863` |
| srs_manifest_path | `artifacts/srs/zkgat_secure_main_srs_v1/srs_manifest.json` |
| srs_manifest_hash | `f5ea7dff25203f129796a2665aa02b2ba33bebb651447c5d5cccb9363a318871` |
| setup_transcript_hash | `5ea2bdfdfd1a155942659dd6334e0de7f9cb2923098f87b14bce2bb81c356f1c` |
| pk_coef_hash | `2338189489bd50feb56e9c99b5c46e705a25aa8d211be0c80929e4ae62dccf2f` |
| vk_hash | `5af8f41e59c03b80adf53de1fb603c69e24aaa9abcd50ecfea20d69227eb5b10` |
| curve_id | `bls12_381` |
| field_id | `bls12_381_fr` |
| domain_cfg_hash | `4e738e3d738ee2cfa5aa52695860f8b14bb1490984e48dce96aaf3c716d30769` |
| degree_bound | `67108863` |
| trusted_setup_boundary | `manifest-bound trusted_srs; runtime tau is not present in manifest or code path` |
| runtime_tau_available | `False` |
| prover_runtime_reads_tau | `False` |
| verifier_runtime_reads_witness | `False` |

## Domain Sizes

| domain | size | lagrange_hash |
| --- | ---: | --- |
| C | 128 | `174622e021198208c717ea272690da74e68c7493c756faa6cc6937f4f7f2fd50` |
| C_16 | 16 | `1c3d24d61bc67c9f52b3b2a13c7c833f3837f1414e2670b663c976cb213af1fe` |
| C_64 | 64 | `b4e936f08e22b34cbbae938e00b5b80d3746c8f1c35a1000a8bc4ed9b903553b` |
| C_8 | 8 | `32b5ff85a0ca8a0d8a4fdeea604cc55296ac80a2930c88a33386829813ed54ca` |
| FH | 33554432 | `0f12a55e91c772a891aa7028439054171a9c65b247153fa47bf8dede4d150a0e` |
| FH_16777216 | 16777216 | `da862d09f3e27a3cdcb98609e79f1a27f2e3b8de182624814342e40d714c8a89` |
| FH_4194304 | 4194304 | `95868981a8f4c50d23e74c7780d422e26aa754f0ff545f4c31c395bda5b56d01` |
| N | 262144 | `d8ca5a2ac0ab4b9adfda495222dffd3a6629822a55e03ad08cce4ec3c180470b` |
| N_32768 | 32768 | `5432027f36cc915c835712f5ab313947c7ba14ef0964d904a24fd7e693942026` |
| N_4096 | 4096 | `58fceb15623b0f34e4c45bf4e1ab52fc76d735b979761922e0121896ce8cd117` |
| N_65536 | 65536 | `c1f7441b7cbb8d121197dcdc22ea61f1d2f98010a00e5ad653aeeaf1a9cb8c5b` |
| cat | 128 | `174622e021198208c717ea272690da74e68c7493c756faa6cc6937f4f7f2fd50` |
| cat_16 | 16 | `1c3d24d61bc67c9f52b3b2a13c7c833f3837f1414e2670b663c976cb213af1fe` |
| d_h | 16 | `1c3d24d61bc67c9f52b3b2a13c7c833f3837f1414e2670b663c976cb213af1fe` |
| edge | 8388608 | `95173b8f50f4c2292b5d28cc0a5aebc56aee201ac8965b14a64712dbdd1d7132` |
| edge_1048576 | 1048576 | `8dc97e933c43cee42d3b946ca6759f4853b008bd41af04dfd1f8f168b402f751` |
| edge_131072 | 131072 | `977e07286a840909043f8f362976c4f7391c79458b163b51d4494e1c4a30428e` |
| edge_262144 | 262144 | `d8ca5a2ac0ab4b9adfda495222dffd3a6629822a55e03ad08cce4ec3c180470b` |
| in | 4096 | `58fceb15623b0f34e4c45bf4e1ab52fc76d735b979761922e0121896ce8cd117` |
| in_2048 | 2048 | `64f7c20bf2e61c1cc9cdd68b85c5b4c7b84f66ed01d7721aca356174b23c2c81` |
| in_256 | 256 | `778b5d1739f0814194186304f269c1fe8bbf30a3f2e529f11921c806c6d59839` |
| in_512 | 512 | `1306d8fde12e228bb50af0d8810041f02aa706405b5b121e1fda9b237c1d694c` |
| in_64 | 64 | `b4e936f08e22b34cbbae938e00b5b80d3746c8f1c35a1000a8bc4ed9b903553b` |

Material files are recorded by path, size, and manifest hash in the JSON audit; large SRS binaries are not dumped.
