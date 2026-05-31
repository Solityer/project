# Proof Size

proof_size_bytes counts the proof transport object: commitments, quotient commitments, opening proofs, claimed values, public metadata, public output, and proof block tags. It excludes SRS files, local caches, raw witness, checkpoints, datasets, logs, and run manifest attachments.

| dataset | backend | proof_size_bytes | proof_hash | public_metadata_hash |
| --- | --- | ---: | --- | --- |
| cora | cpu | 91021 | 13912944497163105896 | 5658230561186663313 |
| citeseer | cpu | 91029 | 678763788880008611 | 17276847189185938677 |
| pubmed | cpu | 91026 | 4260261533857525994 | 15716557680981392993 |
| ppi | cpu | 21626 | 18143550974431219896 | 3740831123812889960 |
| ogbn-arxiv | cpu | 21627 | 7299358753669717243 | 17124364883838270829 |
| cora | gpu | 91021 | 17607712357163588490 | 5658230561186663313 |
| citeseer | gpu | 91029 | 12715189713134693736 | 17276847189185938677 |
| pubmed | gpu | 91026 | 1177562362488685203 | 15716557680981392993 |
| ppi | gpu | 21626 | 10243209961997116574 | 3740831123812889960 |
| ogbn-arxiv | gpu | 21627 | 6781976314683706683 | 17124364883838270829 |
