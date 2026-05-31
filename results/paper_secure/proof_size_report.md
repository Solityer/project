# Proof Size

proof_size_bytes counts the proof transport object: commitments, quotient commitments, opening proofs, claimed values, public metadata, public output, and proof block tags. It excludes SRS files, local caches, raw witness, checkpoints, datasets, logs, and run manifest attachments.

| dataset | backend | proof_size_bytes | proof_hash | public_metadata_hash |
| --- | --- | ---: | --- | --- |
| cora | cpu | 91816 | 16939857789281977336 | 5658230561186663313 |
| citeseer | cpu | 91824 | 6355006191641796501 | 17276847189185938677 |
| pubmed | cpu | 91821 | 484979584332142264 | 15716557680981392993 |
| ppi | cpu | 21805 | 7454686822759679213 | 3740831123812889960 |
| ogbn-arxiv | cpu | 21806 | 5470238488798745217 | 17124364883838270829 |
| cora | gpu | 91816 | 14893196151688360384 | 5658230561186663313 |
| citeseer | gpu | 91824 | 4159216229063968333 | 17276847189185938677 |
| pubmed | gpu | 91821 | 9409876146424896900 | 15716557680981392993 |
| ppi | gpu | 21805 | 9334895809643805251 | 3740831123812889960 |
| ogbn-arxiv | gpu | 21806 | 9244927815482443757 | 17124364883838270829 |
