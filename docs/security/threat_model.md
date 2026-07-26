# zkGAT Threat Model

`threat_model_id = zkgat_secure_profile_v1`

## Parties

- Prover: runs zkGAT inference and produces the proof.
- Verifier: checks public metadata, public inputs, public logits, commitments, openings, quotient relations, and transcript challenges.
- Setup/Ceremony: prepares the trusted or MPC SRS and destroys the toxic setup secret after key generation.
- Dataset/Public Input Provider: supplies public graph structure, public node features, dataset configuration, task configuration, and quantization configuration.

## Public Information

The public view contains graph structure, node features, dataset config, task config, model architecture, quantization config, static table policy, SRS manifest, `srs_id`, `srs_type`, degree bound, final logits, proof size, and runtime metrics.

## Private Information

Private witness material includes raw model parameters, model-parameter openings, intermediate activations, attention scores, softmax weights, lookup witness, route witness, PSQ state, folding vectors, private polynomial raw random-point values, blinding randomness, and the KZG setup toxic secret.

## Attacker Capabilities

The attacker can inspect the proof object, public metadata, public inputs, public logits, commitments, and masked evaluations. The attacker can attempt to tamper with the proof, metadata, commitment points, opening values, transcript challenges, schema order, output logits, SRS identifiers, edge order, route queries, lookup queries, PSQ state, and output average relation.

The attacker cannot read prover memory private witness, cannot read blinding randomness, cannot access a destroyed setup secret, and cannot break the assumed pairing, discrete-log, and KZG binding assumptions.

## Security Goals

- Correctness: honest prover outputs verify.
- Soundness/binding: accepted proofs bind commitments and public logits to the zkGAT relation.
- Witness hiding: private model and trace values appear only through commitments, masked evaluations, zero-knowledge binding checks, or quotient relations.
- Output binding: public logits are bound by metadata, transcript, commitment, and opening checks.
- Metadata binding: public metadata is absorbed before dynamic-commitment challenges.
- SRS binding: `srs_id`, `srs_type`, degree bound, curve id, field id, and manifest hash are fixed before proving and verification.

## Non-Goals

This main experiment does not hide graph structure or node features. It does not publish only labels; logits are public. It does not prove training. It does not claim resistance to side channels, timing leakage, memory-access leakage, GPU-kernel side channels, or a compromised prover host. It does not claim this repository implements a full trusted ceremony; it loads and binds a trusted/MPC SRS manifest. It does not claim baselines use the same SRS, opening, metadata, or proof schema.

## Trust Assumptions

KZG security depends on the setup toxic secret not being available to prover or verifier after setup. Prover, verifier, and benchmark code load SRS files and manifest data. SRS hash, domain size, degree bound, curve id, field modulus id, and `srs_type` mismatch must hard fail. Fiat-Shamir transcript order is fixed and public metadata is absorbed before dynamic commitments produce challenges.

## Leakage Policy

`Public`, `PublicOutput`, and `PrivateWitness` visibility classes must match `ProofSchema`. Private witness objects can appear only as commitments, masked evaluations, zero-knowledge binding subproof inputs, or quotient relation inputs. `private_plain_value_count` is computed from the verifier's typed payload audit and must be zero. Proof objects must not contain raw model parameters, raw intermediate activations, raw attention weights, raw lookup witness, raw route witness, raw PSQ state, raw folding vectors, or blinding randomness.

## Tamper Model

Verifier negative tests cover final-logit/public-output tamper, metadata tamper, `srs_id` tamper, `srs_type` tamper, privacy policy tamper, input privacy tamper, output mode tamper, opening mode tamper, binding mode tamper, proof task tamper, added unknown field, commitment order tamper, dynamic commitment point tamper, quotient commitment tamper, domain opening value tamper, external masked value tamper, and transcript challenge tamper. The same gate category covers edge order, route query, lookup query, PSQ state, output average relation, hidden profile mismatch, and `K_out` mismatch through metadata, schema, quotient, and external-evaluation checks.

## Consistency Requirements

README, security docs, configs, `PublicMetadata`, run manifests, exporter rows, and `results/paper_secure/` must agree on `public_graph_public_features`, `public_logits`, `point_grouped_batch_opening`, `zk_subproof_for_crpc`, `node_classification_full_gat`, and `zkgat_secure_profile_v1`.
