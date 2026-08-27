#![allow(dead_code, unused_imports)]

// This is a test-only mirror of the formal module graph.  Keeping the source
// modules private here lets the tests exercise internal validation helpers
// without widening the production binary's witness or registry API.
mod algebra {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/formal/algebra.rs"
    ));
}
mod compiler {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/formal/compiler.rs"
    ));
}
#[cfg(feature = "gpu")]
mod gpu {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/formal/gpu.rs"));
}
mod hyrax {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/formal/hyrax.rs"));
}
mod prover_backend {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/formal/prover_backend.rs"
    ));
}
mod registry {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/formal/registry.rs"
    ));

    mod moved_tests {
        use super::{
            canonical_dynamic_family, canonical_public_auxiliary_family,
            canonical_registered_parameter_family, Expr,
        };
        use crate::algebra::Scalar;

        #[test]
        fn public_table_lookup_uses_little_endian_bit_columns() {
            let expression = Expr::PublicTableLookup {
                values: vec![10, 20, 30, 40]
                    .into_iter()
                    .map(Scalar::from_i64)
                    .collect(),
                bit_columns: vec![0, 1],
            };
            assert_eq!(
                expression.eval(&[Scalar::one(), Scalar::zero()]).unwrap(),
                Scalar::from_i64(20)
            );
            assert_eq!(
                expression.eval(&[Scalar::zero(), Scalar::one()]).unwrap(),
                Scalar::from_i64(30)
            );
        }

        #[test]
        fn canonical_family_grammar_separates_registered_and_dynamic_tensors() {
            assert!(canonical_registered_parameter_family(
                "hidden.layer_0.head_0.Wcol_0"
            ));
            assert!(canonical_registered_parameter_family("output.head_0.a_src"));
            assert!(canonical_registered_parameter_family("output.head_0.bias"));

            // Adapter-produced intermediates share the same namespace but
            // must never be treated as registered model parameters.
            for family_id in [
                "hidden.layer_0.head_0.projection",
                "hidden.layer_0.head_0.E_src",
                "hidden.layer_0.head_0.Pcol_0",
                "output.head_0.head_output",
            ] {
                assert!(!canonical_registered_parameter_family(family_id));
                assert!(canonical_dynamic_family(family_id));
            }
            assert!(canonical_public_auxiliary_family(
                "output.head_0.alignment_d"
            ));
            assert!(!canonical_dynamic_family(
                "hidden.layer_0.head_0.unknown_alias"
            ));
            assert!(!canonical_registered_parameter_family(
                "hidden.layer_0.head_0.weight_0"
            ));
        }
    }
}
mod sumcheck {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/formal/sumcheck.rs"
    ));

    #[cfg(test)]
    mod moved_tests {
        use super::build_columns;
        use crate::algebra::Scalar;
        use crate::registry::{
            ColumnSource, Expr, OperationColumn, OperationFamilySpec, RelationRegistry,
            TensorFamilySpec, ViewIndex, Visibility, WitnessPackage,
        };
        use std::collections::{BTreeMap, BTreeSet};

        // The production registry has a much larger canonical plan.  This
        // deliberately uses only the private-column loader so the test stays
        // a unit check of the physical-padding invariant.
        #[test]
        fn private_physical_padding_is_zero_and_not_witness_index_zero() {
            let registry = RelationRegistry {
                schema: "test".into(),
                tensor_families: vec![TensorFamilySpec {
                    family_id: "hidden.layer_0.head_0.z".into(),
                    logical_len: 1,
                    padded_len: 2,
                    logical_shape: vec![1],
                    mle_order: vec!["row_major_lsb_first".into()],
                    instance_order: vec!["canonical_program_order".into()],
                    visibility: Visibility::DynamicPrivate,
                }],
                public_tensors: BTreeMap::new(),
                operation_families: vec![],
                mandatory_operation_kinds: vec![],
                registry_id: String::new(),
                canonical_schema: serde_json::Value::Null,
            };
            let witness = WitnessPackage {
                registry_id: String::new(),
                canonical_tensor_digest: String::new(),
                tensors: BTreeMap::from([(
                    "hidden.layer_0.head_0.z".into(),
                    vec![Scalar::from_i64(37)],
                )]),
                witness_id: String::new(),
            };
            let operation = OperationFamilySpec {
                family_id: "hidden.layer_0.head_0.test".into(),
                operation_kind: "test".into(),
                rows: 4,
                real_rows: 1,
                degree_bound: 2,
                columns: vec![OperationColumn {
                    column_id: "z".into(),
                    source: ColumnSource::TensorView {
                        family_id: "hidden.layer_0.head_0.z".into(),
                        canonical_index_by_row: ViewIndex::PaddedAffine {
                            offset: 0,
                            stride: 1,
                            logical_rows: 1,
                            padding_index: 0,
                        },
                    },
                }],
                residuals: vec![Expr::Const(Scalar::zero())],
                requires_line8: false,
            };
            let (columns, padding) = build_columns(&registry, &operation, &witness).unwrap();
            assert_eq!(
                columns[0],
                vec![
                    Scalar::from_i64(37),
                    Scalar::zero(),
                    Scalar::zero(),
                    Scalar::zero(),
                ]
            );
            assert_eq!(padding[0], vec![Scalar::zero(); 4]);
            assert!(columns[0][operation.real_rows..]
                .iter()
                .all(|value| *value == Scalar::zero()));
        }
    }
}
mod transcript {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/formal/transcript.rs"
    ));
}
mod protocol {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/formal/protocol.rs"
    ));

    mod moved_tests {
        use super::*;
        use crate::hyrax::{commit, CommitmentParameters};

        #[test]
        fn canonical_model_family_parser_rejects_aliases() {
            let parsed = parse_model_family_id("hidden.layer_0.head_1.Wcol_3").unwrap();
            assert_eq!(parsed.kind, "hidden");
            assert_eq!(parsed.layer, Some(0));
            assert_eq!(parsed.head, 1);
            assert_eq!(parsed.role, ModelRole::Weight(3));
            assert!(parse_model_family_id("hidden.layer_0.head_1.weight_3").is_err());
            assert!(parse_model_family_id("public.logits").is_err());
        }

        #[test]
        fn scalar_integer_bounds_cover_signed_field_encoding() {
            assert!(scalar_abs_le(Scalar::from_i64(7), 7));
            assert!(scalar_abs_le(Scalar::from_i64(-7), 7));
            assert!(!scalar_abs_le(Scalar::from_i64(-8), 7));
        }

        #[test]
        fn session_rerandomization_rejects_zero_delta_and_same_commitment() {
            let pp = CommitmentParameters::new([7u8; 32], 4).unwrap();
            let (registry, _) = commit(&pp, vec![Scalar::one()], Scalar::one()).unwrap();
            let mut transcript = Transcript::new(PROOF_FORMAT);
            assert!(prove_session_rerandomization(
                &pp,
                "hidden.layer_0.head_0.Wcol_0",
                &registry,
                &registry,
                Scalar::zero(),
                &mut transcript
            )
            .is_err());
        }

        #[test]
        fn registration_publish_supports_separate_public_and_private_parents() {
            let transaction_id = fresh_registration_transaction_id().unwrap();
            let token = transaction_id
                .strip_prefix("sha256:")
                .expect("fresh transaction ID has canonical prefix");
            let root = std::env::temp_dir().join(format!(
                "zkdl-registration-publish-{}-{token}",
                std::process::id()
            ));
            let public_path = root.join("results/model_registry/model.bin");
            let private_path = root.join("artifacts/private_setup/model.secret");

            let mut registry = ModelRegistry {
                format: "test-registration".into(),
                pp_id: "sha256:test-pp".into(),
                registry_id: "sha256:test-registry".into(),
                model_arch_id: "sha256:test-arch".into(),
                model_param_id: "sha256:test-param".into(),
                checkpoint_manifest_id: "sha256:test-checkpoint".into(),
                model_shape_id: "sha256:test-shape".into(),
                quant_cfg_id: "sha256:test-quant".into(),
                static_tables_id: "sha256:test-tables".into(),
                static_table_rows_id: "sha256:test-table-rows".into(),
                certified_bounds_id: "sha256:test-bounds".into(),
                registration_transaction_id: transaction_id.clone(),
                registration_binding_digest: String::new(),
                commitment_set_id: "sha256:test-commitment-set".into(),
                commitment_points_id: "sha256:test-commitment-points".into(),
                commitments: BTreeMap::new(),
                registry_object_id: "test-registry-object".into(),
            };
            let mut secret = ModelOpeningSecret {
                registry_object_id: registry.registry_object_id.clone(),
                registration_transaction_id: transaction_id,
                registration_binding_digest: String::new(),
                blinders: BTreeMap::from([(
                    "hidden.layer_0.head_0.Wcol_0".into(),
                    Scalar::from_i64(17),
                )]),
            };
            registry.registration_binding_digest =
                compute_registration_binding_digest(&registry, &secret).unwrap();
            secret.registration_binding_digest = registry.registration_binding_digest.clone();

            write_model_registration_atomic(&public_path, &registry, &private_path, &secret)
                .unwrap();
            assert!(public_path.is_file());
            assert!(private_path.is_file());
            assert_ne!(public_path.parent(), private_path.parent());
            let loaded_public: ModelRegistry = read_bin(&public_path).unwrap();
            let loaded_secret: ModelOpeningSecret = read_bin(&private_path).unwrap();
            assert_eq!(
                loaded_public.registration_transaction_id,
                loaded_secret.registration_transaction_id
            );
            assert_eq!(
                loaded_public.registration_binding_digest,
                loaded_secret.registration_binding_digest
            );
            assert_eq!(
                loaded_secret.registry_object_id,
                loaded_public.registry_object_id
            );
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                assert_eq!(
                    std::fs::metadata(&private_path)
                        .unwrap()
                        .permissions()
                        .mode()
                        & 0o777,
                    0o600
                );
            }

            // A pair assembled from different transactions is rejected before
            // either destination is materialized.
            let mut mismatched_secret = secret;
            mismatched_secret.registration_transaction_id =
                "sha256:2222222222222222222222222222222222222222222222222222222222222222".into();
            let bad_public = root.join("results/model_registry/mismatch.bin");
            let bad_private = root.join("artifacts/private_setup/mismatch.secret");
            assert!(write_model_registration_atomic(
                &bad_public,
                &registry,
                &bad_private,
                &mismatched_secret
            )
            .is_err());
            assert!(!bad_public.exists());
            assert!(!bad_private.exists());
            std::fs::remove_dir_all(root).unwrap();
        }
    }
}
