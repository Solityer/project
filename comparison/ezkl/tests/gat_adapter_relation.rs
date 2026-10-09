use ezkl::gat_adapter::schema::HiddenLayerSpec;
use ezkl::gat_adapter::{
    evaluate_signed_division, scale_exponent_to_multiplier, CanonicalGat, CanonicalGraph,
    CanonicalModel, ExactTable, GatOp, HeadWeights, ModelBinding, RoundingMode,
};

#[test]
fn signed_round_division_covers_sign_zero_tie_and_toward_zero() {
    let positive = evaluate_signed_division(5, 2, RoundingMode::NearestTiesAway).unwrap();
    assert_eq!(positive.quotient, 3);
    assert_eq!(positive.quotient_floor, 2);
    assert_eq!(positive.remainder, 0);
    assert_eq!(positive.delta, 3);
    assert!(positive.rounded_up);
    positive.validate(RoundingMode::NearestTiesAway).unwrap();

    let negative = evaluate_signed_division(-5, 2, RoundingMode::NearestTiesAway).unwrap();
    assert_eq!(negative.quotient, -3);
    assert!(negative.sigma);
    assert_eq!(negative.remainder, 0);
    negative.validate(RoundingMode::NearestTiesAway).unwrap();

    let zero = evaluate_signed_division(0, 7, RoundingMode::NearestTiesAway).unwrap();
    assert!(zero.zero);
    assert!(!zero.sigma);
    assert_eq!(zero.quotient, 0);
    assert_eq!(zero.remainder, 0);

    let odd_boundary = evaluate_signed_division(1, 3, RoundingMode::NearestTiesAway).unwrap();
    assert_eq!(odd_boundary.quotient, 0);
    assert_eq!(odd_boundary.remainder, 5);
    assert_eq!(odd_boundary.delta, 0);
    odd_boundary
        .validate(RoundingMode::NearestTiesAway)
        .unwrap();

    let toward_zero = evaluate_signed_division(-5, 2, RoundingMode::TowardZero).unwrap();
    assert_eq!(toward_zero.quotient, -2);
    assert!(!toward_zero.rounded_up);
    toward_zero.validate(RoundingMode::TowardZero).unwrap();
}

#[test]
fn signed_round_division_rejects_invalid_divisor_and_tampered_remainder() {
    assert!(evaluate_signed_division(1, 0, RoundingMode::NearestTiesAway).is_err());
    assert!(evaluate_signed_division(1, -2, RoundingMode::NearestTiesAway).is_err());

    let mut tampered = evaluate_signed_division(5, 3, RoundingMode::NearestTiesAway).unwrap();
    tampered.remainder = 3;
    assert!(tampered.validate(RoundingMode::NearestTiesAway).is_err());
}

#[test]
fn scales_and_static_table_rows_are_exact_and_authenticated() {
    assert_eq!(scale_exponent_to_multiplier(0).unwrap(), 1);
    assert_eq!(scale_exponent_to_multiplier(7).unwrap(), 128);
    assert!(scale_exponent_to_multiplier(-1).is_err());

    let mut table = ExactTable {
        table_id: "test".into(),
        input_scale: 0,
        output_scale: 0,
        function_id: "test".into(),
        domain_id: "test-domain".into(),
        generator_id: "test-generator".into(),
        rows: vec![(0, 0), (1, 1), (2, 4)],
        input_min: 0,
        input_max: 2,
        digest: String::new(),
    };
    table.refresh_digest().unwrap();
    table.validate(false, false).unwrap();

    table.rows.swap(0, 1);
    table.refresh_digest().unwrap();
    assert!(table.validate(false, false).is_err());
}

#[test]
fn canonical_graph_requires_unique_directed_edges_and_one_loop_per_node() {
    let mut valid = CanonicalGraph {
        n_total: 4,
        n: 4,
        e: 7,
        g_batch: 2,
        node_ids: vec![0, 1, 2, 3],
        src: vec![1, 0, 0, 1, 2, 2, 3],
        dst: vec![0, 0, 1, 1, 2, 3, 3],
        node_ptr: vec![0, 2, 4],
        edge_ptr: vec![0, 4, 7],
        topology_id: "test-topology".into(),
        topology_digest: String::new(),
    };
    valid.refresh_digest().unwrap();
    valid.validate().unwrap();
    assert_eq!(
        valid.destination_groups().unwrap(),
        vec![(0, 0, 2), (1, 2, 4), (2, 4, 5), (3, 5, 7)],
    );
    assert_eq!(
        valid.comparison_canonical_digest().unwrap(),
        "9b6ebdc703be1d54d32ca1d4a3bfc6cc8921eb7dc6acbf5893b69ca302e5875a",
    );

    let mut duplicate = valid.clone();
    duplicate.src[1] = 1;
    duplicate.refresh_digest().unwrap();
    assert!(duplicate
        .validate()
        .unwrap_err()
        .contains("duplicate directed edge"));

    let mut missing_loop = CanonicalGraph {
        n_total: 4,
        n: 4,
        e: 6,
        g_batch: 2,
        node_ids: vec![0, 1, 2, 3],
        src: vec![1, 0, 1, 2, 2, 3],
        dst: vec![0, 1, 1, 2, 3, 3],
        node_ptr: vec![0, 2, 4],
        edge_ptr: vec![0, 3, 6],
        topology_id: "test-topology".into(),
        topology_digest: String::new(),
    };
    missing_loop.refresh_digest().unwrap();
    assert!(missing_loop
        .validate()
        .unwrap_err()
        .contains("missing its required per-node self-loop"));
}

#[test]
fn redaction_keeps_parameter_shape_but_removes_private_values() {
    let model = CanonicalModel {
        hidden_layers: vec![HiddenLayerSpec {
            d_in: 1,
            heads: vec![HeadWeights {
                projection: vec![17],
                a_src: vec![-3],
                a_dst: vec![5],
                bias: vec![],
            }],
            d_head: 1,
        }],
        output_heads: vec![HeadWeights {
            projection: vec![11],
            a_src: vec![7],
            a_dst: vec![-2],
            bias: vec![13],
        }],
        classes: 1,
        model_arch_id: "test".into(),
    };
    let before_len = model.flattened_params().len();
    let mut op = GatOp {
        spec: CanonicalGat {
            model,
            model_binding: ModelBinding {
                hash: "0x01".into(),
                hash_domain: "ezkl-gat-model-v1".into(),
                canonical_model_digest: "00".repeat(32),
            },
            ..CanonicalGat::default()
        },
        rounding: RoundingMode::NearestTiesAway,
    };
    op.redact_private_params();
    assert_eq!(op.spec.model.flattened_params().len(), before_len);
    assert!(op
        .spec
        .model
        .flattened_params()
        .iter()
        .all(|value| *value == 0));
    assert!(op.spec.private_params_redacted);
    assert_eq!(op.spec.model_binding.hash_domain, "ezkl-gat-model-v1");
}

#[test]
fn actual_gat_layout_reuses_tables_with_identical_known_and_unknown_shape() {
    use ezkl::circuit::region::RegionSettings;
    use ezkl::graph::{GraphCircuit, GraphWitness};
    use ezkl::tensor::{Tensor, ValTensor};
    use halo2_proofs::circuit::Value;
    use halo2curves::bn256::Fr;
    let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gat_lookup_reuse");
    let circuit = GraphCircuit::load(fixture.join("compiled.circuit")).unwrap();
    let witness = GraphWitness::from_path(fixture.join("witness.json")).unwrap();
    let mut known_inputs = witness
        .get_input_tensor()
        .into_iter()
        .map(|input| input.map(|value| Value::known(value)).into())
        .collect::<Vec<ValTensor<Fr>>>();
    let mut known_params = witness
        .get_param_tensor()
        .into_iter()
        .map(|input| input.map(|value| Value::known(value)).into())
        .collect::<Vec<ValTensor<Fr>>>();
    // The old serialized fixture predates the canonical GAT scale chain and
    // leaves RunArgs::param_scale at 1. The adapter validates against the
    // authenticated model scale (2 for this fixture), as the formal runner
    // does, so make the runtime context agree with that declared scale.
    let mut args = circuit.settings().run_args.clone();
    args.param_scale = circuit
        .model()
        .graph
        .nodes
        .values()
        .find_map(|node| match node {
            ezkl::graph::NodeType::Node(node) => match &node.opkind {
                ezkl::graph::SupportedOp::Gat(op) => Some(op.spec.scales.s_w_hidden[0][0]),
                _ => None,
            },
            ezkl::graph::NodeType::SubGraph { .. } => None,
        })
        .expect("fixture includes the canonical GAT parameter scale");
    for (i, input) in known_inputs.iter_mut().enumerate() {
        input
            .reshape(&circuit.model().graph.input_shapes().unwrap()[i])
            .unwrap();
        input.set_scale(circuit.settings().model_input_scales[i]);
    }
    for (input, shape) in known_params
        .iter_mut()
        .zip(circuit.model().graph.param_shapes().unwrap())
    {
        input.reshape(&shape).unwrap();
        input.set_scale(args.param_scale);
    }
    let unknown_inputs = known_inputs
        .iter()
        .map(|input| {
            let mut unknown: ValTensor<Fr> = Tensor::new(
                Some(&vec![Value::<Fr>::unknown(); input.len()]),
                input.dims(),
            )
            .unwrap()
            .into();
            unknown.set_scale(input.scale());
            unknown
        })
        .collect::<Vec<_>>();
    let unknown_params = known_params
        .iter()
        .map(|input| {
            let mut unknown: ValTensor<Fr> = Tensor::new(
                Some(&vec![Value::<Fr>::unknown(); input.len()]),
                input.dims(),
            )
            .unwrap()
            .into();
            unknown.set_scale(input.scale());
            unknown
        })
        .collect::<Vec<_>>();
    let _ = env_logger::try_init();
    let settings = RegionSettings::all_false(args.decomp_base, args.decomp_legs);
    let known = circuit
        .model()
        .dummy_layout_with_runtime_params(&args, &known_inputs, &known_params, settings.clone())
        .unwrap();
    let unknown = circuit
        .model()
        .dummy_layout_with_runtime_params(&args, &unknown_inputs, &unknown_params, settings)
        .unwrap();
    assert_eq!(known.linear_coord, unknown.linear_coord);
    assert_eq!(known.dynamic_lookup_params, unknown.dynamic_lookup_params);
    assert_eq!(known.shuffle_params, unknown.shuffle_params);
    assert_eq!(known.einsum_params, unknown.einsum_params);
    assert_eq!(known.total_const_size, unknown.total_const_size);
    assert!(known.dynamic_lookup_params.num_dynamic_lookups < 64);
}

#[test]
fn mv_lookup_membership_borrowed_encoding_matches_owned_lookup() {
    use halo2curves::bn256::Fr;
    use halo2curves::ff::PrimeField;
    use std::collections::HashMap as FxHashMap;
    use std::time::{Duration, Instant};

    let table = (0..1024).map(Fr::from).collect::<Vec<_>>();
    let membership = table
        .iter()
        .enumerate()
        .map(|(index, value)| (value.to_repr().as_ref().to_owned(), index))
        .collect::<FxHashMap<Vec<u8>, usize>>();
    let queries = (0..1024 * 512)
        .map(|index| table[index % table.len()])
        .collect::<Vec<_>>();

    let mut owned_samples = Vec::new();
    let mut borrowed_samples = Vec::new();
    let mut owned_checksum = 0usize;
    let mut borrowed_checksum = 0usize;
    for _ in 0..3 {
        let started = Instant::now();
        let mut checksum = 0usize;
        for value in &queries {
            let representation = value.to_repr();
            checksum = checksum.wrapping_add(
                *membership
                    .get(&representation.as_ref().to_owned())
                    .expect("owned encoded membership key"),
            );
        }
        owned_samples.push(started.elapsed());
        owned_checksum = checksum;

        let started = Instant::now();
        let mut checksum = 0usize;
        for value in &queries {
            let representation = value.to_repr();
            checksum = checksum.wrapping_add(
                *membership
                    .get(representation.as_ref())
                    .expect("borrowed encoded membership key"),
            );
        }
        borrowed_samples.push(started.elapsed());
        borrowed_checksum = checksum;
    }
    assert_eq!(owned_checksum, borrowed_checksum);
    owned_samples.sort_unstable();
    borrowed_samples.sort_unstable();
    eprintln!(
        "mv_lookup_membership_microbenchmark queries={} key_copy_bytes={} owned_median_ms={:.3} borrowed_median_ms={:.3}",
        queries.len(),
        queries.len() * std::mem::size_of::<Fr>(),
        owned_samples[1].as_secs_f64() * 1000.0,
        borrowed_samples[1].as_secs_f64() * 1000.0,
    );
    assert!(owned_samples[1] > Duration::ZERO);
}

#[test]
fn mv_lookup_table_map_build_avoids_duplicate_key_allocations() {
    use halo2curves::bn256::Fr;
    use halo2curves::ff::PrimeField;
    use std::collections::HashMap as FxHashMap;
    use std::time::Instant;

    let values = (0..(1024 * 512))
        .map(|row| Fr::from((row % 1024) as u64))
        .collect::<Vec<_>>();
    let mut owned_samples = Vec::new();
    let mut bounded_samples = Vec::new();
    let mut owned_result = FxHashMap::default();
    let mut bounded_result = FxHashMap::default();
    for _ in 0..3 {
        let started = Instant::now();
        owned_result = values
            .iter()
            .enumerate()
            .map(|(row, value)| (value.to_repr().as_ref().to_owned(), row))
            .collect::<FxHashMap<_, _>>();
        owned_samples.push(started.elapsed());
        let started = Instant::now();
        let mut index = FxHashMap::default();
        for (row, value) in values.iter().enumerate() {
            let representation = value.to_repr();
            if let Some(previous) = index.get_mut(representation.as_ref()) {
                *previous = row;
            } else {
                index.insert(representation.as_ref().to_owned(), row);
            }
        }
        bounded_result = index;
        bounded_samples.push(started.elapsed());
    }
    assert_eq!(owned_result, bounded_result);
    owned_samples.sort_unstable();
    bounded_samples.sort_unstable();
    let key_bytes = std::mem::size_of::<Fr>();
    eprintln!(
        "mv_lookup_table_map_microbenchmark rows={} unique={} owned_key_allocation_bytes={} bounded_key_allocation_bytes={} owned_final_capacity={} bounded_final_capacity={} owned_median_ms={:.3} bounded_median_ms={:.3}",
        values.len(), owned_result.len(), values.len() * key_bytes, bounded_result.len() * key_bytes,
        owned_result.capacity(), bounded_result.capacity(), owned_samples[1].as_secs_f64() * 1000.0,
        bounded_samples[1].as_secs_f64() * 1000.0,
    );
}

#[test]
fn native_mv_lookup_rejects_out_of_range_queries_before_commitment() {
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            create_proof, keygen_pk, keygen_vk, Advice, Circuit, Column, ConstraintSystem, Error,
            Fixed,
        },
        poly::{
            kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::ProverSHPLONK,
            },
            Rotation,
        },
        transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
    };
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use rand::rngs::OsRng;

    #[derive(Clone, Copy)]
    struct InvalidLookupCircuit;
    #[derive(Clone)]
    struct Config {
        query: Column<Advice>,
        table: Column<Fixed>,
    }
    impl Circuit<Fr> for InvalidLookupCircuit {
        type Config = Config;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();
        fn without_witnesses(&self) -> Self {
            *self
        }
        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let query = meta.advice_column();
            let table = meta.fixed_column();
            meta.lookup_any("tiny invalid lookup", |meta| {
                vec![(
                    meta.query_advice(query, Rotation::cur()),
                    meta.query_fixed(table, Rotation::cur()),
                )]
            });
            Config { query, table }
        }
        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "invalid lookup row",
                |mut region| {
                    region.assign_advice(
                        || "query outside table",
                        config.query,
                        0,
                        || Value::known(Fr::from(7)),
                    )?;
                    region.assign_fixed(
                        || "only table row",
                        config.table,
                        0,
                        || Value::known(Fr::from(1)),
                    )?;
                    Ok(())
                },
            )?;
            Ok(())
        }
    }

    let circuit = InvalidLookupCircuit;
    let mut rng = OsRng;
    let params = ParamsKZG::<Bn256>::setup(3, &mut rng);
    let vk = keygen_vk(&params, &circuit).expect("tiny invalid lookup VK");
    let pk = keygen_pk(&params, vk, &circuit).expect("tiny invalid lookup PK");
    let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
    let result = create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[circuit],
        &[&[]],
        &mut rng,
        &mut transcript,
    );
    match result {
        Err(Error::NativeStorage(error)) => {
            let message = error.to_string();
            assert!(message.contains("argument 0"));
            assert!(message.contains("row 0"));
            assert!(message.contains("active rows"));
        }
        other => panic!("out-of-range lookup must fail before producing a proof: {other:?}"),
    }
}

const TINY_GAT_TEST_LOGROWS: u32 = 13;

fn known_gat_fixture(
    test_logrows: u32,
) -> (ezkl::graph::GraphCircuit, ezkl::graph::GraphWitness, u32) {
    use ezkl::graph::{GraphCircuit, GraphWitness};

    let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gat_lookup_reuse");
    let incomplete = GraphCircuit::load(fixture.join("compiled.circuit")).unwrap();
    let compiled_logrows = incomplete.settings().run_args.logrows;
    let witness = GraphWitness::from_path(fixture.join("witness.json")).unwrap();
    let mut test_args = incomplete.settings().run_args.clone();
    // The serialized fixture uses k=17. For this test-only one-node relation,
    // k=13 is the smallest domain that contains its 5,455 main rows plus
    // reserved rows; the largest required range (0, 16,383) is represented
    // with three lookup-table columns at that domain, below the 12-column
    // native limit. This keeps the same complete constraints while avoiding
    // an unnecessary 16x domain for the bounded test.
    test_args.logrows = test_logrows;
    let mut known = GraphCircuit::new(incomplete.model().clone(), &test_args).expect(
        "fresh fixture settings should configure every range used by the current GAT layout",
    );
    known.load_graph_witness(&witness).unwrap();
    (known, witness, compiled_logrows)
}

fn report_gat_fixture_shape(
    circuit: &ezkl::graph::GraphCircuit,
    witness: &ezkl::graph::GraphWitness,
    compiled_logrows: u32,
) {
    use ezkl::graph::{NodeType, SupportedOp};

    let settings = circuit.settings();
    let spec = circuit
        .model()
        .graph
        .nodes
        .values()
        .find_map(|node| match node {
            NodeType::Node(node) => match &node.opkind {
                SupportedOp::Gat(op) => Some(&op.spec),
                _ => None,
            },
            NodeType::SubGraph { .. } => None,
        })
        .expect("fixture includes the canonical GAT operation");
    let hidden = spec
        .model
        .hidden_layers
        .iter()
        .map(|layer| (layer.heads.len(), layer.d_in, layer.d_head))
        .collect::<Vec<_>>();
    let private_params = witness.params.iter().map(Vec::len).sum::<usize>();
    eprintln!(
        "EZKL_TEST_FIXTURE compiled_k={} test_k={} domain={} graph_nodes={} assignments={} rows={} n_total={} n={} e={} feature_dim={} feature_values={} hidden_heads_din_dhead={:?} output_heads={} classes={} witness_params={} required_ranges={} required_lookups={} dyn_rows={} dyn_max_input={} dyn_lookups={} shuffle_rows={} einsum_rows={} public_instances={:?}",
        compiled_logrows,
        settings.run_args.logrows,
        1usize.checked_shl(settings.run_args.logrows).unwrap_or(0),
        circuit.model().graph.nodes.len(),
        settings.total_assignments,
        settings.num_rows,
        spec.graph.n_total,
        spec.graph.n,
        spec.graph.e,
        spec.feature_dim,
        spec.features.len(),
        hidden,
        spec.model.output_heads.len(),
        spec.model.classes,
        private_params,
        settings.required_range_checks.len(),
        settings.required_lookups.len(),
        settings.dynamic_lookup_params.total_dynamic_col_size,
        settings.dynamic_lookup_params.max_dynamic_input_len,
        settings.dynamic_lookup_params.num_dynamic_lookups,
        settings.shuffle_params.total_shuffle_col_size,
        settings.einsum_params.total_einsum_col_size,
        settings.model_instance_shapes,
    );
}

#[test]
fn native_gat_selector_census_uses_a_complete_witnessless_circuit() {
    use halo2_proofs::plonk::{self, Circuit};
    use halo2curves::bn256::G1Affine;
    use std::time::Instant;

    let _ = env_logger::try_init();
    let (known, witness, compiled_logrows) = known_gat_fixture(TINY_GAT_TEST_LOGROWS);
    let k = known.settings().run_args.logrows;
    report_gat_fixture_shape(&known, &witness, compiled_logrows);
    assert!(known.settings().required_range_checks.contains(&(0, 255)));

    let started = Instant::now();
    eprintln!("EZKL_TEST_PHASE START stage=known_native_census k={k}");
    let known_census = plonk::census_native_shape::<G1Affine, _>(k, &known)
        .expect("native census should synthesize the complete known GAT witness");
    eprintln!(
        "EZKL_TEST_PHASE DONE stage=known_native_census elapsed_ms={} selectors={} copy_calls={} selector_activations={} fixed_assignments={}",
        started.elapsed().as_millis(),
        known_census.selector_fixed_column_count,
        known_census.permutation_copy_call_count,
        known_census.selector_activation_counts.iter().sum::<u64>(),
        known_census.fixed_value_assignment_counts.iter().sum::<u64>(),
    );

    let witnessless = known.without_witnesses();
    let started = Instant::now();
    eprintln!("EZKL_TEST_PHASE START stage=witnessless_native_census k={k}");
    let witnessless_census = plonk::census_native_shape::<G1Affine, _>(k, &witnessless)
        .expect("native census should synthesize the standard witnessless GAT circuit");
    eprintln!(
        "EZKL_TEST_PHASE DONE stage=witnessless_native_census elapsed_ms={} selectors={} copy_calls={} selector_activations={} fixed_assignments={}",
        started.elapsed().as_millis(),
        witnessless_census.selector_fixed_column_count,
        witnessless_census.permutation_copy_call_count,
        witnessless_census.selector_activation_counts.iter().sum::<u64>(),
        witnessless_census.fixed_value_assignment_counts.iter().sum::<u64>(),
    );
    assert_eq!(known_census, witnessless_census);
    assert!(known_census.selector_fixed_column_count > 0);
}

#[test]
fn native_gat_fixture_mockprover_is_satisfied() {
    use halo2_proofs::dev::MockProver;

    let _ = env_logger::try_init();
    let (known, witness, compiled_logrows) = known_gat_fixture(TINY_GAT_TEST_LOGROWS);
    let k = known.settings().run_args.logrows;
    report_gat_fixture_shape(&known, &witness, compiled_logrows);
    let public = known
        .prepare_public_inputs(&witness)
        .expect("tiny GAT fixture public inputs");
    let instance_elements = known.settings().total_instances().iter().sum::<usize>();
    assert_eq!(
        public.len(),
        instance_elements,
        "prepared fixture instances include model and module instances"
    );
    eprintln!("EZKL_TEST_PHASE START stage=tiny_gat_mockprover k={k} public_values={} declared_instance_values={instance_elements}", public.len());
    let prover =
        MockProver::run(k, &known, vec![public]).expect("tiny GAT fixture MockProver construction");
    match prover.verify() {
        Ok(()) => eprintln!("EZKL_TEST_PHASE DONE stage=tiny_gat_mockprover failures=0"),
        Err(failures) => panic!("tiny GAT fixture constraints fail: {failures:#?}"),
    }
}

#[test]
fn native_gat_fixture_proof_verifies_independently() {
    use ezkl::graph::GraphCircuit;
    use halo2_proofs::{
        plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, ProvingKey},
        poly::kzg::{
            commitment::KZGCommitmentScheme,
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
            strategy::SingleStrategy,
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
        SerdeFormat,
    };
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use rand::rngs::OsRng;
    use std::time::Instant;

    let _ = env_logger::try_init();
    let (known, witness, compiled_logrows) = known_gat_fixture(TINY_GAT_TEST_LOGROWS);
    let k = known.settings().run_args.logrows;
    report_gat_fixture_shape(&known, &witness, compiled_logrows);
    let mut rng = OsRng;

    let started = Instant::now();
    eprintln!("EZKL_TEST_PHASE START stage=tiny_kzg_setup k={k}");
    let params = halo2_proofs::poly::kzg::commitment::ParamsKZG::<Bn256>::setup(k, &mut rng);
    eprintln!(
        "EZKL_TEST_PHASE DONE stage=tiny_kzg_setup elapsed_ms={}",
        started.elapsed().as_millis()
    );

    let started = Instant::now();
    eprintln!("EZKL_TEST_PHASE START stage=tiny_keygen_vk");
    let vk = keygen_vk(&params, &known).expect("tiny GAT fixture VK generation");
    eprintln!(
        "EZKL_TEST_PHASE DONE stage=tiny_keygen_vk elapsed_ms={}",
        started.elapsed().as_millis()
    );

    let started = Instant::now();
    eprintln!("EZKL_TEST_PHASE START stage=tiny_keygen_pk");
    let pk = keygen_pk(&params, vk, &known).expect("tiny GAT fixture PK generation");
    eprintln!(
        "EZKL_TEST_PHASE DONE stage=tiny_keygen_pk elapsed_ms={}",
        started.elapsed().as_millis()
    );

    let started = Instant::now();
    eprintln!("EZKL_TEST_PHASE START stage=tiny_pk_native_roundtrip");
    let encoded_pk = pk.to_bytes(SerdeFormat::RawBytes);
    let pk = ProvingKey::<G1Affine>::from_bytes::<GraphCircuit>(
        &encoded_pk,
        SerdeFormat::RawBytes,
        known.settings().clone(),
    )
    .expect("tiny GAT fixture native proving-key roundtrip reconstructs permutation sigmas");
    eprintln!(
        "EZKL_TEST_PHASE DONE stage=tiny_pk_native_roundtrip elapsed_ms={} encoded_bytes={}",
        started.elapsed().as_millis(),
        encoded_pk.len(),
    );

    let public = known
        .prepare_public_inputs(&witness)
        .expect("tiny GAT fixture public inputs");
    assert_eq!(
        public.len(),
        known.settings().total_instances().iter().sum::<usize>(),
        "prepared fixture instances include model and module instances",
    );
    let instances: &[&[&[Fr]]] = &[&[&public]];
    let started = Instant::now();
    eprintln!("EZKL_TEST_PHASE START stage=tiny_native_gat_prove");
    let mut writer = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[known],
        instances,
        &mut rng,
        &mut writer,
    )
    .expect("tiny GAT fixture proof generation");
    let proof = writer.finalize();
    eprintln!(
        "EZKL_TEST_PHASE DONE stage=tiny_native_gat_prove elapsed_ms={} proof_bytes={}",
        started.elapsed().as_millis(),
        proof.len()
    );

    let started = Instant::now();
    eprintln!("EZKL_TEST_PHASE START stage=tiny_independent_verify");
    let mut reader = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
    verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<_>, _, _, SingleStrategy<_>>(
        &params,
        pk.get_vk(),
        SingleStrategy::new(&params),
        instances,
        &mut reader,
        1u64 << k,
    )
    .expect("tiny GAT fixture proof verifies independently");
    eprintln!(
        "EZKL_TEST_PHASE DONE stage=tiny_independent_verify elapsed_ms={}",
        started.elapsed().as_millis()
    );

    let mut tampered_proof = proof.clone();
    let tamper_index = tampered_proof.len() / 2;
    tampered_proof[tamper_index] ^= 1;
    let mut tampered_reader =
        Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&tampered_proof[..]);
    let tampered_result =
        verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<_>, _, _, SingleStrategy<_>>(
            &params,
            pk.get_vk(),
            SingleStrategy::new(&params),
            instances,
            &mut tampered_reader,
            1u64 << k,
        );
    assert!(
        tampered_result.is_err(),
        "independent verifier accepted a modified proof"
    );
    eprintln!("EZKL_TEST_PHASE DONE stage=tiny_tamper_rejected byte_index={tamper_index}");
}

#[test]
fn native_roundtrip_single_phase_instance_copy_proof_verifies() {
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column,
            ConstraintSystem, Error, Instance, ProvingKey,
        },
        poly::{
            commitment::Params,
            kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::{ProverSHPLONK, VerifierSHPLONK},
                strategy::SingleStrategy,
            },
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
        SerdeFormat,
    };
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use rand::rngs::OsRng;

    #[derive(Clone)]
    struct InstanceCopyCircuit {
        value: Fr,
    }

    #[derive(Clone)]
    struct Config {
        advice: Column<Advice>,
        instance: Column<Instance>,
    }

    impl Circuit<Fr> for InstanceCopyCircuit {
        type Config = Config;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self { value: Fr::from(0) }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let advice = meta.advice_column();
            let instance = meta.instance_column();
            meta.enable_equality(instance);
            Config { advice, instance }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            let cell = layouter.assign_region(
                || "native roundtrip instance copy",
                |mut region| {
                    region.assign_advice(
                        || "private value",
                        config.advice,
                        0,
                        || Value::known(self.value),
                    )
                },
            )?;
            layouter.constrain_instance(cell.cell(), config.instance, 0)
        }
    }

    let k = 4;
    let circuit = InstanceCopyCircuit { value: Fr::from(7) };
    let mut rng = OsRng;
    let params = ParamsKZG::<Bn256>::setup(k, &mut rng);
    let vk = keygen_vk(&params, &circuit).expect("tiny instance-copy VK");
    let pk = keygen_pk(&params, vk, &circuit).expect("tiny instance-copy PK");
    let encoded_pk = pk.to_bytes(SerdeFormat::RawBytes);
    let pk = ProvingKey::<G1Affine>::from_bytes::<InstanceCopyCircuit>(
        &encoded_pk,
        SerdeFormat::RawBytes,
        (),
    )
    .expect("native PK roundtrip reconstructs sparse permutation sigma");
    let public = [Fr::from(7)];
    let instances: &[&[&[Fr]]] = &[&[&public]];
    let mut writer = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[circuit],
        instances,
        &mut rng,
        &mut writer,
    )
    .expect("tiny instance-copy proof generation");
    let proof = writer.finalize();
    let mut reader = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
    verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<_>, _, _, SingleStrategy<_>>(
        &params,
        pk.get_vk(),
        SingleStrategy::new(&params),
        instances,
        &mut reader,
        params.n(),
    )
    .expect("tiny instance-copy proof verifies independently");
}

#[test]
fn native_roundtrip_identity_permutation_gate_proof_verifies() {
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column,
            ConstraintSystem, Error, ProvingKey, Selector,
        },
        poly::{
            commitment::Params,
            kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::{ProverSHPLONK, VerifierSHPLONK},
                strategy::SingleStrategy,
            },
            Rotation,
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
        SerdeFormat,
    };
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use rand::rngs::OsRng;

    #[derive(Clone)]
    struct ZeroAdviceCircuit;

    #[derive(Clone)]
    struct Config {
        advice: Column<Advice>,
        selector: Selector,
    }

    impl Circuit<Fr> for ZeroAdviceCircuit {
        type Config = Config;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let advice = meta.advice_column();
            let instance = meta.instance_column();
            meta.enable_equality(advice);
            meta.enable_equality(instance);
            let selector = meta.selector();
            meta.create_gate("zero advice", |meta| {
                vec![meta.query_selector(selector) * meta.query_advice(advice, Rotation::cur())]
            });
            Config { advice, selector }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "satisfied no-copy row",
                |mut region| {
                    config.selector.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "zero witness",
                        config.advice,
                        0,
                        || Value::known(Fr::from(0)),
                    )?;
                    Ok(())
                },
            )?;
            Ok(())
        }
    }

    let k = 4;
    let circuit = ZeroAdviceCircuit;
    let mut rng = OsRng;
    let params = ParamsKZG::<Bn256>::setup(k, &mut rng);
    let vk = keygen_vk(&params, &circuit).expect("tiny no-copy VK");
    let expected_vk_repr = vk.transcript_repr();
    let pk = keygen_pk(&params, vk, &circuit).expect("tiny no-copy PK");
    let encoded_pk = pk.to_bytes(SerdeFormat::RawBytes);
    let pk = ProvingKey::<G1Affine>::from_bytes::<ZeroAdviceCircuit>(
        &encoded_pk,
        SerdeFormat::RawBytes,
        (),
    )
    .expect("native no-copy PK roundtrip");
    assert_eq!(
        pk.get_vk().transcript_repr(),
        expected_vk_repr,
        "raw PK roundtrip preserves the VK transcript representation"
    );
    let public = [Fr::from(7)];
    let instances: &[&[&[Fr]]] = &[&[&public]];
    let mut writer = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[circuit],
        instances,
        &mut rng,
        &mut writer,
    )
    .expect("tiny no-copy proof generation");
    let proof = writer.finalize();
    let mut reader = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
    verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<_>, _, _, SingleStrategy<_>>(
        &params,
        pk.get_vk(),
        SingleStrategy::new(&params),
        instances,
        &mut reader,
        params.n(),
    )
    .expect("tiny no-copy proof verifies independently");
}

#[test]
fn native_k4_gate_permutation_matrix_reports_exact_cases() {
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            census_native_shape, create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit,
            Column, ConstraintSystem, Error, Instance, ProvingKey, Selector,
        },
        poly::{
            commitment::Params,
            kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::{ProverSHPLONK, VerifierSHPLONK},
                strategy::SingleStrategy,
            },
            Rotation,
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
        SerdeFormat,
    };
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use rand::rngs::OsRng;

    #[derive(Clone, Copy, Debug, Default)]
    struct Case {
        advice_equality: bool,
        instance_equality: bool,
        active_gate: bool,
    }

    #[derive(Clone)]
    struct MatrixCircuit {
        case: Case,
    }

    #[derive(Clone)]
    struct Config {
        advice: Column<Advice>,
        selector: Selector,
        active_gate: bool,
    }

    impl Circuit<Fr> for MatrixCircuit {
        type Config = Config;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = Case;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn params(&self) -> Self::Params {
            self.case
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            Self::configure_case(meta, Case::default())
        }

        fn configure_with_params(
            meta: &mut ConstraintSystem<Fr>,
            case: Self::Params,
        ) -> Self::Config {
            Self::configure_case(meta, case)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "k4 gate permutation control",
                |mut region| {
                    if config.active_gate {
                        config.selector.enable(&mut region, 0)?;
                    }
                    region.assign_advice(
                        || "zero witness",
                        config.advice,
                        0,
                        || Value::known(Fr::from(0)),
                    )?;
                    Ok(())
                },
            )?;
            Ok(())
        }
    }

    impl MatrixCircuit {
        fn configure_case(meta: &mut ConstraintSystem<Fr>, case: Case) -> Config {
            let advice = meta.advice_column();
            let instance = meta.instance_column();
            if case.advice_equality {
                meta.enable_equality(advice);
            }
            if case.instance_equality {
                meta.enable_equality(instance);
            }
            let selector = meta.selector();
            meta.create_gate("zero advice", |meta| {
                vec![meta.query_selector(selector) * meta.query_advice(advice, Rotation::cur())]
            });
            Config {
                advice,
                selector,
                active_gate: case.active_gate,
            }
        }
    }

    let k = 4;
    let mut setup_rng = OsRng;
    let params = ParamsKZG::<Bn256>::setup(k, &mut setup_rng);
    let public = [Fr::from(7)];
    let instances: &[&[&[Fr]]] = &[&[&public]];
    for case in [
        Case {
            advice_equality: false,
            instance_equality: false,
            active_gate: false,
        },
        Case {
            advice_equality: false,
            instance_equality: false,
            active_gate: true,
        },
        Case {
            advice_equality: true,
            instance_equality: false,
            active_gate: false,
        },
        Case {
            advice_equality: true,
            instance_equality: false,
            active_gate: true,
        },
        Case {
            advice_equality: false,
            instance_equality: true,
            active_gate: false,
        },
        Case {
            advice_equality: false,
            instance_equality: true,
            active_gate: true,
        },
        Case {
            advice_equality: true,
            instance_equality: true,
            active_gate: false,
        },
        Case {
            advice_equality: true,
            instance_equality: true,
            active_gate: true,
        },
    ] {
        let circuit = MatrixCircuit { case };
        let mut configured = ConstraintSystem::<Fr>::default();
        MatrixCircuit::configure_case(&mut configured, case);
        let degree = configured.degree();
        let permutation_columns =
            usize::from(case.advice_equality) + usize::from(case.instance_equality);
        let chunk_len = degree.saturating_sub(2);
        let sets = permutation_columns.div_ceil(chunk_len.max(1));
        let census =
            census_native_shape::<G1Affine, _>(k, &circuit).expect("bounded k4 native census");
        assert_eq!(census.permutation_copy_call_count, 0);
        assert_eq!(
            census.selector_activation_counts,
            vec![u64::from(case.active_gate)]
        );
        println!(
            "EZKL_K4_CONTROL advice_eq={} instance_eq={} active_gate={} permutation_columns=[{}] degree={} chunk_len={} sets={} gate_queries=[Advice(0,cur)] selector_rows=[{}] instance_vector=[7]",
            case.advice_equality,
            case.instance_equality,
            case.active_gate,
            [
                case.advice_equality.then_some("Advice(0,FirstPhase)"),
                case.instance_equality.then_some("Instance(0,N/A)"),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(","),
            degree,
            chunk_len,
            sets,
            if case.active_gate { "0" } else { "none" },
        );
        let vk = keygen_vk(&params, &circuit).expect("k4 control VK");
        let pk = keygen_pk(&params, vk, &circuit).expect("k4 control PK");
        let encoded_pk = pk.to_bytes(SerdeFormat::RawBytes);
        let pk = ProvingKey::<G1Affine>::from_bytes::<MatrixCircuit>(
            &encoded_pk,
            SerdeFormat::RawBytes,
            case,
        )
        .expect("k4 control PK roundtrip");
        let mut proof_rng = OsRng;
        let mut writer = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
        create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<_>, _, _, _, _>(
            &params,
            &pk,
            &[circuit],
            instances,
            &mut proof_rng,
            &mut writer,
        )
        .expect("k4 control proof generation");
        let proof = writer.finalize();
        let mut reader = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
        let result = verify_proof::<
            KZGCommitmentScheme<Bn256>,
            VerifierSHPLONK<_>,
            _,
            _,
            SingleStrategy<_>,
        >(
            &params,
            pk.get_vk(),
            SingleStrategy::new(&params),
            instances,
            &mut reader,
            params.n(),
        );
        println!(
            "EZKL_K4_CONTROL_RESULT advice_eq={} instance_eq={} active_gate={} result={}",
            case.advice_equality,
            case.instance_equality,
            case.active_gate,
            match result {
                Ok(()) => "VERIFY_OK".to_owned(),
                Err(error) => format!("VERIFY_ERR:{error:?}"),
            },
        );
    }
}

#[test]
fn halo2_commitment_timing_counts_public_and_phase_zero_only() {
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Challenge, Circuit, Column,
            ConstraintSystem, Error, FirstPhase, Instance, ProvingKey, SecondPhase, Selector,
        },
        poly::{
            commitment::Params,
            kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::{ProverSHPLONK, VerifierSHPLONK},
                strategy::SingleStrategy,
            },
            Rotation,
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
        SerdeFormat,
    };
    use halo2curves::bn256::{Bn256, Fr, G1Affine};

    #[derive(Clone)]
    struct PhaseConfig {
        first: Column<Advice>,
        second: Column<Advice>,
        instance: Column<Instance>,
        challenge: Challenge,
        selector: Selector,
    }

    #[derive(Clone)]
    struct TwoPhaseCircuit;

    impl Circuit<Fr> for TwoPhaseCircuit {
        type Config = PhaseConfig;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let first = meta.advice_column_in::<FirstPhase>(FirstPhase);
            let challenge = meta.challenge_usable_after::<FirstPhase>(FirstPhase);
            let second = meta.advice_column_in::<SecondPhase>(SecondPhase);
            let instance = meta.instance_column();
            let selector = meta.selector();
            meta.enable_equality(first);
            meta.enable_equality(instance);
            meta.create_gate("phase-zero value plus transcript challenge", |meta| {
                let q = meta.query_selector(selector);
                let first_value = meta.query_advice(first, Rotation::cur());
                let second_value = meta.query_advice(second, Rotation::cur());
                let challenge_value = meta.query_challenge(challenge);
                vec![q * (second_value - first_value - challenge_value)]
            });
            PhaseConfig {
                first,
                second,
                instance,
                challenge,
                selector,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            let challenge_value = layouter.get_challenge(config.challenge);
            let first_cell = layouter.assign_region(
                || "two-phase advice row",
                |mut region| {
                    config.selector.enable(&mut region, 0)?;
                    let first = region.assign_advice(
                        || "phase-zero advice",
                        config.first,
                        0,
                        || Value::known(Fr::from(7)),
                    )?;
                    let _second = region.assign_advice(
                        || "challenge-dependent phase-one advice",
                        config.second,
                        0,
                        || challenge_value.map(|challenge| Fr::from(7) + challenge),
                    )?;
                    Ok(first.cell())
                },
            )?;
            layouter.constrain_instance(first_cell, config.instance, 0)
        }
    }

    struct DeterministicRng(u64);

    impl rand::RngCore for DeterministicRng {
        fn next_u32(&mut self) -> u32 {
            self.next_u64() as u32
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for chunk in dest.chunks_mut(8) {
                let bytes = self.next_u64().to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
            self.fill_bytes(dest);
            Ok(())
        }
    }

    impl rand::CryptoRng for DeterministicRng {}

    let circuit = TwoPhaseCircuit;
    let mut rng = DeterministicRng(0x4f5a_2731_9912_0123);
    let params = ParamsKZG::<Bn256>::setup(5, &mut rng);
    let vk = keygen_vk(&params, &circuit).unwrap();
    let pk = keygen_pk(&params, vk, &circuit).unwrap();
    // The production native proving path reloads the serialized key, which
    // reconstructs the full permutation sigma polynomials omitted by the
    // in-memory keygen representation. Exercise that same path here.
    let encoded_pk = pk.to_bytes(SerdeFormat::RawBytes);
    let pk = ProvingKey::<G1Affine>::from_bytes::<TwoPhaseCircuit>(
        &encoded_pk,
        SerdeFormat::RawBytes,
        (),
    )
    .expect("native proving-key roundtrip reconstructs permutation sigmas");
    let public = [Fr::from(7)];
    let instances: &[&[&[Fr]]] = &[&[&public]];
    let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
    let measurement = halo2_proofs::plonk::commitment_timing::begin().unwrap();
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[circuit],
        instances,
        &mut rng,
        &mut transcript,
    )
    .unwrap();
    let measurement = measurement.finish().unwrap();
    // This EZKL SHPLONK profile absorbs public instances as transcript scalars;
    // it does not create a public-instance KZG polynomial commitment.
    assert_eq!(measurement.instance_commitments, 0);
    assert_eq!(measurement.advice_commitments, 1);
    assert_eq!(measurement.normalization_batches, 1);

    let proof = transcript.finalize();
    let mut verifier_transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
    verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<_>, _, _, _>(
        &params,
        pk.get_vk(),
        SingleStrategy::new(&params),
        instances,
        &mut verifier_transcript,
        params.n(),
    )
    .unwrap();
}
