use ezkl::gat_adapter::schema::HiddenLayerSpec;
use ezkl::gat_adapter::{
    evaluate_signed_division, scale_exponent_to_multiplier, CanonicalGat, CanonicalGraph,
    CanonicalModel, ExactTable, GatOp, HeadWeights, ModelBinding, RoundingMode,
};

#[test]
fn native_selector_resource_shape_handles_virtual_selectors() {
    use halo2_proofs::plonk::{self, ConstraintSystem};
    use halo2_proofs::poly::Rotation;
    use halo2curves::bn256::{Fr, G1Affine};

    let mut cs = ConstraintSystem::<Fr>::default();
    let advice = cs.advice_column();
    let selector_a = cs.complex_selector();
    let selector_b = cs.complex_selector();

    cs.create_gate("virtual selector resource shape", |meta| {
        let left = meta.query_selector(selector_a);
        let right = meta.query_selector(selector_b);
        let value = meta.query_advice(advice, Rotation::cur());
        vec![(left + right) * value]
    });

    let (evaluation_cs, _) = cs
        .clone()
        .directly_convert_selectors_to_fixed(vec![Vec::new(); cs.num_selectors()], false);
    let evaluation_shape = plonk::native_evaluation_resource_shape::<G1Affine>(&evaluation_cs);
    assert_eq!(evaluation_shape.custom_gate_dependency_columns, vec![3]);

    let original_shplonk = plonk::native_shplonk_resource_shape::<Fr>(&cs, 16, 0, 1, 2);
    let compressed_shplonk = plonk::native_shplonk_resource_shape::<Fr>(
        &cs.with_native_selector_resource_columns(1),
        16,
        0,
        1,
        2,
    );
    assert_eq!(
        compressed_shplonk.fixed_query_polynomial_count,
        original_shplonk.fixed_query_polynomial_count + 1
    );
    assert_eq!(
        compressed_shplonk.query_polynomial_count,
        original_shplonk.query_polynomial_count + 1
    );
}

#[test]
fn region_bulk_increment_matches_elementwise_next_at_column_boundaries() {
    use ezkl::circuit::ops::region::{RegionCtx, RegionSettings};
    use halo2curves::bn256::Fr;

    for columns in [1usize, 2, 3, 7, 16] {
        for initial_row in [0usize, 1, 3, 15] {
            for steps in [0usize, 1, 2, 3, 15, 31, 1023] {
                let settings = RegionSettings::all_false(2, 16);
                let mut ordinary = RegionCtx::<Fr>::new_dummy(initial_row, columns, settings.clone());
                let mut bulk = RegionCtx::<Fr>::new_dummy(initial_row, columns, settings);
                for _ in 0..steps { ordinary.next(); }
                bulk.increment(steps);
                assert_eq!(bulk.row(), ordinary.row());
                assert_eq!(bulk.linear_coord(), ordinary.linear_coord());
            }
        }
    }
}

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

fn source_built_tiny_gat_spec() -> CanonicalGat {
    use ezkl::gat_adapter::schema::{
        self, BoundFamily, CanonicalGraph, CanonicalModel, DigestIdentity, HeadWeights,
        HiddenLayerSpec, IntegerBoundLedger, ModelBinding, ScaleChain, StaticTables,
        GAT_SCHEMA_VERSION, MODEL_HASH_DOMAIN,
    };
    use halo2curves::ff::PrimeField;

    fn head(input: usize, output: usize, with_bias: bool) -> HeadWeights {
        HeadWeights {
            projection: vec![0; input * output],
            a_src: vec![0; output],
            a_dst: vec![0; output],
            bias: if with_bias { vec![0; output] } else { Vec::new() },
        }
    }

    fn exact_table(
        table_id: &str,
        function_id: &str,
        input_min: i128,
        input_max: i128,
        input_scale: i32,
        output_scale: i32,
        output: impl Fn(i128) -> i128,
    ) -> ezkl::gat_adapter::ExactTable {
        let mut table = ezkl::gat_adapter::ExactTable {
            table_id: table_id.into(),
            input_scale,
            output_scale,
            function_id: function_id.into(),
            domain_id: format!("{}-domain", table_id),
            generator_id: format!("{}-generator", table_id),
            rows: (input_min..=input_max)
                .map(|input| (input, output(input)))
                .collect(),
            input_min,
            input_max,
            digest: String::new(),
        };
        table.refresh_digest().unwrap();
        table
    }

    fn family(layer: usize, head: usize, is_output: bool) -> BoundFamily {
        let max_aggregation_num = if layer == 0 && head == 0 && !is_output {
            3_600
        } else {
            0
        };
        BoundFamily {
            layer,
            head,
            is_output,
            max_score_abs: 600,
            max_weight_abs: 1,
            max_attention_abs: 1,
            max_bias_abs: 1,
            max_table_output_abs: 3_600,
            max_delta: 1_200,
            max_exp: 1,
            max_sum: 3,
            max_aggregation_num,
            max_dividend_abs: 7_200,
            max_aux_abs: 43_212,
            max_divisor: 6,
        }
    }

    let mut graph = CanonicalGraph {
        n_total: 5,
        n: 5,
        e: 9,
        g_batch: 2,
        node_ids: vec![0, 1, 2, 3, 4],
        src: vec![0, 1, 0, 2, 1, 0, 3, 4, 3],
        dst: vec![0, 1, 1, 2, 2, 2, 3, 4, 4],
        node_ptr: vec![0, 3, 5],
        edge_ptr: vec![0, 6, 9],
        topology_id: "tiny-two-graph-topology".into(),
        topology_digest: String::new(),
    };
    graph.refresh_digest().unwrap();

    let model = CanonicalModel {
        hidden_layers: vec![
            HiddenLayerSpec {
                d_in: 2,
                heads: vec![head(2, 2, false), head(2, 2, false)],
                d_head: 2,
            },
            HiddenLayerSpec {
                d_in: 4,
                heads: vec![head(4, 3, false)],
                d_head: 3,
            },
        ],
        output_heads: vec![head(3, 2, true), head(3, 2, true)],
        classes: 2,
        model_arch_id: "tiny-two-layer-multihead".into(),
    };
    let scales = ScaleChain {
        s_input: 1,
        s_delta: 2,
        s_exp: 1,
        s_h: vec![1, 1, 1],
        s_w_hidden: vec![vec![0, 0], vec![0]],
        s_h_prime: vec![vec![1, 1], vec![1]],
        s_a_src_hidden: vec![vec![1, 1], vec![1]],
        s_a_dst_hidden: vec![vec![1, 1], vec![1]],
        s_agg_pre: vec![1, 1],
        s_agg: vec![1, 1],
        s_cat: vec![1, 1],
        s_w_output: vec![0, 0],
        s_y_prime: vec![1, 1],
        s_a_src_output: vec![1, 1],
        s_a_dst_output: vec![1, 1],
        s_b_output: vec![2, 2],
        s_out_pre: 1,
        s_out_head: 2,
        s_out: 2,
    };
    let mut tables = StaticTables {
        lrelu: exact_table("tiny-lrelu", "lrelu", -600, 600, 2, 2, |value| value),
        exp: exact_table("tiny-exp", "exp_neg", 0, 1_200, 2, 1, |_| 1),
        elu: exact_table("tiny-elu", "elu", -3_600, 3_600, 1, 1, |value| value),
        max_relu: exact_table("tiny-max-relu", "max_relu", -1_200, 1_200, 2, 2, |value| {
            value.max(0)
        }),
        static_tables_id: "tiny-static-tables".into(),
        backend_aux_tables_id: "tiny-backend-aux".into(),
        static_tables_digest: String::new(),
    };
    tables.refresh_digest().unwrap();

    let mut families = vec![family(0, 0, false), family(0, 1, false), family(1, 0, false)];
    families.push(family(usize::MAX, 0, true));
    families.push(family(usize::MAX, 1, true));
    let mut ledger = IntegerBoundLedger {
        max_score_abs: 600,
        max_table_output_abs: 3_600,
        max_delta: 1_200,
        max_exp: 1,
        max_sum: 3,
        max_aggregation_num: 3_600,
        max_dividend_abs: 7_200,
        max_aux_abs: 43_212,
        max_divisor: 6,
        field_modulus: halo2curves::bn256::Fr::MODULUS.into(),
        ledger_id: "tiny-bounds".into(),
        ledger_digest: String::new(),
        families,
    };
    ledger.refresh_digest().unwrap();
    let topology = DigestIdentity {
        id: graph.topology_id.clone(),
        digest: graph.topology_digest.clone(),
    };
    let quant_cfg_id = "tiny-quantization".to_string();
    let quantization = DigestIdentity {
        id: quant_cfg_id.clone(),
        digest: schema::sha256_digest(&scales.canonical_bytes()).unwrap(),
    };
    let ledger_identity = DigestIdentity {
        id: ledger.ledger_id.clone(),
        digest: schema::sha256_digest(&ledger.canonical_bytes()).unwrap(),
    };
    let mut spec = CanonicalGat {
        schema_version: GAT_SCHEMA_VERSION,
        graph,
        features: vec![0; 10],
        feature_dim: 2,
        model,
        scales,
        tables,
        ledger,
        quant_cfg_id,
        ezkl_security_profile_id: "tiny-native-profile".into(),
        topology,
        quantization,
        ledger_identity,
        circuit_ref: None,
        model_binding: ModelBinding {
            hash: "1".into(),
            hash_domain: MODEL_HASH_DOMAIN.into(),
            canonical_model_digest: String::new(),
        },
        private_params_redacted: false,
    };
    spec.features[0] = 1;
    spec.model.hidden_layers[0].heads[0].projection[0] = 1;
    let field_params = spec
        .model
        .flattened_params()
        .into_iter()
        .map(ezkl::fieldutils::integer_rep_to_felt::<halo2curves::bn256::Fr>)
        .collect::<Vec<_>>();
    spec.model_binding.hash = ezkl::graph::modules::native_poseidon_hash_string(&field_params)
        .unwrap();
    spec.model_binding.canonical_model_digest = spec.canonical_model_digest().unwrap();
    spec.validate_for_compile().unwrap();
    spec
}

fn source_built_tiny_gat_fixture(
    test_logrows: u32,
) -> (ezkl::graph::GraphCircuit, ezkl::graph::GraphWitness, u32) {
    use ezkl::circuit::ops::region::RegionSettings;
    use ezkl::gat_adapter::{build_model_for_run_args, gat_settings_for_run_args, RoundingMode};
    use ezkl::graph::{GraphCircuit, Visibility};
    use ezkl::tensor::Tensor;
    use halo2curves::bn256::{Bn256, Fr};

    let spec = source_built_tiny_gat_spec();
    let mut build_args = ezkl::RunArgs::default();
    // Match the production GAT settings path: its module scales come from the
    // authenticated scale chain, not RunArgs' generic defaults.
    build_args.input_scale = spec.scales.s_input;
    build_args.param_scale = spec.scales.s_w_hidden[0][0];
    build_args.param_visibility = Visibility::Hashed {
        hash_is_public: true,
        outlets: Vec::new(),
    };
    build_args.num_inner_cols = 2;
    build_args.decomp_base = 256;
    build_args.decomp_legs = 2;
    let compiled_logrows = build_args.logrows;
    let model = build_model_for_run_args(
        spec.clone(),
        RoundingMode::NearestTiesAway,
        &build_args,
    )
    .expect("source-built fixture uses the native GAT model builder");

    let mut run_args = build_args;
    run_args.logrows = test_logrows;
    let settings = gat_settings_for_run_args(&spec, &run_args)
        .expect("source-built fixture uses current canonical GAT settings");
    let expected_input_scales = [spec.scales.s_input, 0, 0, 0, 0, 0];
    assert_eq!(run_args.input_scale, spec.scales.s_input);
    assert_eq!(run_args.param_scale, spec.scales.s_w_hidden[0][0]);
    assert_eq!(settings.run_args.input_scale, spec.scales.s_input);
    assert_eq!(settings.run_args.param_scale, spec.scales.s_w_hidden[0][0]);
    assert_eq!(settings.model_input_scales, expected_input_scales);
    assert_eq!(settings.model_output_scales, [spec.scales.s_out]);
    let input_scales = settings.model_input_scales.clone();
    let mut circuit = GraphCircuit::new_from_settings(model, settings, run_args.check_mode)
        .expect("source-built fixture creates a native GraphCircuit");

    let as_tensor = |values: &[i128], dimensions: &[usize], scale: i32| {
        let field_values = values
            .iter()
            .map(|value| ezkl::fieldutils::integer_rep_to_felt::<Fr>(*value))
            .collect::<Vec<_>>();
        let mut tensor = Tensor::new(Some(&field_values), dimensions)
            .expect("fixture tensor shape matches the canonical GAT input");
        tensor.set_scale(scale);
        tensor
    };
    let graph = &spec.graph;
    let node_ids = graph.node_ids.iter().map(|value| *value as i128).collect::<Vec<_>>();
    let src = graph.src.iter().map(|value| *value as i128).collect::<Vec<_>>();
    let dst = graph.dst.iter().map(|value| *value as i128).collect::<Vec<_>>();
    let node_ptr = graph.node_ptr.iter().map(|value| *value as i128).collect::<Vec<_>>();
    let edge_ptr = graph.edge_ptr.iter().map(|value| *value as i128).collect::<Vec<_>>();
    let mut inputs = vec![
        as_tensor(&spec.features, &[graph.n_total, spec.feature_dim], input_scales[0]),
        as_tensor(&node_ids, &[graph.n], input_scales[1]),
        as_tensor(&src, &[graph.e], input_scales[2]),
        as_tensor(&dst, &[graph.e], input_scales[3]),
        as_tensor(&node_ptr, &[graph.g_batch + 1], input_scales[4]),
        as_tensor(&edge_ptr, &[graph.g_batch + 1], input_scales[5]),
    ];
    let flat_params = spec.model.flattened_params();
    let params = as_tensor(
        &flat_params,
        &[flat_params.len()],
        run_args.param_scale,
    );
    let witness = circuit
        .forward_with_runtime_params::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<Bn256>,
        >(
            &mut inputs,
            Some(&params),
            None,
            None,
            RegionSettings::all_true(run_args.decomp_base, run_args.decomp_legs),
        )
        .expect("native full GAT forward creates the witness and H(W)");
    assert_eq!(witness.inputs.len(), 6, "GAT witness preserves all public inputs");
    assert_eq!(witness.params.len(), 1, "GAT witness preserves the private parameter closure");
    let witnessed_hash = witness
        .processed_params
        .as_ref()
        .and_then(|processed| processed.poseidon_hash.as_ref())
        .and_then(|hash| hash.first())
        .map(|value| ezkl::pfsys::canonical_fr_to_string(value))
        .expect("native parameter module produces public H(W)");
    assert_eq!(witnessed_hash, spec.model_binding.hash, "witness H(W) matches the canonical binding");
    circuit
        .load_graph_witness(&witness)
        .expect("native source-built witness loads into its GraphCircuit");
    (circuit, witness, compiled_logrows)
}

fn known_gat_fixture(
    test_logrows: u32,
) -> (ezkl::graph::GraphCircuit, ezkl::graph::GraphWitness, u32) {
    use ezkl::graph::{GraphCircuit, GraphWitness, NodeType, SupportedOp};

    let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gat_lookup_reuse");
    let incomplete = GraphCircuit::load(fixture.join("compiled.circuit")).unwrap();
    let compiled_logrows = incomplete.settings().run_args.logrows;
    let witness = GraphWitness::from_path(fixture.join("witness.json")).unwrap();
    let mut test_args = incomplete.settings().run_args.clone();
    // The serialized fixture uses k=17. Keep the same complete one-node
    // relation at k=13; its public wide bounds use the adapter-owned byte
    // range gadget rather than per-bound native lookup tables.
    test_args.logrows = test_logrows;
    let spec = incomplete
        .model()
        .graph
        .nodes
        .values()
        .find_map(|node| match node {
            NodeType::Node(node) => match &node.opkind {
                SupportedOp::Gat(op) => Some(op.spec.clone()),
                _ => None,
            },
            NodeType::SubGraph { .. } => None,
        })
        .expect("fixture includes the canonical GAT operation");
    let settings = ezkl::gat_adapter::gat_settings_for_run_args(&spec, &test_args)
        .expect("fresh fixture settings use the authenticated GAT layout and range-gadget version");
    let mut known = GraphCircuit::new_from_settings(
        incomplete.model().clone(),
        settings,
        test_args.check_mode,
    )
    .expect("fresh fixture circuit uses its complete canonical GAT settings");
    known.load_graph_witness(&witness).unwrap();
    (known, witness, compiled_logrows)
}

fn report_current_five_dataset_geometry() {
    use halo2_proofs::plonk::{Circuit, ConstraintSystem};
    use halo2curves::bn256::Fr;

    let baseline_root = std::env::current_dir().expect("canonical geometry test runs from the EZKL baseline root");
    let manifest_path = std::path::Path::new("resources/resource_manifest.json");
    for dataset in ["cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv"] {
        let loaded = ezkl::gat_adapter::load_dataset_with_manifest(
            &baseline_root,
            &manifest_path,
            dataset,
        )
        .unwrap_or_else(|error| panic!("authenticate canonical {dataset} geometry resources: {error}"));
        let parameter_count = loaded
            .spec
            .model
            .flattened_params_len()
            .unwrap_or_else(|error| panic!("count canonical {dataset} parameters: {error}"));
        let (model, settings) = ezkl::execute::build_canonical_gat_settings(&loaded)
            .unwrap_or_else(|error| panic!("build production {dataset} geometry settings: {error}"));
        // Configure only the current production constraint system. This reads
        // column/lookup geometry without synthesizing the graph, allocating
        // selector activation arrays, or creating N-domain polynomial data.
        let mut cs = ConstraintSystem::<Fr>::default();
        let circuit_config = ezkl::graph::GraphCircuit::configure_with_params(
            &mut cs,
            settings.clone(),
        );
        let native_cs = cs.clone().chunk_lookups();
        let native_lookup_arguments = native_cs.lookups().len();
        let native_query_counts = native_cs
            .lookups()
            .iter()
            .map(|lookup| lookup.input_expressions().len())
            .collect::<Vec<_>>();
        let native_query_polynomials = native_query_counts
            .iter()
            .try_fold(0usize, |total, count| total.checked_add(*count))
            .expect("canonical lookup query count fits usize");
        let native_table_groups = cs.lookups_map.len();
        let mut chunked_tables = Vec::new();
        for lookup in native_cs.lookups() {
            let table = lookup.table_expressions();
            if !chunked_tables.iter().any(|known| *known == table) {
                chunked_tables.push(table);
            }
        }
        // T counts the pre-chunk lookup-map table groups; newT counts unique
        // table-expression vectors in the final native chunked arguments.
        let native_chunked_table_groups = chunked_tables.len();
        let native_degree = native_cs.degree();
        let used_einsums = settings
            .einsum_params
            .equations
            .iter()
            .enumerate()
            .map(|(index, (equation, axes))| ((index, equation.clone()), axes.clone()))
            .collect();
        let einsum_analysis = ezkl::circuit::ops::chip::einsum::analysis::analyze_einsum_usage(
            &used_einsums,
        )
        .unwrap_or_else(|error| panic!("analyze production {dataset} Einsum allocation: {error}"));
        let main_advice_cells = (settings.total_assignments as u128) * 3;
        let main_advice_logical_bytes = main_advice_cells * 32;
        let dynamic_capacity = settings.dynamic_lookup_params.total_dynamic_col_size
            + settings.shuffle_params.total_shuffle_col_size;
        let extra_advice_cells = if settings.requires_dynamic_lookup() || settings.requires_shuffle() {
            (dynamic_capacity as u128) * 3
        } else {
            0
        };
        let extra_advice_bytes = extra_advice_cells * 32;
        let einsum_advice_cells = if settings.einsum_params.total_einsum_col_size > 0 {
            (einsum_analysis.reduction_length as u128) * 6
        } else {
            0
        };
        let domain = 1usize
            .checked_shl(settings.run_args.logrows)
            .expect("canonical geometry domain fits usize");
        let reserved_advice_rows =
            ezkl::gat_adapter::GAT_RANGE_BLINDING_FACTOR_RESERVE + 1;
        let usable_advice_rows = domain
            .checked_sub(reserved_advice_rows)
            .expect("formal k covers the authenticated GAT row reserve");
        let main_blocks = ezkl::tensor::VarTensor::advice_blocks_for_capacity(
            settings.total_assignments,
            usable_advice_rows * settings.run_args.num_inner_cols,
        );
        let main_advice_columns = main_blocks * settings.run_args.num_inner_cols * 3;
        let has_dynamic_advice = settings.requires_dynamic_lookup() || settings.requires_shuffle();
        let dynamic_blocks = if has_dynamic_advice {
            ezkl::tensor::VarTensor::advice_blocks_for_capacity(dynamic_capacity, usable_advice_rows)
        } else {
            0
        };
        let dynamic_advice_columns = dynamic_blocks * 3;
        let einsum_blocks = if settings.einsum_params.total_einsum_col_size > 0 {
            ezkl::tensor::VarTensor::advice_blocks_for_capacity(
                einsum_analysis.reduction_length,
                usable_advice_rows * settings.run_args.num_inner_cols,
            )
        } else {
            0
        };
        let einsum_advice_columns = einsum_blocks * settings.run_args.num_inner_cols * 6;
        let hashed_parameter_advice_cells =
            if settings.run_args.param_visibility.is_hashed_public() { 1 } else { 0 };
        let hashed_parameter_advice_columns = hashed_parameter_advice_cells as usize;
        let total_advice_logical_cells = main_advice_cells
            + extra_advice_cells
            + einsum_advice_cells
            + hashed_parameter_advice_cells;
        let core_advice_columns = main_advice_columns
            + dynamic_advice_columns
            + einsum_advice_columns
            + hashed_parameter_advice_columns;
        let core_advice_polynomial_bytes =
            (core_advice_columns as u128) * (domain as u128) * 32;
        let dense_advice_domain_bytes =
            (native_cs.num_advice_columns() as u128) * (domain as u128) * 32;
        let dense_fixed_domain_bytes =
            (native_cs.num_fixed_columns() as u128) * (domain as u128) * 32;
        let scalar_domain_storage_bytes = (native_cs.num_advice_columns() as u128
            + native_cs.num_fixed_columns() as u128
            + native_cs.num_instance_columns() as u128)
            * (domain as u128)
            * 32;
        eprintln!(
            "EZKL_FORMAL_GEOMETRY dataset={} resource_manifest_sha256={} n_total={} n={} e={} graphs={} feature_dim={} classes={} hidden_layers={} hidden_heads={} output_heads={} parameter_count={} k={} N={} domain={} inner_cols={} rows={} T={} range_gadget_version={} range_blinding_factor_reserve={} usable_advice_rows={} main_blocks={} main_advice_columns={} main_advice_cells={} main_advice_logical_bytes={} einsum_equations={} einsum_col_size={} einsum_reduction_capacity={} einsum_blocks={} einsum_advice_columns={} einsum_advice_cells={} einsum_advice_first_phase_cells={} einsum_advice_second_phase_cells={} einsum_max_output_axes={} dynamic_lookups={} dynamic_col_size={} dynamic_max_input={} shuffle_col_size={} dynamic_blocks={} extra_advice_arrays={} dynamic_advice_columns={} extra_advice_cells={} extra_advice_logical_bytes={} hashed_parameter_advice_cells={} hashed_parameter_advice_columns={} total_advice_logical_cells={} total_advice_logical_bytes={} core_advice_columns={} core_advice_polynomial_bytes={} dense_advice_domain_bytes={} dense_fixed_domain_bytes={} scalar_domain_storage_bytes={} const_cells={} required_ranges={} F0={} A={} P={} degree={} L={} Q={} T_lookup_groups={} newT={} per_argument_Q={:?} custom_gate_groups={} shuffle_arguments={} rawSelectors={} model_instance_shapes={:?} module_sizes={:?} total_instances={:?}",
            loaded.dataset,
            loaded.resource_manifest_sha256,
            loaded.spec.graph.n_total,
            loaded.spec.graph.n,
            loaded.spec.graph.e,
            loaded.spec.graph.g_batch,
            loaded.spec.feature_dim,
            loaded.spec.model.classes,
            loaded.spec.model.hidden_layers.len(),
            loaded.spec.model.hidden_layers.iter().map(|layer| layer.heads.len()).sum::<usize>(),
            loaded.spec.model.output_heads.len(),
            parameter_count,
            settings.run_args.logrows,
            domain,
            domain,
            settings.run_args.num_inner_cols,
            settings.num_rows,
            settings.total_assignments,
            settings.gat_range_gadget_version.unwrap_or_default(),
            ezkl::gat_adapter::GAT_RANGE_BLINDING_FACTOR_RESERVE,
            usable_advice_rows,
            main_blocks,
            main_advice_columns,
            main_advice_cells,
            main_advice_logical_bytes,
            settings.einsum_params.equations.len(),
            settings.einsum_params.total_einsum_col_size,
            einsum_analysis.reduction_length,
            einsum_blocks,
            einsum_advice_columns,
            einsum_advice_cells,
            (einsum_analysis.reduction_length as u128) * 3,
            (einsum_analysis.reduction_length as u128) * 3,
            einsum_analysis.max_num_output_axes,
            settings.dynamic_lookup_params.num_dynamic_lookups,
            settings.dynamic_lookup_params.total_dynamic_col_size,
            settings.dynamic_lookup_params.max_dynamic_input_len,
            settings.shuffle_params.total_shuffle_col_size,
            dynamic_blocks,
            if settings.requires_dynamic_lookup() || settings.requires_shuffle() { 3 } else { 0 },
            dynamic_advice_columns,
            extra_advice_cells,
            extra_advice_bytes,
            hashed_parameter_advice_cells,
            hashed_parameter_advice_columns,
            total_advice_logical_cells,
            total_advice_logical_cells * 32,
            core_advice_columns,
            core_advice_polynomial_bytes,
            dense_advice_domain_bytes,
            dense_fixed_domain_bytes,
            scalar_domain_storage_bytes,
            settings.total_const_size,
            settings.required_range_checks.len(),
            native_cs.num_fixed_columns(),
            native_cs.num_advice_columns(),
            native_cs.num_permutation_columns(),
            native_degree,
            native_lookup_arguments,
            native_query_polynomials,
            native_table_groups,
            native_chunked_table_groups,
            native_query_counts,
            native_cs.gates().len(),
            native_cs.shuffles().len(),
            cs.num_selectors(),
            &settings.model_instance_shapes,
            &settings.module_sizes,
            settings.total_instances(),
        );
        drop(native_cs);
        drop(circuit_config);
        drop(cs);
        drop(settings);
        drop(model);
        drop(loaded);
    }
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
    let (known, witness, compiled_logrows) = source_built_tiny_gat_fixture(TINY_GAT_TEST_LOGROWS);
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
    #[cfg(feature = "formal-gpu")]
    let gpu_stats_before_prove = {
        assert_eq!(
            std::env::var("ENABLE_ICICLE_GPU").unwrap_or_default(),
            "1",
            "formal GPU proof fixture must not silently use the CPU MSM backend",
        );
        halo2_proofs::icicle::execution_stats()
    };
    let started = Instant::now();
    eprintln!("EZKL_TEST_PHASE START stage=tiny_native_gat_prove");
    let mut writer = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        std::slice::from_ref(&known),
        instances,
        &mut rng,
        &mut writer,
    )
    .expect("tiny GAT fixture proof generation");
    let proof = writer.finalize();
    #[cfg(feature = "formal-gpu")]
    {
        let gpu_stats_after_prove = halo2_proofs::icicle::execution_stats();
        let msm_operations = gpu_stats_after_prove
            .0
            .checked_sub(gpu_stats_before_prove.0)
            .expect("native ICICLE MSM operation counter did not reset during proof");
        let gpu_msm_chunks = gpu_stats_after_prove
            .1
            .checked_sub(gpu_stats_before_prove.1)
            .expect("native ICICLE GPU chunk counter did not reset during proof");
        let synchronized_calls = gpu_stats_after_prove
            .4
            .checked_sub(gpu_stats_before_prove.4)
            .expect("native ICICLE synchronization counter did not reset during proof");
        assert!(
            msm_operations > 0 && gpu_msm_chunks >= msm_operations,
            "formal GAT proof did not dispatch its MSM commitments to native ICICLE GPU"
        );
        assert_eq!(
            synchronized_calls, gpu_msm_chunks,
            "native GPU MSM chunks must be synchronized before proof completion"
        );
    }
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

    assert!(known.settings().run_args.output_visibility.is_public());
    let output_count = witness.outputs.iter().map(|output| output.len()).sum::<usize>();
    assert!(output_count > 0, "tiny GAT fixture must expose public logits");
    assert!(public.len() >= output_count, "public instance vector contains every GAT logit");
    let mut tampered_public = public.clone();
    let tampered_logit = tampered_public
        .last_mut()
        .expect("public output logits are the final public instances");
    *tampered_logit += Fr::from(1u64);
    let tampered_instances: &[&[&[Fr]]] = &[&[&tampered_public]];
    let mut tampered_instance_reader = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
    let tampered_instance_result =
        verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<_>, _, _, SingleStrategy<_>>(
            &params,
            pk.get_vk(),
            SingleStrategy::new(&params),
            tampered_instances,
            &mut tampered_instance_reader,
            1u64 << k,
        );
    assert!(
        tampered_instance_result.is_err(),
        "independent verifier accepted a proof against a modified public logit"
    );
    eprintln!("EZKL_TEST_PHASE DONE stage=tampered_public_logit_rejected");

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
fn native_h_part_relation() {
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column,
            ConstraintSystem, Error, Fixed, Selector,
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
    };
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use rand::rngs::OsRng;

    #[derive(Clone, Copy)]
    struct HPartCircuit;

    #[derive(Clone)]
    struct HPartConfig {
        advice: Vec<Column<Advice>>,
        table: Column<Fixed>,
        selector: Selector,
    }

    impl Circuit<Fr> for HPartCircuit {
        type Config = HPartConfig;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            *self
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let advice = (0..8)
                .map(|_| meta.advice_column())
                .collect::<Vec<_>>();
            for column in &advice {
                meta.enable_equality(*column);
            }
            let table = meta.fixed_column();
            let selector = meta.selector();
            meta.create_gate("strided H rotation", |meta| {
                let q = meta.query_selector(selector);
                let current = meta.query_advice(advice[0], Rotation::cur());
                let next = meta.query_advice(advice[0], Rotation::next());
                vec![q * (current - next)]
            });
            meta.lookup_any("strided H lookup", |meta| {
                vec![(
                    meta.query_advice(advice[0], Rotation::cur()),
                    meta.query_fixed(table, Rotation::cur()),
                )]
            });
            meta.shuffle("strided H shuffle", |meta| {
                vec![(
                    meta.query_advice(advice[0], Rotation::cur()),
                    meta.query_advice(advice[1], Rotation::cur()),
                )]
            });
            // k=5 and degree 9 produce an 8-way extended domain, so the proof
            // evaluates all eight strided H parts.
            meta.set_minimum_degree(9);
            HPartConfig {
                advice,
                table,
                selector,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "strided H rotations and arguments",
                |mut region| {
                    for row in 0..2 {
                        let value = Fr::from(1);
                        if row == 0 {
                            config.selector.enable(&mut region, row)?;
                        }
                        region.assign_fixed(
                            || "lookup table value",
                            config.table,
                            row,
                            || Value::known(value),
                        )?;
                        let mut cells = Vec::with_capacity(config.advice.len());
                        for (column_index, column) in config.advice.iter().enumerate() {
                            cells.push(region.assign_advice(
                                || "advice value",
                                *column,
                                row,
                                || Value::known(value),
                            )?);
                            if column_index > 0 {
                                region.constrain_equal(cells[0].cell(), cells[column_index].cell())?;
                            }
                        }
                    }
                    Ok(())
                },
            )?;
            Ok(())
        }
    }

    let k = 5;
    let circuit = HPartCircuit;
    let mut rng = OsRng;
    let params = ParamsKZG::<Bn256>::setup(k, &mut rng);
    let vk = keygen_vk(&params, &circuit).expect("H-part relation VK");
    assert_eq!(
        vk.get_domain().extended_k() - k,
        3,
        "k=5 and minimum degree 9 must exercise all m=8 H parts"
    );
    let pk = keygen_pk(&params, vk, &circuit).expect("H-part relation PK");
    let instances: &[&[&[Fr]]] = &[&[]];
    let mut writer = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[circuit],
        instances,
        &mut rng,
        &mut writer,
    )
    .expect("strided H proof satisfies gate, permutation, lookup, and shuffle relations");
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
    .expect("independent verification accepts the strided H proof");
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
            Column, ConstraintSystem, Error, ProvingKey, Selector,
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

#[test]
fn native_fixed_storage_contract() {
    use ezkl::pfsys::{load_pk_with_native_backing_budget_and_timing, save_pk};
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            keygen_pk, keygen_vk_custom, Advice, Circuit, Column, ConstraintSystem, Error, Fixed,
            Selector,
        },
        poly::{
            kzg::commitment::{KZGCommitmentScheme, ParamsKZG},
            Rotation,
        },
        SerdeFormat,
    };
    use halo2curves::bn256::{Bn256, Fr};
    use rand::rngs::OsRng;

    #[derive(Clone, Copy)]
    struct FixedStorageCircuit;

    impl Circuit<Fr> for FixedStorageCircuit {
        type Config = (Column<Advice>, Column<Advice>, Column<Fixed>, Selector);
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            *self
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let left = meta.advice_column();
            let right = meta.advice_column();
            let fixed = meta.fixed_column();
            let selector = meta.selector();
            meta.enable_equality(left);
            meta.enable_equality(right);
            meta.create_gate("fixed source and selector", |meta| {
                let q = meta.query_selector(selector);
                let value = meta.query_advice(left, Rotation::cur());
                let fixed_value = meta.query_fixed(fixed, Rotation::cur());
                vec![q * (value - fixed_value)]
            });
            (left, right, fixed, selector)
        }

        fn synthesize(
            &self,
            (left, right, fixed, selector): Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "compact fixed source and permutation edges",
                |mut region| {
                    for row in 0..5 {
                        let value = Fr::from((row + 1) as u64);
                        selector.enable(&mut region, row)?;
                        region.assign_fixed(|| "fixed", fixed, row, || Value::known(value))?;
                        let left_cell = region.assign_advice(|| "left", left, row, || Value::known(value))?;
                        let right_cell = region.assign_advice(|| "right", right, row, || Value::known(value))?;
                        region.constrain_equal(left_cell.cell(), right_cell.cell())?;
                    }
                    Ok(())
                },
            )
        }
    }

    let mut rng = OsRng;
    let params = ParamsKZG::<Bn256>::setup(5, &mut rng);
    let circuit = FixedStorageCircuit;
    let vk = keygen_vk_custom(&params, &circuit, true).expect("compact fixed-source VK");
    let pk = keygen_pk(&params, vk, &circuit).expect("compact fixed-source PK");
    assert!(pk.get_vk().fixed_commitments().len() >= 2);

    let mut vk_before = Vec::new();
    pk.get_vk()
        .write(&mut vk_before, SerdeFormat::RawBytes)
        .unwrap();
    let pk_before = pk.to_bytes(SerdeFormat::RawBytes);
    let temp = tempfile::Builder::new()
        .prefix("ezkl-native-fixed-pk-")
        .tempdir()
        .unwrap();
    let pk_path = temp.path().join("fixed.pk");
    save_pk(&pk_path, &pk).unwrap();
    drop(pk);

    let (restored, timing) = load_pk_with_native_backing_budget_and_timing::<
        KZGCommitmentScheme<Bn256>,
        FixedStorageCircuit,
    >(pk_path, (), 64 * 1024 * 1024)
    .expect("file-backed fixed and permutation PK restore");
    assert_eq!(timing.sigma_rebuild, std::time::Duration::ZERO);
    let mut vk_after = Vec::new();
    restored
        .get_vk()
        .write(&mut vk_after, SerdeFormat::RawBytes)
        .unwrap();
    assert_eq!(vk_before, vk_after);
    assert_eq!(pk_before, restored.to_bytes(SerdeFormat::RawBytes));
}
