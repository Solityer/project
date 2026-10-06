use ezkl::gat_adapter::schema::HiddenLayerSpec;
use ezkl::gat_adapter::{
    evaluate_signed_division, scale_exponent_to_multiplier, CanonicalGat, CanonicalGraph, CanonicalModel, ExactTable, GatOp, HeadWeights, ModelBinding,
    RoundingMode,
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
    odd_boundary.validate(RoundingMode::NearestTiesAway).unwrap();

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
    assert!(duplicate.validate().unwrap_err().contains("duplicate directed edge"));

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
    assert!(missing_loop.validate().unwrap_err().contains("missing its required per-node self-loop"));
}

#[test]
fn redaction_keeps_parameter_shape_but_removes_private_values() {
    let model = CanonicalModel {
        hidden_layers: vec![HiddenLayerSpec {
            d_in: 1,
            heads: vec![HeadWeights { projection: vec![17], a_src: vec![-3], a_dst: vec![5], bias: vec![],}],
            d_head: 1,
        }],
        output_heads: vec![HeadWeights { projection: vec![11], a_src: vec![7], a_dst: vec![-2], bias: vec![13],}],
        classes: 1,
        model_arch_id: "test".into(),
    };
    let before_len = model.flattened_params().len();
    let mut op = GatOp {
        spec: CanonicalGat {
            model,
            model_binding: ModelBinding { hash: "0x01".into(), hash_domain: "ezkl-gat-model-v1".into(), canonical_model_digest: "00".repeat(32),},
            ..CanonicalGat::default()
        },
        rounding: RoundingMode::NearestTiesAway,
    };
    op.redact_private_params();
    assert_eq!(op.spec.model.flattened_params().len(), before_len);
    assert!(op.spec.model.flattened_params().iter().all(|value| *value == 0));
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
    let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gat_lookup_reuse");
    let circuit = GraphCircuit::load(fixture.join("compiled.circuit")).unwrap();
    let witness = GraphWitness::from_path(fixture.join("witness.json")).unwrap();
    let mut known_inputs = witness.get_input_tensor().into_iter().map(|input| input.map(|value| Value::known(value)).into()).collect::<Vec<ValTensor<Fr>>>();
    let mut known_params = witness.get_param_tensor().into_iter().map(|input| input.map(|value| Value::known(value)).into()).collect::<Vec<ValTensor<Fr>>>();
    for (i, input) in known_inputs.iter_mut().enumerate() {
        input.reshape(&circuit.model().graph.input_shapes().unwrap()[i]).unwrap();
        input.set_scale(circuit.settings().model_input_scales[i]);
    }
    for (input, shape) in known_params.iter_mut().zip(circuit.model().graph.param_shapes().unwrap()) {
        input.reshape(&shape).unwrap();
        input.set_scale(circuit.settings().run_args.param_scale);
    }
    let unknown_inputs = known_inputs.iter().map(|input| Tensor::new(Some(&vec![Value::<Fr>::unknown(); input.len()]), input.dims()).unwrap().into()).collect::<Vec<ValTensor<Fr>>>();
    let unknown_params = known_params.iter().map(|input| Tensor::new(Some(&vec![Value::<Fr>::unknown(); input.len()]), input.dims()).unwrap().into()).collect::<Vec<ValTensor<Fr>>>();
    let args = &circuit.settings().run_args;
    let _ = env_logger::try_init();
    let settings = RegionSettings::all_false(args.decomp_base, args.decomp_legs);
    let known = circuit.model().dummy_layout_with_runtime_params(args, &known_inputs, &known_params, settings.clone()).unwrap();
    let unknown = circuit.model().dummy_layout_with_runtime_params(args, &unknown_inputs, &unknown_params, settings).unwrap();
    assert_eq!(known.linear_coord, unknown.linear_coord);
    assert_eq!(known.dynamic_lookup_params, unknown.dynamic_lookup_params);
    assert_eq!(known.shuffle_params, unknown.shuffle_params);
    assert_eq!(known.einsum_params, unknown.einsum_params);
    assert_eq!(known.total_const_size, unknown.total_const_size);
    assert!(known.dynamic_lookup_params.num_dynamic_lookups < 64);
}

#[test]
fn halo2_commitment_timing_counts_public_and_phase_zero_only() {
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Challenge, Circuit,
            Column, ConstraintSystem, Error, FirstPhase, Instance, SecondPhase,
            Selector,
        },
        poly::{
            Rotation,
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
            PhaseConfig { first, second, instance, challenge, selector }
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
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
    let public = [Fr::from(7)];
    let instances: &[&[&[Fr]]] = &[&[&public]];
    let mut transcript =
        Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
    let measurement = halo2_proofs::plonk::commitment_timing::begin().unwrap();
    create_proof::<
        KZGCommitmentScheme<Bn256>,
        ProverSHPLONK<_>,
        _,
        _,
        _,
        _,
    >(
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
    let mut verifier_transcript =
        Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
    verify_proof::<
        KZGCommitmentScheme<Bn256>,
        VerifierSHPLONK<_>,
        _,
        _,
        _,
    >(
        &params,
        pk.get_vk(),
        SingleStrategy::new(&params),
        instances,
        &mut verifier_transcript,
        params.n(),
    )
    .unwrap();
}
