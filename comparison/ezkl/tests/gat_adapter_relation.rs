use ezkl::gat_adapter::schema::HiddenLayerSpec;
use ezkl::gat_adapter::{
    evaluate_signed_division, scale_exponent_to_multiplier, CanonicalGat, CanonicalModel, ExactTable, GatOp, HeadWeights, ModelBinding,
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
