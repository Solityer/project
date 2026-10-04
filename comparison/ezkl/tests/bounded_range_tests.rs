use super::*;
use crate::circuit::ops::region::RegionSettings;
use halo2_proofs::{circuit::{Layouter, SimpleFloorPlanner}, dev::MockProver, plonk::{Circuit, ConstraintSystem, Error}};

const K: usize = 10;

#[derive(Clone)]
struct RangeCircuit {
    value: i128,
    bounds: (i128, i128),
}

#[derive(Clone, Default)]
struct TestParams;

impl Circuit<Fp> for RangeCircuit {
    type Config = BaseConfig<Fp>;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = TestParams;

    fn without_witnesses(&self) -> Self { Self { value: 0, bounds: self.bounds } }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let input = crate::tensor::VarTensor::new_advice(meta, K, 1, 128);
        let index = crate::tensor::VarTensor::new_advice(meta, K, 1, 128);
        let output = crate::tensor::VarTensor::new_advice(meta, K, 1, 128);
        crate::tensor::VarTensor::constant_cols(meta, K, 128, false);
        let mut config = BaseConfig::configure(meta, &[input.clone(), index.clone()], &output, crate::circuit::ops::chip::CheckMode::SAFE);
        for range in [(-1, 1), (0, 1), (0, 255)] {
            config.configure_range_check(meta, &input, &index, range, K).unwrap();
        }
        config
    }

    fn synthesize(&self, mut config: Self::Config, mut layouter: impl Layouter<Fp>) -> Result<(), Error> {
        config.layout_range_checks(&mut layouter).map_err(|_| Error::Synthesis)?;
        let value = Self::witness(self.value);
        layouter.assign_region(|| "bounded signed range", |region| {
            let settings = RegionSettings::new(true, false, 256, 4);
            let mut region = RegionCtx::new_with_settings(region, 0, 1, settings);
            GatOp::range_check_bounded(&config, &mut region, &[&value], &self.bounds).map_err(|_| Error::Synthesis)
        })
    }
}

impl RangeCircuit {
    fn witness(value: i128) -> ValTensor<Fp> {
        Tensor::new(Some(&[ValType::Value(Value::known(integer_rep_to_felt(value)))]), &[1]).unwrap().into()
    }
}

fn verify(value: i128, bounds: (i128, i128)) -> Result<(), String> {
    let circuit = RangeCircuit { value, bounds };
    let prover = MockProver::run(K as u32, &circuit, vec![]).map_err(|error| format!("mock prover synthesis failed: {error:?}"))?;
    prover.verify().map_err(|errors| format!("mock prover constraints failed: {errors:?}"))
}

#[test]
fn wide_signed_ranges_accept_endpoints_and_reject_adjacent_values() {
    let signed = (-11_341_398_016, 11_341_398_016);
    for value in [signed.0, 0, signed.1] {
        assert!(verify(value, signed).is_ok(), "valid signed value {value} was rejected");
    }
    assert!(verify(signed.0 - 1, signed).is_err());
    assert!(verify(signed.1 + 1, signed).is_err());
    let nonnegative = (0, 22_683_142_144);
    for value in [nonnegative.0, nonnegative.1] {
        assert!(verify(value, nonnegative).is_ok(), "valid nonnegative value {value} was rejected");
    }
    assert!(verify(nonnegative.1 + 1, nonnegative).is_err());
    let arxiv = (0, 452_002_385_172_480);
    for value in [arxiv.0, arxiv.1] {
        assert!(verify(value, arxiv).is_ok(), "valid seven-limb value {value} was rejected");
    }
    assert!(verify(arxiv.0 - 1, arxiv).is_err());
    assert!(verify(arxiv.1 + 1, arxiv).is_err());
}

#[test]
fn wide_range_dummy_layout_uses_only_compact_tables_for_known_and_unknown_values() {
    let bounds = (0, 22_683_142_144);
    let layout = |known: bool| {
        let config = BaseConfig::<Fp>::dummy(16, 1);
        let settings = RegionSettings::all_true(256, 4);
        let mut region = RegionCtx::new_dummy(0, 1, settings);
        let value = if known { RangeCircuit::witness(bounds.1) } else { Tensor::new(Some(&[ValType::Value(Value::unknown())]), &[1]).unwrap().into() };
        GatOp::range_check_bounded(&config, &mut region, &[&value], &bounds).unwrap();
        (region.linear_coord(), region.used_range_checks())
    };
    let known = layout(true);
    let unknown = layout(false);
    assert_eq!(known, unknown);
    assert_eq!(known.1, [(-1, 1), (0, 1), (0, 255)].into_iter().collect());
}

#[test]
fn wide_range_rejects_unrepresentable_spans_before_decomposition() {
    let config = BaseConfig::<Fp>::dummy(16, 1);
    let mut region = RegionCtx::new_dummy(0, 1, RegionSettings::all_true(256, 4));
    let value = RangeCircuit::witness(0);
    assert!(GatOp::range_check_bounded(&config, &mut region, &[&value], &(i128::MIN, i128::MAX)).is_err());
    assert!(GatOp::range_check_bounded(&config, &mut region, &[&value], &(0, 1_i128 << 120)).is_err());
    assert!(GatOp::checked_signed_field(integer_rep_to_felt(i128::MIN)) == Some(i128::MIN));
    assert!(GatOp::checked_signed_field(Fp::from_u128(1_u128 << 127)).is_none());
    assert!(verify(i128::MIN, (i128::MIN, i128::MIN + 2048)).is_ok());
    assert!(verify(0, (i128::MIN, i128::MIN + 2048)).is_err());
}
