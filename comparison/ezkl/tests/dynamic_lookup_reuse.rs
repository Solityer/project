use super::*;

const K_REUSE: usize = 6;

#[derive(Clone)]
struct ReuseCircuit {
    wrong_value: bool,
    unknown: bool,
}

impl ReuseCircuit {
    fn tensor(values: &[u64], constant: bool, unknown: bool) -> ValTensor<F> {
        let vals = values.iter().map(|v| if constant { crate::tensor::ValType::Constant(F::from(*v)) } else { crate::tensor::ValType::Value(if unknown { Value::unknown() } else { Value::known(F::from(*v)) }) }).collect::<Vec<_>>();
        Tensor::new(Some(&vals), &[vals.len()]).unwrap().into()
    }

    fn assignments(&self, config: &BaseConfig<F>, region: &mut RegionCtx<F>) -> Result<(), CircuitError> {
        let keys = Self::tensor(&[0, 1], true, false);
        let first = Self::tensor(&[10, 11], true, false);
        let second = Self::tensor(&[20, 21], true, false);
        let a = layouts::assign_dynamic_lookup_table(config, region, &[&keys, &first])?;
        let b = layouts::assign_dynamic_lookup_table(config, region, &[&keys, &second])?;
        for (key, value, table) in [(0, 10, a), (1, if self.wrong_value { 21 } else { 11 }, a), (1, 21, b)] {
            let input = Self::tensor(&[key], false, self.unknown);
            let output = Self::tensor(&[value], false, self.unknown);
            layouts::dynamic_lookup_from_table(config, region, &[&input, &output], table)?;
        }
        assert_eq!(region.dynamic_lookup_index(), 2);
        assert_eq!(region.dynamic_lookup_col_coord(), 4);
        assert_eq!(region.linear_coord(), 3);
        Ok(())
    }
}

impl Circuit<F> for ReuseCircuit {
    type Config = BaseConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = TestParams;

    fn without_witnesses(&self) -> Self { Self { wrong_value: false, unknown: true } }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let a = VarTensor::new_advice(cs, K_REUSE, 1, 8);
        let b = VarTensor::new_advice(cs, K_REUSE, 1, 8);
        let c = VarTensor::new_advice(cs, K_REUSE, 1, 8);
        let d = VarTensor::new_advice(cs, K_REUSE, 1, 8);
        let e = VarTensor::new_advice(cs, K_REUSE, 1, 8);
        let f = VarTensor::new_advice(cs, K_REUSE, 1, 8);
        VarTensor::constant_cols(cs, K_REUSE, 16, false);
        let mut config = BaseConfig::configure(cs, &[a.clone(), b.clone()], &c, CheckMode::SAFE);
        config.configure_dynamic_lookup(cs, &[a, b, c], &[d, e, f]).unwrap();
        config
    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        layouter.assign_region(|| "reused native tables", |region| self.assignments(&config, &mut RegionCtx::new(region, 0, 1, 128, 2)).map_err(|_| Error::Synthesis))
    }
}

#[test]
fn reused_native_table_queries_preserve_identity_and_layout() {
    let valid = ReuseCircuit { wrong_value: false, unknown: false };
    MockProver::run(K_REUSE as u32, &valid, vec![]).unwrap().assert_satisfied();
    let invalid = ReuseCircuit { wrong_value: true, unknown: false };
    assert!(MockProver::run(K_REUSE as u32, &invalid, vec![]).unwrap().verify().is_err());
    let mut cs = ConstraintSystem::default();
    let config = ReuseCircuit::configure(&mut cs);
    let mut dummy = RegionCtx::new_dummy(0, 1, ops::region::RegionSettings::all_false(128, 2));
    valid.without_witnesses().assignments(&config, &mut dummy).unwrap();
}
