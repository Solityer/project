//! Host-side canonical evaluator used to construct witnesses for the native
//! ezkl graph operation.
//!
//! This evaluator is intentionally a *reference* for witness generation and
//! manifest validation.  The circuit operation in [`super::op`] repeats the
//! same relations with Halo2 constraints; callers must not use this module as
//! a host-only replacement for the circuit.

use num::{BigUint, Zero};

use super::schema::{parse_authenticated_bn254_modulus, scale_exponent_to_multiplier, CanonicalGat, ExactTable,};

/// Rounding rule used by signed fixed-point divisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RoundingMode {
    /// Truncate toward zero.
    TowardZero,
    /// Round to nearest integer, with ties away from zero.
    NearestTiesAway,
}

/// One checked signed quotient and remainder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedDivision {
    /// Sign bit of the dividend.
    pub sigma: bool,
    /// Zero-dividend bit.
    pub zero: bool,
    /// Absolute dividend.
    pub abs_dividend: i128,
    /// Positive divisor.
    pub divisor: i128,
    /// Signed quotient after rounding.
    pub quotient: i128,
    /// Non-negative quotient before nearest-rounding.
    pub quotient_floor: i128,
    /// Whether nearest-rounding incremented the absolute quotient.
    pub rounded_up: bool,
    /// Non-negative remainder in the canonical division identity.
    pub remainder: i128,
    /// Canonical non-negative remainder complement (`d - 1 - r` for toward
    /// zero, `2*d - 1 - r` for nearest/ties-away).
    pub delta: i128,
}

impl SignedDivision {
    /// Validate the complete canonical division witness.
    ///
    /// This is intentionally independent of the constructor so callers and
    /// tests can reject a witness whose remainder or sign bit was tampered
    /// after generation.
    pub fn validate(&self, mode: RoundingMode) -> Result<(), String> {
        if self.divisor <= 0 {
            return Err("signed division requires a positive divisor".into());
        }
        if self.abs_dividend < 0 || self.quotient_floor < 0 || self.remainder < 0 {
            return Err("signed division witness has a negative unsigned component".into());
        }
        let remainder_upper = match mode {
            RoundingMode::TowardZero => self.divisor.checked_sub(1),
            RoundingMode::NearestTiesAway => { self.divisor.checked_mul(2).and_then(|x| x.checked_sub(1)) } }
        .ok_or_else(|| "signed division remainder bound overflow".to_string())?;
        if self.remainder > remainder_upper {
            return Err("signed division remainder is outside its canonical range".into());
        }
        if self.zero != (self.abs_dividend == 0) {
            return Err("signed division zero bit disagrees with absolute dividend".into());
        }
        if self.zero && self.sigma {
            return Err("zero signed division witness must have a clear sign bit".into());
        }
        let expected_delta = match mode {
            RoundingMode::TowardZero => {
                let reconstructed = self
                    .divisor
                    .checked_mul(self.quotient_floor)
                    .and_then(|x| x.checked_add(self.remainder))
                    .ok_or_else(|| "signed division reconstruction overflow".to_string())?;
                if reconstructed != self.abs_dividend {
                    return Err("signed division quotient/remainder identity failed".into());
                }
                self.divisor.checked_sub(1).and_then(|x| x.checked_sub(self.remainder)).ok_or_else(|| "signed division delta underflow".to_string())?
            }
            RoundingMode::NearestTiesAway => {
                let rounded_abs =
                    self.quotient_floor.checked_add(i128::from(self.rounded_up)).ok_or_else(|| "signed division rounded quotient overflow".to_string())?;
                let lhs = self
                    .abs_dividend
                    .checked_mul(2)
                    .and_then(|x| x.checked_add(self.divisor))
                    .ok_or_else(|| "signed division nearest equation overflow".to_string())?;
                let rhs = self
                    .divisor
                    .checked_mul(2)
                    .and_then(|x| x.checked_mul(rounded_abs))
                    .and_then(|x| x.checked_add(self.remainder))
                    .ok_or_else(|| "signed division nearest equation overflow".to_string())?;
                if lhs != rhs {
                    return Err("signed division nearest equation failed".into());
                }
                self.divisor
                    .checked_mul(2)
                    .and_then(|x| x.checked_sub(1))
                    .and_then(|x| x.checked_sub(self.remainder))
                    .ok_or_else(|| "signed division nearest delta underflow".to_string())?
            }
        };
        if self.delta != expected_delta {
            return Err("signed division remainder complement failed".into());
        }
        let expected_up = match mode {
            RoundingMode::TowardZero => false,
            RoundingMode::NearestTiesAway => {
                let floor_remainder = self
                    .abs_dividend
                    .checked_sub(self.divisor.checked_mul(self.quotient_floor).ok_or_else(|| "signed division floor product overflow".to_string())?,)
                    .ok_or_else(|| "signed division floor remainder underflow".to_string())?;
                floor_remainder.checked_mul(2).ok_or_else(|| "signed division tie comparison overflow".to_string())? >= self.divisor
            }
        };
        if self.rounded_up != expected_up {
            return Err("signed division nearest-rounding bit failed".into());
        }
        let rounded_abs =
            self.quotient_floor.checked_add(i128::from(self.rounded_up)).ok_or_else(|| "signed division rounded quotient overflow".to_string())?;
        let expected_quotient =
            if self.sigma { rounded_abs.checked_neg().ok_or_else(|| "signed division quotient overflow".to_string())? } else { rounded_abs };
        if self.quotient != expected_quotient {
            return Err("signed division sign/quotient relation failed".into());
        }
        Ok(())
    }
}

/// The integer and field paths for one grouped aggregation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregationTrace {
    /// Destination node.
    pub dst: usize,
    /// Exponential numerator for each output channel.
    pub numerators: Vec<i128>,
    /// Positive group denominator.
    pub denominator: i128,
    /// Signed rounded output for each channel.
    pub values: Vec<i128>,
    /// The exact field alpha path, represented canonically as integers.
    pub field_values: Vec<String>,
    /// Division witnesses for every output channel.
    pub divisions: Vec<SignedDivision>,
    /// Optional output-head scale-alignment divisions.
    pub alignment_divisions: Vec<SignedDivision>,
}

/// A hidden layer witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenTrace {
    /// Projected/activated node tensor, flattened by node then head/channel.
    pub node_values: Vec<i128>,
    /// Group aggregation traces for every head and destination.
    pub heads: Vec<Vec<AggregationTrace>>,
}

/// Complete canonical GAT witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatTrace {
    /// Hidden layer traces in model order.
    pub hidden: Vec<HiddenTrace>,
    /// Final output tensor, flattened as `[node, class]`.
    pub output: Vec<i128>,
    /// Output head traces before the final average.
    pub output_heads: Vec<Vec<AggregationTrace>>,
}

fn checked_add(a: i128, b: i128) -> Result<i128, String> {
    a.checked_add(b).ok_or_else(|| "integer addition overflow".into())
}

fn checked_mul(a: i128, b: i128) -> Result<i128, String> {
    a.checked_mul(b).ok_or_else(|| "integer multiplication overflow".into())
}

fn dot(a: &[i128], b: &[i128]) -> Result<i128, String> {
    if a.len() != b.len() {
        return Err("dot product dimensions do not match".into());
    }
    a.iter().zip(b).try_fold(0_i128, |acc, (x, y)| checked_add(acc, checked_mul(*x, *y)?))
}

fn signed_round_div(x: i128, d: i128, mode: RoundingMode) -> Result<SignedDivision, String> {
    if d <= 0 {
        return Err("signed division requires a positive divisor".into());
    }
    let sigma = x < 0;
    let zero = x == 0;
    let abs = x.checked_abs().ok_or_else(|| "signed dividend overflow".to_string())?;
    let q0 = abs / d;
    let r0 = abs % d;
    let rounded_up = match mode {
        RoundingMode::TowardZero => false,
        RoundingMode::NearestTiesAway => { r0.checked_mul(2).ok_or_else(|| "remainder doubling overflow".to_string())? >= d} };
    let rounded_abs = q0.checked_add(i128::from(rounded_up)).ok_or_else(|| "quotient overflow".to_string())?;
    let quotient = if sigma { -rounded_abs } else { rounded_abs };
    let remainder = match mode {
        RoundingMode::TowardZero => r0,
        RoundingMode::NearestTiesAway => abs
            .checked_mul(2)
            .and_then(|x| x.checked_add(d))
            .and_then(|x| x.checked_sub(d.checked_mul(2)?.checked_mul(rounded_abs)?))
            .ok_or_else(|| "nearest remainder overflow".to_string())?,
    };
    let delta = match mode {
        RoundingMode::TowardZero => d.checked_sub(1).and_then(|x| x.checked_sub(remainder)).ok_or_else(|| "remainder complement underflow".to_string())?,
        RoundingMode::NearestTiesAway => d
            .checked_mul(2)
            .and_then(|x| x.checked_sub(1))
            .and_then(|x| x.checked_sub(remainder))
            .ok_or_else(|| "nearest remainder complement underflow".to_string())?,
    };
    let division = SignedDivision { sigma, zero, abs_dividend: abs, divisor: d, quotient, quotient_floor: q0, rounded_up, remainder, delta,};
    division.validate(mode)?;
    Ok(division)
}

/// Expose one signed division witness to the circuit adapter and tests.
pub fn evaluate_signed_division(x: i128, d: i128, mode: RoundingMode,) -> Result<SignedDivision, String> {
    signed_round_div(x, d, mode)
}

fn mod_from_signed(value: i128, modulus: &BigUint) -> BigUint {
    if value < 0 {
        let magnitude = BigUint::from(value.unsigned_abs()) % modulus;
        if magnitude.is_zero() {
            BigUint::zero()
        } else {
            modulus.clone() - magnitude
        }
    } else {
        BigUint::from(value as u128) % modulus
    }
}

fn modulus_from_spec(model: &CanonicalGat) -> Result<BigUint, String> {
    parse_authenticated_bn254_modulus(&model.ledger.field_modulus)
}

fn mod_mul(a: i128, b: i128, modulus: &BigUint) -> BigUint {
    (mod_from_signed(a, modulus) * mod_from_signed(b, modulus)) % modulus
}

fn inverse_mod(a: i128, modulus: &BigUint) -> Result<BigUint, String> {
    let residue = mod_from_signed(a, modulus);
    if residue.is_zero() {
        return Err("group denominator is zero in the field".into());
    }
    Ok(residue.modpow(&(modulus.clone() - BigUint::from(2_u8)), modulus))
}

fn scale_multiplier_i128(exponent: i32, label: &str) -> Result<i128, String> {
    let multiplier = scale_exponent_to_multiplier(exponent)?;
    i128::try_from(multiplier).map_err(|_| { format!("{} scale multiplier does not fit signed host integer", label) }) }

fn lookup(table: &ExactTable, x: i128) -> Result<i128, String> {
    table.lookup(x)
}

fn project(features: &[i128], row: usize, d_in: usize, projection: &[i128], d_out: usize,) -> Result<Vec<i128>, String> {
    let start = row.checked_mul(d_in).ok_or_else(|| "feature offset overflow".to_string())?;
    let end = start.checked_add(d_in).ok_or_else(|| "feature row end overflow".to_string())?;
    let input = features.get(start..end).ok_or_else(|| "feature row is outside feature table".to_string())?;
    (0..d_out)
        .map(|j| {
            let weights = (0..d_in).map(|i| projection[i * d_out + j]).collect::<Vec<_>>();
            dot(input, &weights)
        })
        .collect()
}

fn aggregate(
    graph: &super::schema::CanonicalGraph,
    src_values: &[i128],
    dst_values: &[i128],
    d_out: usize,
    a_src: &[i128],
    a_dst: &[i128],
    lrelu: &ExactTable,
    exp: &ExactTable,
    mode: RoundingMode,
    modulus: &BigUint,
    numerator_scale: i128,
    denominator_scale: i128,
    elu: Option<&ExactTable>,
) -> Result<(Vec<i128>, Vec<AggregationTrace>), String> {
    if src_values.len() != graph.n * d_out || dst_values.len() != graph.n * d_out {
        return Err("attention tensor shape does not match graph".into());
    }
    if a_src.len() != d_out || a_dst.len() != d_out {
        return Err("attention vector shape does not match head width".into());
    }
    let mut scores = Vec::with_capacity(graph.e);
    for edge in 0..graph.e {
        let s = dot(&src_values[graph.src[edge] * d_out..(graph.src[edge] + 1) * d_out], a_src,)?;
        let d = dot(&dst_values[graph.dst[edge] * d_out..(graph.dst[edge] + 1) * d_out], a_dst,)?;
        scores.push(lookup(lrelu, checked_add(s, d)?)?);
    }
    let mut out = vec![0_i128; graph.n * d_out];
    let mut traces = Vec::new();
    for (dst, start, end) in graph.destination_groups()? {
        let max_score = scores[start..end].iter().copied().max().ok_or_else(|| "empty destination group".to_string())?;
        let mut exps = Vec::with_capacity(end - start);
        for score in &scores[start..end] {
            exps.push(lookup(exp, max_score.checked_sub(*score).ok_or_else(|| "score difference overflow".to_string())?,)?);
        }
        let denominator = exps.iter().try_fold(0_i128, |acc, x| checked_add(acc, *x))?;
        if denominator <= 0 {
            return Err("softmax denominator must be positive".into());
        }
        let inv = inverse_mod(denominator, modulus)?;
        let mut numerators = vec![0_i128; d_out];
        let mut raw_field_values = vec![BigUint::zero(); d_out];
        for (local, edge) in (start..end).enumerate() {
            let src = graph.src[edge];
            for channel in 0..d_out {
                numerators[channel] = checked_add(numerators[channel], checked_mul(exps[local], src_values[src * d_out + channel])?,)?;
                raw_field_values[channel] = (raw_field_values[channel].clone() + mod_mul(exps[local], src_values[src * d_out + channel], modulus)) % modulus;
            }
        }
        let values = numerators
            .iter()
            .map(|num| {
                let dividend = checked_mul(*num, numerator_scale)?;
                let divisor = checked_mul(denominator, denominator_scale)?;
                signed_round_div(dividend, divisor, mode).map(|d| d.quotient)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for channel in 0..d_out {
            let raw_numerator = mod_from_signed(numerators[channel], modulus);
            if raw_field_values[channel] != raw_numerator {
                return Err("field alpha numerator disagrees with integer numerator path".into());
            }
        }
        let field_values = numerators.iter().map(|numerator| (mod_from_signed(*numerator, modulus) * &inv) % modulus).collect::<Vec<_>>();
        let divisions = numerators
            .iter()
            .map(|num| { signed_round_div(checked_mul(*num, numerator_scale)?, checked_mul(denominator, denominator_scale)?, mode,) })
            .collect::<Result<Vec<_>, _>>()?;
        let values = if let Some(elu_table) = elu { values.iter().map(|x| lookup(elu_table, *x)).collect::<Result<Vec<_>, _>>()? } else { values };
        out[dst * d_out..(dst + 1) * d_out].copy_from_slice(&values);
        traces.push(AggregationTrace {
            dst,
            numerators,
            denominator,
            values,
            field_values: field_values.into_iter().map(|value| value.to_string()).collect(),
            divisions,
            alignment_divisions: Vec::new(),
        });
    }
    Ok((out, traces))
}

/// Evaluate and validate the complete canonical integer GAT relation.
pub fn evaluate(model: &CanonicalGat, mode: RoundingMode) -> Result<GatTrace, String> {
    model.validate()?;
    evaluate_validated(model, mode)
}

/// Evaluate the complete integer relation for a compile-stage test fixture.
///
/// Native `CircuitRef` is only available after setup, while adapter geometry
/// and circuit-equivalence tests run before key generation. Keep production
/// callers on [`evaluate`], which requires the fully bound native identity;
/// this test-only entry still validates every canonical relation/resource
/// field available before setup and executes the same numeric evaluator.
#[cfg(test)]
pub(crate) fn evaluate_for_compile_for_tests(
    model: &CanonicalGat,
    mode: RoundingMode,
) -> Result<GatTrace, String> {
    model.validate_for_compile()?;
    evaluate_validated(model, mode)
}

fn evaluate_validated(model: &CanonicalGat, mode: RoundingMode) -> Result<GatTrace, String> {
    let modulus = modulus_from_spec(model)?;
    let output_recovery_scale = scale_multiplier_i128(model.scales.s_out_pre, "s_out_pre")?;
    let output_head_scale = scale_multiplier_i128(model.scales.s_out_head, "s_out_head")?;
    let output_head_count = i128::try_from(model.model.output_heads.len()).map_err(|_| "output head count does not fit host integer".to_string())?;
    let graph = &model.graph;
    let mut current = vec![0_i128; graph.n * model.feature_dim];
    for (node, table_row) in graph.node_ids.iter().copied().enumerate() {
        let source = table_row * model.feature_dim;
        current[node * model.feature_dim..(node + 1) * model.feature_dim].copy_from_slice(&model.features[source..source + model.feature_dim]);
    }
    let mut hidden = Vec::new();
    let mut current_dim = model.feature_dim;
    for (layer_idx, layer) in model.model.hidden_layers.iter().enumerate() {
        let mut layer_values = vec![0_i128; graph.n * layer.heads.len() * layer.d_head];
        let mut layer_heads = Vec::new();
        for (head_idx, head) in layer.heads.iter().enumerate() {
            let mut projected = vec![0_i128; graph.n * layer.d_head];
            for node in 0..graph.n {
                let row = project(&current, node, current_dim, &head.projection, layer.d_head)?;
                projected[node * layer.d_head..(node + 1) * layer.d_head].copy_from_slice(&row);
            }
            let (aggregated, traces) = aggregate(
                graph,
                &projected,
                &projected,
                layer.d_head,
                &head.a_src,
                &head.a_dst,
                &model.tables.lrelu,
                &model.tables.exp,
                mode,
                &modulus,
                scale_multiplier_i128(model.scales.s_agg_pre[layer_idx], "s_agg_pre hidden layer")?,
                scale_multiplier_i128(model.scales.s_h_prime[layer_idx][head_idx], "hidden projection scale",)?,
                Some(&model.tables.elu),
            )?;
            for node in 0..graph.n {
                let base = (node * layer.heads.len() + head_idx) * layer.d_head;
                layer_values[base..base + layer.d_head].copy_from_slice(&aggregated[node * layer.d_head..(node + 1) * layer.d_head]);
            }
            layer_heads.push(traces);
        }
        current = layer_values.clone();
        current_dim = layer.heads.len() * layer.d_head;
        hidden.push(HiddenTrace { node_values: layer_values, heads: layer_heads,});
    }
    let mut output_heads = Vec::new();
    let mut head_outputs = Vec::new();
    for (head_idx, head) in model.model.output_heads.iter().enumerate() {
        let mut projected = vec![0_i128; graph.n * model.model.classes];
        for node in 0..graph.n {
            let row = project(&current, node, current_dim, &head.projection, model.model.classes,)?;
            projected[node * model.model.classes..(node + 1) * model.model.classes].copy_from_slice(&row);
        }
        let (aggregated, mut traces) = aggregate(
            graph,
            &projected,
            &projected,
            model.model.classes,
            &head.a_src,
            &head.a_dst,
            &model.tables.lrelu,
            &model.tables.exp,
            mode,
            &modulus,
            output_recovery_scale,
            scale_multiplier_i128(model.scales.s_y_prime[head_idx], "output projection scale")?,
            None,
        )?;
        let with_bias = aggregated
            .chunks_exact(model.model.classes)
            .zip(std::iter::repeat(&head.bias))
            .flat_map(|(row, bias)| {
                row.iter().zip(bias).map(|(x, b)| {
                    let aligned = signed_round_div(checked_mul(*x, output_head_scale)?, scale_multiplier_i128(model.scales.s_out_pre, "s_out_pre")?, mode,)?;
                    let value = checked_add(aligned.quotient, *b)?;
                    Ok::<i128, String>(value)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let out_pre = scale_multiplier_i128(model.scales.s_out_pre, "s_out_pre")?;
        for trace in &mut traces {
            trace.alignment_divisions =
                trace.values.iter().map(|value| { signed_round_div(checked_mul(*value, output_head_scale)?, out_pre, mode) }).collect::<Result<Vec<_>, _>>()?;
        }
        head_outputs.push(with_bias);
        output_heads.push(traces);
    }
    let mut output = vec![0_i128; graph.n * model.model.classes];
    for i in 0..output.len() {
        output[i] = signed_round_div(head_outputs.iter().try_fold(0_i128, |acc, h| checked_add(acc, h[i]))?, output_head_count, RoundingMode::NearestTiesAway,)?
            .quotient;
    }
    Ok(GatTrace { hidden, output, output_heads,}) }
