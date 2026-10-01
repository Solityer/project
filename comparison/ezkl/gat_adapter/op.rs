//! Native ezkl `Op` implementation for the canonical integer GAT relation.
//!
//! The operation is deliberately one graph node so the adapter can preserve
//! the official `GraphCircuit`/`SupportedOp`/settings path.  Internally it is
//! composed from ezkl's existing einsum, polynomial, shuffle/max, dynamic
//! lookup and equality layouts.  No witness is accepted without a matching
//! circuit relation.

use halo2_proofs::circuit::Value;
use halo2curves::bn256::Fr as Fp;
use halo2curves::ff::Field;

use crate::circuit::ops::region::RegionCtx;
use crate::circuit::ops::{base::BaseOp, layouts, Op};
use crate::circuit::{BaseConfig, CircuitError};
use crate::fieldutils::{integer_rep_to_felt, IntegerRep};
use crate::tensor::{Tensor, ValTensor, ValType};

use super::relation::RoundingMode;
use super::schema::{scale_exponent_to_multiplier, BoundFamily, CanonicalGat, ExactTable};

/// Input positions used by [`GatOp`].
pub const FEATURE_INPUT: usize = 0;
/// Public node-id input position.
pub const NODE_ID_INPUT: usize = 1;
/// Public source-edge input position.
pub const SRC_INPUT: usize = 2;
/// Public destination-edge input position.
pub const DST_INPUT: usize = 3;
/// Public node-prefix input position.
pub const NODE_PTR_INPUT: usize = 4;
/// Public edge-prefix input position.
pub const EDGE_PTR_INPUT: usize = 5;
/// First private model tensor input position.
pub const FIRST_WEIGHT_INPUT: usize = 6;

/// A native graph operation carrying the complete canonical GAT contract.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GatOp {
    /// Canonical topology, model shapes, table identities and range ledger.
    pub spec: CanonicalGat,
    /// Exact signed-division rounding mode.
    pub rounding: RoundingMode,
}

impl GatOp {
    /// Construct and validate a canonical operation.
    pub fn new(spec: CanonicalGat, rounding: RoundingMode) -> Result<Self, String> {
        spec.validate_for_compile()?;
        Ok(Self { spec, rounding })
    }

    fn bound_family(&self, layer: usize, head: usize, is_output: bool,) -> Result<&BoundFamily, CircuitError> {
        self.spec
            .ledger
            .families
            .iter()
            .find(|family| { family.layer == layer && family.head == head && family.is_output == is_output})
            .ok_or(CircuitError::UnsupportedOp)
    }

    /// Remove private model values from a serialized compiled graph while
    /// retaining every shape, scale, bound and public H(W) binding needed to
    /// lay out the runtime witness.  The normal in-memory operation is never
    /// redacted; this is called only on the clone made by `GraphCircuit::save`.
    pub fn redact_private_params(&mut self) {
        for layer in &mut self.spec.model.hidden_layers {
            for head in &mut layer.heads {
                for value in &mut head.projection {
                    *value = 0;
                }
                for value in &mut head.a_src {
                    *value = 0;
                }
                for value in &mut head.a_dst {
                    *value = 0;
                }
            }
        }
        for head in &mut self.spec.model.output_heads {
            for value in &mut head.projection {
                *value = 0;
            }
            for value in &mut head.a_src {
                *value = 0;
            }
            for value in &mut head.a_dst {
                *value = 0;
            }
            for value in &mut head.bias {
                *value = 0;
            }
        }
        self.spec.private_params_redacted = true;
    }

    fn unknown_or_map<FN>(input: &ValTensor<Fp>, dims: &[usize], map: FN,) -> Result<ValTensor<Fp>, CircuitError>
    where
        FN: Fn(IntegerRep) -> Result<IntegerRep, String>,
    {
        let vals = if input.any_unknowns()? {
            vec![ValType::Value(Value::unknown()); dims.iter().product()]
        } else {
            input
                .int_evals()?
                .into_iter()
                .map(|x| {
                    map(x)
                        .map(|v| ValType::Value(Value::known(integer_rep_to_felt(v))))
                        .map_err(|e| { CircuitError::TensorError(crate::tensor::TensorError::DimError(e)) }) })
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut tensor: ValTensor<Fp> = Tensor::new(Some(&vals), dims)?.into();
        tensor.set_scale(input.scale());
        Ok(tensor)
    }

    fn constant_i128(values: &[i128], dims: &[usize]) -> Result<ValTensor<Fp>, CircuitError> {
        let vals = values.iter().map(|x| ValType::Constant(integer_rep_to_felt(*x))).collect::<Vec<_>>();
        Ok(Tensor::new(Some(&vals), dims)?.into())
    }

    /// Create witness/advice values for data-dependent intermediates.
    ///
    /// A known value here is still a `Value::known`, not a circuit constant;
    /// key generation can therefore use unknown advice without baking a
    /// prover's witness into the compiled graph or changing the relation's
    /// fixed layout.
    fn value_i128(values: &[i128], dims: &[usize]) -> Result<ValTensor<Fp>, CircuitError> {
        let vals = values.iter().map(|x| ValType::Value(Value::known(integer_rep_to_felt(*x)))).collect::<Vec<_>>();
        Ok(Tensor::new(Some(&vals), dims)?.into())
    }

    fn unknown_tensor(dims: &[usize]) -> Result<ValTensor<Fp>, CircuitError> {
        let vals = vec![ValType::Value(Value::unknown()); dims.iter().product()];
        Ok(Tensor::new(Some(&vals), dims)?.into())
    }

    fn table(table: &ExactTable) -> Result<(ValTensor<Fp>, ValTensor<Fp>), CircuitError> {
        let keys = table.rows.iter().map(|(x, _)| ValType::Constant(integer_rep_to_felt(*x))).collect::<Vec<_>>();
        let values = table.rows.iter().map(|(_, y)| ValType::Constant(integer_rep_to_felt(*y))).collect::<Vec<_>>();
        Ok((Tensor::new(Some(&keys), &[keys.len()])?.into(), Tensor::new(Some(&values), &[values.len()])?.into(),)) }

    fn exact_lookup(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        input: &ValTensor<Fp>,
        table: &ExactTable,
    ) -> Result<ValTensor<Fp>, CircuitError> {
        let output = Self::unknown_or_map(input, input.dims(), |x| table.lookup(x))?;
        let (keys, values) = Self::table(table)?;
        let (_, checked_output) = layouts::dynamic_lookup(config, region, &[input, &output], &[&keys, &values])?;
        Ok(checked_output)
    }

    fn require_scale(tensor: &ValTensor<Fp>, expected: i32, label: &str,) -> Result<(), CircuitError> {
        if tensor.scale() != expected {
            return Err(CircuitError::DimMismatch(format!("{} scale {} does not match canonical scale {}", label, tensor.scale(), expected)));
        }
        Ok(())
    }

    fn require_dims(tensor: &ValTensor<Fp>, expected: &[usize], label: &str,) -> Result<(), CircuitError> {
        if tensor.dims() != expected {
            return Err(CircuitError::DimMismatch(format!("{} shape {:?} does not match canonical shape {:?}", label, tensor.dims(), expected)));
        }
        Ok(())
    }

    fn topology_constant(&self, which: usize) -> Result<ValTensor<Fp>, CircuitError> {
        let graph = &self.spec.graph;
        let (data, dims): (Vec<i128>, Vec<usize>) = match which {
            NODE_ID_INPUT => (graph.node_ids.iter().map(|x| *x as i128).collect(), vec![graph.n],),
            SRC_INPUT => (graph.src.iter().map(|x| *x as i128).collect(), vec![graph.e],),
            DST_INPUT => (graph.dst.iter().map(|x| *x as i128).collect(), vec![graph.e],),
            NODE_PTR_INPUT => (graph.node_ptr.iter().map(|x| *x as i128).collect(), vec![graph.g_batch + 1],),
            EDGE_PTR_INPUT => (graph.edge_ptr.iter().map(|x| *x as i128).collect(), vec![graph.g_batch + 1],),
            _ => return Err(CircuitError::UnsupportedOp),
        };
        Self::constant_i128(&data, &dims)
    }

    fn bind_topology(&self, config: &BaseConfig<Fp>, region: &mut RegionCtx<Fp>, input: &ValTensor<Fp>, slot: usize,) -> Result<ValTensor<Fp>, CircuitError> {
        let public = layouts::identity(config, region, &[input])?;
        let expected = self.topology_constant(slot)?;
        layouts::enforce_equality(config, region, &[&public, &expected])
    }

    fn weight_tensors<'a>(&self, values: &'a [&ValTensor<Fp>],) -> Result<Vec<ValTensor<Fp>>, CircuitError> {
        let flat = values.get(FIRST_WEIGHT_INPUT).ok_or_else(|| CircuitError::DimMismatch("missing flat GAT parameter input".into()))?;
        if values.len() != FIRST_WEIGHT_INPUT + 1 {
            return Err(CircuitError::DimMismatch("canonical GAT requires one flat runtime parameter input".into(),));
        }
        Self::require_scale(
            flat,
            self.spec.scales.s_w_hidden[0][0],
            "flat GAT parameter input",
        )?;
        let expected_len = self.spec.model.flattened_params().len();
        if flat.len() != expected_len || flat.dims() != [expected_len] {
            return Err(CircuitError::DimMismatch(format!("flat GAT parameter length {} does not match canonical closure {}", flat.len(), expected_len)));
        }
        let mut at = 0usize;
        let mut out = Vec::new();
        for (layer_idx, layer) in self.spec.model.hidden_layers.iter().enumerate() {
            for (head_idx, _head) in layer.heads.iter().enumerate() {
                for (dims, scale) in [
                    (vec![layer.d_in, layer.d_head], self.spec.scales.s_w_hidden[layer_idx][head_idx],),(vec![layer.d_head], self.spec.scales.s_a_src_hidden[layer_idx][head_idx],),(vec![layer.d_head], self.spec.scales.s_a_dst_hidden[layer_idx][head_idx],),
                ] {
                    let len = dims.iter().product::<usize>();
                    let tensor = Self::flatten_slice(flat, at, at + len, &dims)?;
                    let mut tensor = tensor;
                    tensor.set_scale(scale);
                    out.push(tensor);
                    at += len;
                }
            }
        }
        for (head_idx, _head) in self.spec.model.output_heads.iter().enumerate() {
            for (dims, scale) in [
                (
                    vec![
                        self.spec.model.hidden_layers.last().unwrap().heads.len() * self.spec.model.hidden_layers.last().unwrap().d_head,
                        self.spec.model.classes,
                    ],
                    self.spec.scales.s_w_output[head_idx],
                ),
                (vec![self.spec.model.classes], self.spec.scales.s_a_src_output[head_idx],),(vec![self.spec.model.classes], self.spec.scales.s_a_dst_output[head_idx],),(vec![self.spec.model.classes], self.spec.scales.s_b_output[head_idx],),
            ] {
                let len = dims.iter().product::<usize>();
                let tensor = Self::flatten_slice(flat, at, at + len, &dims)?;
                let mut tensor = tensor;
                tensor.set_scale(scale);
                out.push(tensor);
                at += len;
            }
        }
        if at != flat.len() {
            return Err(CircuitError::DimMismatch("GAT flat parameter closure was not consumed exactly".into(),));
        }
        Ok(out)
    }

    /// Validate the real formal graph/tensor metadata before GraphCircuit
    /// allocates its expensive dummy layout.  This deliberately checks the
    /// adapter's rank-preserving broadcast plan, not just element counts.
    pub fn validate_pre_synthesis(
        spec: &CanonicalGat,
        input_shapes: &[Vec<usize>],
        output_shapes: &[Vec<usize>],
        param_shapes: &[Vec<usize>],
        input_scales: &[crate::Scale],
        param_scales: &[crate::Scale],
        output_scales: &[crate::Scale],
        inputs: &[Tensor<Fp>],
        params: &Tensor<Fp>,
        formal_input_scale: crate::Scale,
        formal_param_scale: crate::Scale,
    ) -> Result<(), String> {
        spec.validate_for_compile()?;
        let graph = &spec.graph;
        let public_shapes = vec![
            vec![graph.n_total, spec.feature_dim],
            vec![graph.n],
            vec![graph.e],
            vec![graph.e],
            vec![graph.g_batch + 1],
            vec![graph.g_batch + 1],
        ];
        if input_shapes != public_shapes.as_slice() || inputs.len() != public_shapes.len() {
            return Err(format!("formal GAT input shape metadata {:?} does not match {:?}", input_shapes, public_shapes));
        }
        let expected_input_scales = [spec.scales.s_input, 0, 0, 0, 0, 0];
        if input_scales != expected_input_scales.as_slice() || formal_input_scale != spec.scales.s_input {
            return Err("formal GAT input scale metadata disagrees with canonical scale chain".into());
        }
        for (index, (tensor, shape)) in inputs.iter().zip(&public_shapes).enumerate() {
            if tensor.dims() != shape.as_slice() || tensor.scale() != Some(expected_input_scales[index]) {
                return Err(format!("formal GAT input {} tensor metadata has shape {:?}, scale {:?}; expected {:?}, {}", index, tensor.dims(), tensor.scale(), shape, expected_input_scales[index]));
            }
        }
        let parameter_len = spec.model.flattened_params().len();
        let parameter_shape = vec![parameter_len];
        if parameter_len == 0
            || param_shapes != [parameter_shape.clone()]
            || param_scales != [formal_param_scale]
            || params.dims() != parameter_shape.as_slice()
            || params.scale() != Some(formal_param_scale)
            || formal_param_scale != spec.scales.s_w_hidden[0][0]
        {
            return Err("formal GAT parameter tensor shape/scale metadata is not canonical".into());
        }
        let output_shape = vec![graph.n, spec.model.classes];
        if output_shapes != [output_shape.clone()] || output_scales != [spec.scales.s_out] {
            return Err(format!("formal GAT output metadata {:?}/{:?} does not match {:?}/{}", output_shapes, output_scales, output_shape, spec.scales.s_out));
        }
        let formal_output_shape = output_shapes[0].clone();
        let formal_output_scale = output_scales[0];
        let mut output_scalar = Self::constant_i128(&[1], &[1]).map_err(|e| e.to_string())?;
        output_scalar.set_scale(formal_output_scale);
        let repeated = Self::repeated_scalar(&output_scalar, &formal_output_shape).map_err(|e| e.to_string())?;
        if repeated.dims() != formal_output_shape.as_slice() || repeated.scale() != formal_output_scale {
            return Err(format!("formal output scalar repeat produced shape {:?}, scale {}; expected {:?}, {}", repeated.dims(), repeated.scale(), formal_output_shape, formal_output_scale));
        }
        let mut broadcast = repeated.clone();
        broadcast.expand(&formal_output_shape).map_err(|e| e.to_string())?;
        let shape = |label: &str, actual: &[usize], expected: &[usize]| -> Result<(), String> {
            if actual != expected {
                return Err(format!("{} shape {:?} does not match {:?}", label, actual, expected));
            }
            Ok(())
        };
        let scalar = [1];
        let edge = [graph.e];
        let groups = graph.destination_groups()?;
        if groups.is_empty() {
            return Err("formal GAT has no destination groups".into());
        }
        let mut current_dim = spec.feature_dim;
        for (layer_idx, layer) in spec.model.hidden_layers.iter().enumerate() {
            shape("hidden input", &[graph.n, current_dim], &[graph.n, layer.d_in])?;
            for (head_idx, _) in layer.heads.iter().enumerate() {
                shape("hidden projection", &[layer.d_in, layer.d_head], &[layer.d_in, layer.d_head])?;
                shape("hidden projected", &[graph.n, layer.d_head], &[graph.n, layer.d_head])?;
                shape("hidden gathered", &edge, &[graph.e])?;
                shape("hidden attention vectors", &[layer.d_head], &[layer.d_head])?;
                shape("hidden scores", &edge, &[graph.e])?;
                for (_, start, end) in &groups {
                    let group = [end - start];
                    shape("group score", &group, &[end - start])?;
                    shape("group scalar repeat", &group, &[end - start])?;
                    shape("group source values", &[end - start, layer.d_head], &[end - start, layer.d_head])?;
                    shape("group tiled values", &[end - start, layer.d_head], &[end - start, layer.d_head])?;
                    shape("group channel values", &group, &[end - start])?;
                    shape("group numerator", &scalar, &[1])?;
                }
                let _ = (layer_idx, head_idx);
            }
            current_dim = layer.heads.len().checked_mul(layer.d_head).ok_or_else(|| "hidden concat shape overflow".to_string())?;
            shape("hidden head concat", &[graph.n, current_dim], &[graph.n, current_dim])?;
        }
        for (head_idx, _) in spec.model.output_heads.iter().enumerate() {
            shape("output projection", &[current_dim, spec.model.classes], &[current_dim, spec.model.classes])?;
            shape("output projected", &[graph.n, spec.model.classes], &[graph.n, spec.model.classes])?;
            shape("output attention vectors", &[spec.model.classes], &[spec.model.classes])?;
            shape("output scores", &edge, &[graph.e])?;
            for (_, start, end) in &groups {
                let group = [end - start];
                shape("output group scalar repeat", &group, &[end - start])?;
                shape("output group source values", &[end - start, spec.model.classes], &[end - start, spec.model.classes])?;
                shape("output group numerator", &scalar, &[1])?;
            }
            shape("output alignment dividend", &output_shape, &[graph.n, spec.model.classes])?;
            shape("output alignment scalar repeat", &output_shape, &[graph.n, spec.model.classes])?;
            shape("output head", &[graph.n, spec.model.classes], &[graph.n, spec.model.classes])?;
            let _ = head_idx;
        }
        shape("head average scalar repeat", &output_shape, &[graph.n, spec.model.classes])?;
        shape("head average", &output_shape, &output_shape)?;
        shape("output reshape", &output_shape, &[graph.n, spec.model.classes])?;
        Ok(())
    }

    fn repeated_scalar(scalar: &ValTensor<Fp>, dims: &[usize]) -> Result<ValTensor<Fp>, CircuitError> {
        if scalar.len() != 1 {
            return Err(CircuitError::DimMismatch("GAT repeat source must be scalar".into()));
        }
        let value = scalar.get_inner_tensor()?.first()?.get_scalar();
        let len = dims.iter().product::<usize>();
        let mut repeated: ValTensor<Fp> = Tensor::new(Some(&vec![value; len]), dims)?.into();
        repeated.set_scale(scalar.scale());
        Ok(repeated)
    }

    fn flatten_slice(value: &ValTensor<Fp>, start: usize, end: usize, dims: &[usize],) -> Result<ValTensor<Fp>, CircuitError> {
        let mut flat = value.clone();
        flat.flatten();
        let mut slice = flat.get_slice(&[start..end])?;
        slice.reshape(dims)?;
        Ok(slice)
    }

    fn append_tensors(tensors: &[ValTensor<Fp>], dims: &[usize],) -> Result<ValTensor<Fp>, CircuitError> {
        let mut vals = Vec::new();
        for tensor in tensors {
            vals.extend(tensor.get_inner_tensor()?.iter().cloned());
        }
        Ok(Tensor::new(Some(&vals), dims)?.into())
    }

    /// Interleave hidden-head outputs into the canonical `[node, head*width]`
    /// layout.  A plain concatenation would produce head-major storage and
    /// silently change the next layer's projection relation.
    fn interleave_heads(tensors: &[ValTensor<Fp>], nodes: usize, channels: usize,) -> Result<ValTensor<Fp>, CircuitError> {
        if tensors.is_empty() {
            return Err(CircuitError::UnsupportedOp);
        }
        if tensors.iter().any(|tensor| tensor.len() != nodes * channels || tensor.dims() != [nodes, channels]) {
            return Err(CircuitError::DimMismatch("hidden head output shape does not match graph".into(),));
        }
        let mut vals = Vec::with_capacity(nodes * tensors.len() * channels);
        for node in 0..nodes {
            for tensor in tensors {
                let inner = tensor.get_inner_tensor()?;
                let start = node * channels;
                vals.extend(inner[start..start + channels].iter().cloned());
            }
        }
        Ok(Tensor::new(Some(&vals), &[nodes, tensors.len() * channels])?.into())
    }

    fn tile_rows(value: &ValTensor<Fp>, rows: usize, cols: usize,) -> Result<ValTensor<Fp>, CircuitError> {
        if value.len() != rows || value.dims() != [rows] {
            return Err(CircuitError::DimMismatch("cannot tile tensor with incompatible rows".into(),));
        }
        let mut vals = Vec::with_capacity(rows * cols);
        for item in value.get_inner_tensor()?.iter() {
            vals.extend(std::iter::repeat(item.clone()).take(cols));
        }
        Ok(Tensor::new(Some(&vals), &[rows, cols])?.into())
    }

    fn inverse_exact(&self, config: &BaseConfig<Fp>, region: &mut RegionCtx<Fp>, denominator: &ValTensor<Fp>,) -> Result<ValTensor<Fp>, CircuitError> {
        let inv = if denominator.any_unknowns()? {
            Self::unknown_tensor(denominator.dims())?
        } else {
            let d = denominator.int_evals()?[0];
            if d == 0 {
                return Err(CircuitError::UnsupportedOp);
            }
            let value = ValType::Value(Value::known(Option::from(integer_rep_to_felt::<Fp>(d).invert()).ok_or(CircuitError::UnsupportedOp)?,));
            Tensor::new(Some(&[value]), denominator.dims())?.into()
        };
        let product = layouts::pairwise(config, region, &[denominator, &inv], BaseOp::Mult)?;
        let one = Self::constant_i128(&[1], denominator.dims())?;
        layouts::enforce_equality(config, region, &[&product, &one])?;
        Ok(inv)
    }

    fn signed_round_div_layout(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        dividend: &ValTensor<Fp>,
        divisor: &ValTensor<Fp>,
        bounds: Option<&BoundFamily>,
    ) -> Result<ValTensor<Fp>, CircuitError> {
        self.signed_round_div_layout_mode(config, region, dividend, divisor, self.rounding, bounds)
    }

    fn signed_round_div_layout_mode(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        dividend: &ValTensor<Fp>,
        divisor: &ValTensor<Fp>,
        rounding: RoundingMode,
        bounds: Option<&BoundFamily>,
    ) -> Result<ValTensor<Fp>, CircuitError> {
        if divisor.len() != 1 {
            return Err(CircuitError::DimMismatch("GAT divisor must be scalar".into(),));
        }
        let (q, q_floor, rem, abs, sigma, zero, rounded_up) = if dividend.any_unknowns()? || divisor.any_unknowns()? {
            let dims = dividend.dims();
            (
                Self::unknown_tensor(dims)?,
                Self::unknown_tensor(dims)?,
                Self::unknown_tensor(dims)?,
                Self::unknown_tensor(dims)?,
                Self::unknown_tensor(dims)?,
                Self::unknown_tensor(dims)?,
                Self::unknown_tensor(dims)?,
            )
        } else {
            let d = divisor.int_evals()?[0];
            if d <= 0 {
                return Err(CircuitError::UnsupportedOp);
            }
            let mut qv = Vec::new();
            let mut q_floor_v = Vec::new();
            let mut rv = Vec::new();
            let mut av = Vec::new();
            let mut sv = Vec::new();
            let mut zv = Vec::new();
            let mut rounded_up_v = Vec::new();
            for x in dividend.int_evals()?.iter().copied() {
                let signed = super::relation::evaluate_signed_division(x, d, rounding).map_err(|_| CircuitError::UnsupportedOp)?;
                qv.push(signed.quotient);
                q_floor_v.push(signed.quotient_floor);
                rv.push(signed.remainder);
                av.push(signed.abs_dividend);
                sv.push(if signed.sigma { 1 } else { 0 });
                zv.push(if signed.zero { 1 } else { 0 });
                rounded_up_v.push(if signed.rounded_up { 1 } else { 0 });
            }
            (
                Self::value_i128(&qv, dividend.dims())?,
                Self::value_i128(&q_floor_v, dividend.dims())?,
                Self::value_i128(&rv, dividend.dims())?,
                Self::value_i128(&av, dividend.dims())?,
                Self::value_i128(&sv, dividend.dims())?,
                Self::value_i128(&zv, dividend.dims())?,
                Self::value_i128(&rounded_up_v, dividend.dims())?,
            )
        };
        layouts::boolean_identity(config, region, &[&sigma], true)?;
        layouts::boolean_identity(config, region, &[&zero], true)?;
        layouts::boolean_identity(config, region, &[&rounded_up], true)?;
        let two = Self::constant_i128(&vec![2; dividend.len()], dividend.dims())?;
        let two_sigma = layouts::pairwise(config, region, &[&sigma, &two], BaseOp::Mult)?;
        let one = Self::constant_i128(&vec![1; dividend.len()], dividend.dims())?;
        let sign_factor = layouts::pairwise(config, region, &[&one, &two_sigma], BaseOp::Sub)?;
        let signed_abs = layouts::pairwise(config, region, &[&sign_factor, &abs], BaseOp::Mult)?;
        layouts::enforce_equality(config, region, &[dividend, &signed_abs])?;
        let q_abs = layouts::pairwise(config, region, &[&q_floor, &rounded_up], BaseOp::Add)?;
        let signed_q = layouts::pairwise(config, region, &[&sign_factor, &q_abs], BaseOp::Mult)?;
        layouts::enforce_equality(config, region, &[&q, &signed_q])?;
        let d_repeated = Self::repeated_scalar(divisor, dividend.dims())?;
        let max_divisor = bounds.map(|family| family.max_divisor).unwrap_or(self.spec.ledger.max_divisor);
        let max_aux_abs = bounds.map(|family| family.max_aux_abs).unwrap_or(self.spec.ledger.max_aux_abs);
        let max_dividend_abs = bounds.map(|family| family.max_dividend_abs).unwrap_or(self.spec.ledger.max_dividend_abs);
        layouts::range_check(config, region, &[&q_abs], &(0, max_dividend_abs))?;
        layouts::range_check(config, region, &[&q], &(-max_dividend_abs, max_dividend_abs),)?;
        layouts::range_check(config, region, &[&sign_factor], &(-1, 1))?;
        layouts::range_check(config, region, &[divisor], &(1, max_divisor))?;
        let d_inv_scalar = self.inverse_exact(config, region, divisor)?;
        let d_inv = Self::repeated_scalar(&d_inv_scalar, dividend.dims())?;
        let d_product = layouts::pairwise(config, region, &[&d_repeated, &d_inv], BaseOp::Mult)?;
        let d_one = Self::constant_i128(&vec![1; dividend.len()], dividend.dims())?;
        layouts::enforce_equality(config, region, &[&d_product, &d_one])?;
        let abs_inv = if abs.any_unknowns()? {
            Self::unknown_tensor(abs.dims())?
        } else {
            let vals = abs
                .int_evals()?
                .into_iter()
                .map(|x| -> Result<ValType<Fp>, CircuitError> {
                    if x == 0 {
                        Ok(ValType::Value(Value::known(Fp::ZERO)))
                    } else {
                        Ok(ValType::Value(Value::known(Option::from(integer_rep_to_felt::<Fp>(x).invert()).ok_or(CircuitError::UnsupportedOp)?,))) } })
                .collect::<Result<Vec<_>, _>>()?;
            Tensor::new(Some(&vals), abs.dims())?.into()
        };
        let abs_product = layouts::pairwise(config, region, &[&abs, &abs_inv], BaseOp::Mult)?;
        let one_minus_zero = layouts::pairwise(config, region, &[&one, &zero], BaseOp::Sub)?;
        layouts::enforce_equality(config, region, &[&abs_product, &one_minus_zero])?;
        let zero_from_abs = layouts::equals_zero(config, region, &[&abs])?;
        layouts::enforce_equality(config, region, &[&zero, &zero_from_abs])?;
        let zero_abs = layouts::pairwise(config, region, &[&zero, &abs], BaseOp::Mult)?;
        let zero_sigma = layouts::pairwise(config, region, &[&zero, &sigma], BaseOp::Mult)?;
        let zero_inv = layouts::pairwise(config, region, &[&zero, &abs_inv], BaseOp::Mult)?;
        let zero_q = layouts::pairwise(config, region, &[&zero, &q_abs], BaseOp::Mult)?;
        let zeros = Self::constant_i128(&vec![0; dividend.len()], dividend.dims())?;
        layouts::enforce_equality(config, region, &[&zero_abs, &zeros])?;
        layouts::enforce_equality(config, region, &[&zero_sigma, &zeros])?;
        layouts::enforce_equality(config, region, &[&zero_inv, &zeros])?;
        layouts::enforce_equality(config, region, &[&zero_q, &zeros])?;
        let d_minus_one = layouts::pairwise(config, region, &[&d_repeated, &one], BaseOp::Sub)?;
        let delta_vals = if rem.any_unknowns()? {
            Self::unknown_tensor(rem.dims())?
        } else {
            let d = divisor.int_evals()?[0];
            let values = rem.int_evals()?;
            let deltas = values
                .iter()
                .map(|r| match rounding {
                    RoundingMode::TowardZero => d.checked_sub(1).and_then(|v| v.checked_sub(*r)),
                    RoundingMode::NearestTiesAway => d.checked_mul(2).and_then(|v| v.checked_sub(1)).and_then(|v| v.checked_sub(*r)),
                })
                .collect::<Option<Vec<_>>>()
                .ok_or(CircuitError::UnsupportedOp)?;
            Self::value_i128(&deltas, rem.dims())?
        };

        match rounding {
            RoundingMode::TowardZero => {
                let dq_floor = layouts::pairwise(config, region, &[&d_repeated, &q_floor], BaseOp::Mult)?;
                let abs_reconstructed = layouts::pairwise(config, region, &[&dq_floor, &rem], BaseOp::Add)?;
                layouts::enforce_equality(config, region, &[&abs, &abs_reconstructed])?;
                let r_delta = layouts::pairwise(config, region, &[&rem, &delta_vals], BaseOp::Add)?;
                layouts::enforce_equality(config, region, &[&r_delta, &d_minus_one])?;
            }
            RoundingMode::NearestTiesAway => {
                let two_abs = layouts::pairwise(config, region, &[&two, &abs], BaseOp::Mult)?;
                let lhs = layouts::pairwise(config, region, &[&two_abs, &d_repeated], BaseOp::Add)?;
                let dq_abs = layouts::pairwise(config, region, &[&d_repeated, &q_abs], BaseOp::Mult)?;
                let two_dq_abs = layouts::pairwise(config, region, &[&two, &dq_abs], BaseOp::Mult)?;
                let rhs = layouts::pairwise(config, region, &[&two_dq_abs, &rem], BaseOp::Add)?;
                layouts::enforce_equality(config, region, &[&lhs, &rhs])?;

                let two_d = layouts::pairwise(config, region, &[&two, &d_repeated], BaseOp::Mult)?;
                let two_d_minus_one = layouts::pairwise(config, region, &[&two_d, &one], BaseOp::Sub)?;
                let r_delta = layouts::pairwise(config, region, &[&rem, &delta_vals], BaseOp::Add)?;
                layouts::enforce_equality(config, region, &[&r_delta, &two_d_minus_one])?;

                let dq_floor = layouts::pairwise(config, region, &[&d_repeated, &q_floor], BaseOp::Mult)?;
                let floor_rem = layouts::pairwise(config, region, &[&abs, &dq_floor], BaseOp::Sub)?;
                let two_floor_rem = layouts::pairwise(config, region, &[&two, &floor_rem], BaseOp::Mult)?;
                let low = layouts::pairwise(config, region, &[&d_minus_one, &two_floor_rem], BaseOp::Sub,)?;
                let high = layouts::pairwise(config, region, &[&two_floor_rem, &d_repeated], BaseOp::Sub)?;
                let one_minus_round = layouts::pairwise(config, region, &[&one, &rounded_up], BaseOp::Sub)?;
                let low_selected = layouts::pairwise(config, region, &[&one_minus_round, &low], BaseOp::Mult)?;
                let high_selected = layouts::pairwise(config, region, &[&rounded_up, &high], BaseOp::Mult)?;
                let selected = layouts::pairwise(config, region, &[&low_selected, &high_selected], BaseOp::Add,)?;
                layouts::range_check(config, region, &[&floor_rem], &(0, max_aux_abs))?;
                layouts::range_check(config, region, &[&selected], &(0, max_aux_abs))?;
            }
        }

        layouts::range_check(config, region, &[&rem], &(0, max_aux_abs))?;
        layouts::range_check(config, region, &[&delta_vals], &(0, max_aux_abs))?;
        layouts::range_check(config, region, &[&q_floor], &(0, max_dividend_abs))?;
        match rounding {
            RoundingMode::TowardZero => {
                let zero_round = Self::constant_i128(&vec![0; dividend.len()], dividend.dims())?;
                layouts::enforce_equality(config, region, &[&rounded_up, &zero_round])?;
            }
            RoundingMode::NearestTiesAway => {}
        }
        layouts::range_check(config, region, &[&abs], &(0, max_dividend_abs))?;
        layouts::range_check(config, region, &[&rem], &(0, max_aux_abs))?;
        Ok(q)
    }

    /// Prove a group maximum using ezkl's native arithmetic/lookup pieces.
    ///
    /// The claimed maximum is a witness value, `w=x-(m-1)` is looked up in
    /// the backend-only exact ReLU table, the outputs are boolean, and a
    /// second ReLU lookup proves that at least one entry reaches `m`.  This
    /// keeps max sound under ties and avoids treating a host/native max as a
    /// proof relation.
    fn exact_group_max(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        values: &ValTensor<Fp>,
        bounds: &BoundFamily,
    ) -> Result<ValTensor<Fp>, CircuitError> {
        let claimed = if values.any_unknowns()? {
            Self::unknown_tensor(&[1])?
        } else {
            let max = values.int_evals()?.into_iter().max().ok_or(CircuitError::UnsupportedOp)?;
            Self::value_i128(&[max], &[1])?
        };
        layouts::range_check(config, region, &[&claimed], &(-bounds.max_table_output_abs, bounds.max_table_output_abs),)?;
        let claimed_repeated = Self::repeated_scalar(&claimed, values.dims())?;
        let one = Self::constant_i128(&vec![1; values.len()], values.dims())?;
        let max_minus_one = layouts::pairwise(config, region, &[&claimed_repeated, &one], BaseOp::Sub)?;
        let w = layouts::pairwise(config, region, &[values, &max_minus_one], BaseOp::Sub)?;
        let y = self.exact_lookup(config, region, &w, &self.spec.tables.max_relu)?;
        let y = layouts::boolean_identity(config, region, &[&y], true)?;
        let y_sum = layouts::sum(config, region, &[&y])?;
        let one_scalar = Self::constant_i128(&[1], &[1])?;
        let deficit = layouts::pairwise(config, region, &[&one_scalar, &y_sum], BaseOp::Sub)?;
        let zero_or_one = self.exact_lookup(config, region, &deficit, &self.spec.tables.max_relu)?;
        let zero = Self::constant_i128(&[0], &[1])?;
        layouts::enforce_equality(config, region, &[&zero_or_one, &zero])?;
        Ok(claimed)
    }

    fn aggregate_head(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        projected: &ValTensor<Fp>,
        source: &ValTensor<Fp>,
        dst: &ValTensor<Fp>,
        attention_src: &ValTensor<Fp>,
        attention_dst: &ValTensor<Fp>,
        channels: usize,
        use_elu: bool,
        bounds: &BoundFamily,
        recovery_scale: i32,
        elu_output_scale: i32,
        projected_scale: i32,
        attention_src_scale: i32,
        attention_dst_scale: i32,
    ) -> Result<ValTensor<Fp>, CircuitError> {
        let graph = &self.spec.graph;
        Self::require_dims(projected, &[graph.n, channels], "projected tensor")?;
        Self::require_dims(source, &[graph.e], "source edge indices")?;
        Self::require_dims(dst, &[graph.e], "destination edge indices")?;
        Self::require_dims(attention_src, &[channels], "source attention vector")?;
        Self::require_dims(attention_dst, &[channels], "destination attention vector")?;
        Self::require_scale(projected, projected_scale, "projected tensor")?;
        Self::require_scale(attention_src, attention_src_scale, "source attention vector",)?;
        Self::require_scale(attention_dst, attention_dst_scale, "destination attention vector",)?;
        let mut src_values = layouts::gather(config, region, &[projected, source], 0)?;
        let mut dst_values = layouts::gather(config, region, &[projected, dst], 0)?;
        src_values.set_scale(projected_scale);
        dst_values.set_scale(projected_scale);
        let mut src_score = layouts::einsum(config, region, &[&src_values, attention_src], "ij,j->i")?;
        let mut dst_score = layouts::einsum(config, region, &[&dst_values, attention_dst], "ij,j->i")?;
        src_score.set_scale(self.spec.scales.s_delta);
        dst_score.set_scale(self.spec.scales.s_delta);
        let mut score = layouts::pairwise(config, region, &[&src_score, &dst_score], BaseOp::Add)?;
        score.set_scale(self.spec.scales.s_delta);
        layouts::range_check(config, region, &[&score], &(-bounds.max_score_abs, bounds.max_score_abs),)?;
        let mut score = self.exact_lookup(config, region, &score, &self.spec.tables.lrelu)?;
        score.set_scale(self.spec.scales.s_delta);
        layouts::range_check(config, region, &[&score], &(-bounds.max_table_output_abs, bounds.max_table_output_abs),)?;
        let mut node_values = vec![None; graph.n * channels];
        for (node, start, end) in graph.destination_groups().map_err(|_| CircuitError::UnsupportedOp)? {
            let group_score = Self::flatten_slice(&score, start, end, &[end - start])?;
            let max = self.exact_group_max(config, region, &group_score, bounds)?;
            let max_repeated = Self::repeated_scalar(&max, group_score.dims())?;
            let score_group = Self::flatten_slice(&score, start, end, &[end - start])?;
            let mut delta = layouts::pairwise(config, region, &[&max_repeated, &score_group], BaseOp::Sub)?;
            delta.set_scale(self.spec.scales.s_delta);
            layouts::range_check(config, region, &[&delta], &(0, bounds.max_delta))?;
            let mut exp = self.exact_lookup(config, region, &delta, &self.spec.tables.exp)?;
            exp.set_scale(self.spec.scales.s_exp);
            layouts::range_check(config, region, &[&exp], &(1, bounds.max_exp))?;
            let mut denominator = layouts::sum(config, region, &[&exp])?;
            denominator.set_scale(self.spec.scales.s_exp);
            layouts::range_check(config, region, &[&denominator], &(1, bounds.max_sum))?;
            let inverse = self.inverse_exact(config, region, &denominator)?;
            let alpha = layouts::pairwise(config, region, &[&exp, &Self::repeated_scalar(&inverse, exp.dims())?], BaseOp::Mult,)?;
            let src_group = Self::flatten_slice(&src_values, start * channels, end * channels, &[end - start, channels],)?;
            let exp_group = Self::tile_rows(&exp, end - start, channels)?;
            let alpha_group = Self::tile_rows(&alpha, end - start, channels)?;
            let mut channel_outputs = Vec::new();
            for channel in 0..channels {
                let mut vals = Vec::new();
                let mut alpha_vals = Vec::new();
                let mut exp_vals = Vec::new();
                for row in 0..(end - start) {
                    vals.push(src_group.get_inner_tensor()?.get(&[row, channel]).clone());
                    alpha_vals.push(alpha_group.get_inner_tensor()?.get(&[row, channel]).clone());
                    exp_vals.push(exp_group.get_inner_tensor()?.get(&[row, channel]).clone());
                }
                let values = Tensor::new(Some(&vals), &[end - start])?.into();
                let alpha_values: ValTensor<Fp> = Tensor::new(Some(&alpha_vals), &[end - start])?.into();
                let exp_values: ValTensor<Fp> = Tensor::new(Some(&exp_vals), &[end - start])?.into();
                let field_product = layouts::pairwise(config, region, &[&alpha_values, &values], BaseOp::Mult)?;
                let field_sum = layouts::sum(config, region, &[&field_product])?;
                let numerator_product = layouts::pairwise(config, region, &[&exp_values, &values], BaseOp::Mult)?;
                let mut numerator = layouts::sum(config, region, &[&numerator_product])?;
                numerator.set_scale(self.spec.scales.s_exp.checked_add(projected_scale).ok_or(CircuitError::UnsupportedOp)?,);
                let field_from_numerator = layouts::pairwise(config, region, &[&numerator, &inverse], BaseOp::Mult)?;
                layouts::enforce_equality(config, region, &[&field_sum, &field_from_numerator])?;
                layouts::range_check(config, region, &[&numerator], &(-bounds.max_aggregation_num, bounds.max_aggregation_num),)?;
                let recovery_multiplier = scale_exponent_to_multiplier(recovery_scale)
                    .map_err(|_| CircuitError::UnsupportedOp)
                    .and_then(|value| { i128::try_from(value).map_err(|_| CircuitError::UnsupportedOp) })?;
                let denominator_multiplier = scale_exponent_to_multiplier(projected_scale)
                    .map_err(|_| CircuitError::UnsupportedOp)
                    .and_then(|value| { i128::try_from(value).map_err(|_| CircuitError::UnsupportedOp) })?;
                let numerator_scale = Self::constant_i128(&vec![recovery_multiplier; numerator.len()], numerator.dims(),)?;
                let scaled_numerator = layouts::pairwise(config, region, &[&numerator, &numerator_scale], BaseOp::Mult,)?;
                let denominator_scale = Self::constant_i128(&[denominator_multiplier], &[1])?;
                let scaled_denominator = layouts::pairwise(config, region, &[&denominator, &denominator_scale], BaseOp::Mult,)?;
                let q = self.signed_round_div_layout(config, region, &scaled_numerator, &scaled_denominator, Some(bounds),)?;
                let q = if use_elu {
                    let mut q = q;
                    q.set_scale(recovery_scale);
                    let mut elu = self.exact_lookup(config, region, &q, &self.spec.tables.elu)?;
                    elu.set_scale(elu_output_scale);
                    elu
                } else {
                    let mut q = q;
                    q.set_scale(self.spec.scales.s_out_pre);
                    q
                };
                channel_outputs.push(q);
            }
            for channel in 0..channels {
                node_values[node * channels + channel] = Some(channel_outputs[channel].clone());
            }
        }
        let mut out = Vec::with_capacity(graph.n * channels);
        for value in node_values {
            out.push(value.ok_or(CircuitError::UnsupportedOp)?);
        }
        Self::append_tensors(&out, &[graph.n, channels])
    }
}

impl Op<Fp> for GatOp {
    fn as_string(&self) -> String {
        format!("CanonicalGAT({})", self.spec.model.model_arch_id)
    }

    fn layout(&self, config: &mut BaseConfig<Fp>, region: &mut RegionCtx<Fp>, values: &[&ValTensor<Fp>],) -> Result<Option<ValTensor<Fp>>, CircuitError> {
        if values.len() < FIRST_WEIGHT_INPUT {
            return Err(CircuitError::DimMismatch("GAT requires six public inputs".into(),));
        }
        self.spec.validate_for_compile().map_err(|e| CircuitError::DimMismatch(format!("invalid canonical GAT: {}", e)))?;
        let graph = &self.spec.graph;
        for (value, dims, label) in [
            (values[FEATURE_INPUT], &[graph.n_total, self.spec.feature_dim][..], "feature input"),
            (values[NODE_ID_INPUT], &[graph.n][..], "node ids"),
            (values[SRC_INPUT], &[graph.e][..], "source edges"),
            (values[DST_INPUT], &[graph.e][..], "destination edges"),
            (values[NODE_PTR_INPUT], &[graph.g_batch + 1][..], "node pointers"),
            (values[EDGE_PTR_INPUT], &[graph.g_batch + 1][..], "edge pointers"),
        ] {
            Self::require_dims(value, dims, label)?;
        }
        Self::require_scale(values[FEATURE_INPUT], self.spec.scales.s_input, "feature input",)?;
        let mut features = layouts::identity(config, region, &[values[FEATURE_INPUT]])?;
        features.set_scale(self.spec.scales.s_input);
        let mut node_ids = self.bind_topology(config, region, values[NODE_ID_INPUT], NODE_ID_INPUT)?;
        let mut src = self.bind_topology(config, region, values[SRC_INPUT], SRC_INPUT)?;
        let mut dst = self.bind_topology(config, region, values[DST_INPUT], DST_INPUT)?;
        node_ids.set_scale(0);
        src.set_scale(0);
        dst.set_scale(0);
        let _node_ptr = self.bind_topology(config, region, values[NODE_PTR_INPUT], NODE_PTR_INPUT)?;
        let _edge_ptr = self.bind_topology(config, region, values[EDGE_PTR_INPUT], EDGE_PTR_INPUT)?;
        let weights = self.weight_tensors(values)?;
        let mut weight_idx = 0;
        let mut current = layouts::gather(config, region, &[&features, &node_ids], 0)?;
        Self::require_dims(&current, &[graph.n, self.spec.feature_dim], "gathered features")?;
        current.set_scale(self.spec.scales.s_h[0]);
        let mut current_dim = self.spec.feature_dim;
        for (layer_idx, layer) in self.spec.model.hidden_layers.iter().enumerate() {
            let mut heads = Vec::new();
            for (head_idx, _head) in layer.heads.iter().enumerate() {
                let bounds = self.bound_family(layer_idx, head_idx, false)?;
                let projection = layouts::identity(config, region, &[&weights[weight_idx]])?;
                let attention_src = layouts::identity(config, region, &[&weights[weight_idx + 1]])?;
                let attention_dst = layouts::identity(config, region, &[&weights[weight_idx + 2]])?;
                weight_idx += 3;
                Self::require_scale(&projection, self.spec.scales.s_w_hidden[layer_idx][head_idx], "hidden projection weight",)?;
                Self::require_scale(&attention_src, self.spec.scales.s_a_src_hidden[layer_idx][head_idx], "hidden source attention weight",)?;
                Self::require_scale(&attention_dst, self.spec.scales.s_a_dst_hidden[layer_idx][head_idx], "hidden destination attention weight",)?;
                if projection.dims() != [current_dim, layer.d_head] {
                    return Err(CircuitError::DimMismatch("GAT projection shape".into()));
                }
                let mut projected = layouts::einsum(config, region, &[&current, &projection], "ij,jk->ik")?;
                projected.set_scale(self.spec.scales.s_h_prime[layer_idx][head_idx]);
                let aggregated = self.aggregate_head(
                    config,
                    region,
                    &projected,
                    &src,
                    &dst,
                    &attention_src,
                    &attention_dst,
                    layer.d_head,
                    true,
                    bounds,
                    self.spec.scales.s_agg_pre[layer_idx],
                    self.spec.scales.s_agg[layer_idx],
                    self.spec.scales.s_h_prime[layer_idx][head_idx],
                    self.spec.scales.s_a_src_hidden[layer_idx][head_idx],
                    self.spec.scales.s_a_dst_hidden[layer_idx][head_idx],
                )?;
                heads.push(aggregated);
            }
            current = Self::interleave_heads(&heads, self.spec.graph.n, layer.d_head)?;
            current.set_scale(self.spec.scales.s_h[layer_idx + 1]);
            current_dim = layer.heads.len() * layer.d_head;
        }
        let mut output_heads = Vec::new();
        for (head_idx, _head) in self.spec.model.output_heads.iter().enumerate() {
            let bounds = self.bound_family(usize::MAX, head_idx, true)?;
            let projection = layouts::identity(config, region, &[&weights[weight_idx]])?;
            let attention_src = layouts::identity(config, region, &[&weights[weight_idx + 1]])?;
            let attention_dst = layouts::identity(config, region, &[&weights[weight_idx + 2]])?;
            let bias = layouts::identity(config, region, &[&weights[weight_idx + 3]])?;
            weight_idx += 4;
            Self::require_scale(&projection, self.spec.scales.s_w_output[head_idx], "output projection weight",)?;
            Self::require_scale(&attention_src, self.spec.scales.s_a_src_output[head_idx], "output source attention weight",)?;
            Self::require_scale(&attention_dst, self.spec.scales.s_a_dst_output[head_idx], "output destination attention weight",)?;
            Self::require_scale(&bias, self.spec.scales.s_b_output[head_idx], "output bias")?;
            let mut projected = layouts::einsum(config, region, &[&current, &projection], "ij,jk->ik")?;
            projected.set_scale(self.spec.scales.s_y_prime[head_idx]);
            let aggregated = self.aggregate_head(
                config,
                region,
                &projected,
                &src,
                &dst,
                &attention_src,
                &attention_dst,
                self.spec.model.classes,
                false,
                bounds,
                self.spec.scales.s_out_pre,
                self.spec.scales.s_out_pre,
                self.spec.scales.s_y_prime[head_idx],
                self.spec.scales.s_a_src_output[head_idx],
                self.spec.scales.s_a_dst_output[head_idx],
            )?;
            let out_head_multiplier = scale_exponent_to_multiplier(self.spec.scales.s_out_head).map_err(|_| CircuitError::UnsupportedOp)?;
            let out_head_multiplier = i128::try_from(out_head_multiplier).map_err(|_| CircuitError::UnsupportedOp)?;
            let out_pre_multiplier = scale_exponent_to_multiplier(self.spec.scales.s_out_pre).map_err(|_| CircuitError::UnsupportedOp)?;
            let out_pre_multiplier = i128::try_from(out_pre_multiplier).map_err(|_| CircuitError::UnsupportedOp)?;
            let scale_const = Self::constant_i128(&vec![out_head_multiplier; aggregated.len()], aggregated.dims(),)?;
            let aligned_dividend = layouts::pairwise(config, region, &[&aggregated, &scale_const], BaseOp::Mult)?;
            let divisor = Self::constant_i128(&[out_pre_multiplier], &[1])?;
            let mut aligned = self.signed_round_div_layout(config, region, &aligned_dividend, &divisor, Some(bounds),)?;
            aligned.set_scale(self.spec.scales.s_out_head);
            let mut bias_repeated =
                Self::append_tensors(&(0..self.spec.graph.n).map(|_| bias.clone()).collect::<Vec<_>>(), &[self.spec.graph.n, self.spec.model.classes],)?;
            bias_repeated.set_scale(self.spec.scales.s_b_output[head_idx]);
            let mut head_output = layouts::pairwise(config, region, &[&aligned, &bias_repeated], BaseOp::Add)?;
            head_output.set_scale(self.spec.scales.s_out_head);
            output_heads.push(head_output);
        }
        let mut output = output_heads.first().cloned().ok_or(CircuitError::UnsupportedOp)?;
        for head in output_heads.iter().skip(1) {
            output = layouts::pairwise(config, region, &[&output, head], BaseOp::Add)?;
            output.set_scale(self.spec.scales.s_out_head);
        }
        output.set_scale(self.spec.scales.s_out_head);
        let count = Self::constant_i128(&[i128::try_from(self.spec.model.output_heads.len()).map_err(|_| CircuitError::UnsupportedOp)?], &[1],)?;
        output = self.signed_round_div_layout_mode(config, region, &output, &count, RoundingMode::NearestTiesAway, None,)?;
        output.set_scale(self.spec.scales.s_out);
        Self::require_dims(&output, &[graph.n, self.spec.model.classes], "final GAT output")?;
        output.reshape(&[graph.n, self.spec.model.classes])?;
        Ok(Some(output))
    }

    fn out_scale(&self, _in_scales: Vec<crate::Scale>) -> Result<crate::Scale, CircuitError> {
        Ok(self.spec.scales.s_out)
    }

    fn clone_dyn(&self) -> Box<dyn Op<Fp>> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
