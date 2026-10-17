use ff::{BatchInvert, Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use group::Curve;
use instant::Instant;
use rand_core::RngCore;
use rustc_hash::FxBuildHasher;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::collections::BTreeSet;
use std::io::{self, BufRead, Read};
use std::iter;
use std::ops::RangeTo;
use std::path::PathBuf;

use super::{
    circuit::{
        sealed::{self},
        Advice, Any, Assignment, Challenge, Circuit, Column, ConstraintSystem, Fixed, FloorPlanner,
        Instance, Selector,
    },
    permutation, shuffle, vanishing, ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX,
    ChallengeY, Error, ProvingKey,
};
#[cfg(feature = "mv-lookup")]
use maybe_rayon::iter::{IndexedParallelIterator, ParallelIterator};

#[cfg(not(feature = "mv-lookup"))]
use super::lookup;
#[cfg(feature = "mv-lookup")]
use super::mv_lookup as lookup;

#[cfg(feature = "mv-lookup")]
use maybe_rayon::iter::{IntoParallelIterator, IntoParallelRefIterator};

use crate::transcript::{EncodedChallenge, TranscriptWrite};
use crate::{
    arithmetic::{eval_polynomial, CurveAffine},
    circuit::Value,
    plonk::Assigned,
    poly::{
        commitment::{Blind, CommitmentScheme, Params, Prover},
        Basis, Coeff, LagrangeCoeff, Polynomial, ProverQuery,
    },
};

#[cfg(feature = "mv-lookup")]
/// Returns the native in-memory sizes used to account prepared lookup state.
pub fn native_mv_lookup_resource_layout<C: crate::arithmetic::CurveAffine>() -> (usize, usize, usize) {
    (
        std::mem::size_of::<lookup::prover::Prepared<C>>(),
        std::mem::size_of::<lookup::prover::PreparedTable<C>>(),
        std::mem::size_of::<lookup::prover::Committed<C>>(),
    )
}

#[cfg(feature = "mv-lookup")]
/// Returns the shallow `size_of` values for the compact table, packed ID tape,
/// and lookup polynomial source, in that order; backing allocations are excluded.
pub fn native_mv_lookup_compact_resource_layout<C: crate::arithmetic::CurveAffine>() -> (usize, usize, usize) {
    lookup::prover::native_compact_resource_layout::<C>()
}

#[cfg(feature = "mv-lookup")]
/// Returns the native hash-map header size used by multi-lookup table membership.
pub fn native_mv_lookup_map_header_bytes() -> usize {
    std::mem::size_of::<HashMap<Vec<u8>, lookup::prover::TableMembership>>()
}

#[cfg(feature = "mv-lookup")]
/// Returns the shallow `size_of` for one `(Vec<u8>, TableMembership)` map entry.
/// Heap allocations owned by either value are not included.
pub fn native_mv_lookup_map_entry_bytes() -> usize {
    std::mem::size_of::<(Vec<u8>, lookup::prover::TableMembership)>()
}

/// Returns the native heap size of one retained advice event during bounded
/// batch evaluation and conversion.
pub fn native_assigned_event_bytes<F: PrimeField>() -> usize {
    std::mem::size_of::<AssignedEvent<F>>()
}

#[cfg(feature = "mv-lookup")]
struct NativePreparePolicy {
    python_budget_bytes: Option<u64>,
    device_headroom_bytes: u64,
    table_cardinality_bounds: Option<Vec<usize>>,
}

fn invalid_native_prepare_policy(message: impl Into<String>) -> Error {
    Error::NativeStorage(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

#[cfg(feature = "mv-lookup")]
fn native_prepare_policy() -> Result<NativePreparePolicy, Error> {
    let formal = std::env::var_os("EZKL_GAT_RUN_DIR").is_some();
    let python_budget_bytes = match std::env::var("EZKL_NATIVE_PREPARE_BUDGET_BYTES") {
        Ok(value) => Some(value.parse::<u64>().map_err(|_| {
            invalid_native_prepare_policy("EZKL_NATIVE_PREPARE_BUDGET_BYTES is not a u64")
        })?),
        Err(std::env::VarError::NotPresent) if !formal => None,
        Err(_) => {
            return Err(invalid_native_prepare_policy(
                "formal mv-lookup proving requires EZKL_NATIVE_PREPARE_BUDGET_BYTES",
            ))
        }
    };
    let device_headroom_bytes = match std::env::var("EZKL_NATIVE_PREPARE_DEVICE_HEADROOM_BYTES") {
        Ok(value) => value.parse::<u64>().map_err(|_| {
            invalid_native_prepare_policy(
                "EZKL_NATIVE_PREPARE_DEVICE_HEADROOM_BYTES is not a u64",
            )
        })?,
        Err(std::env::VarError::NotPresent) if !formal => 0,
        Err(_) => {
            return Err(invalid_native_prepare_policy(
                "formal mv-lookup proving requires EZKL_NATIVE_PREPARE_DEVICE_HEADROOM_BYTES",
            ))
        }
    };
    let table_cardinality_bounds = match std::env::var("EZKL_NATIVE_LOOKUP_CARDINALITY_BOUNDS") {
        Ok(value) if value.is_empty() => Some(Vec::new()),
        Ok(value) => {
            let bounds = value
                .split(',')
                .map(|item| {
                    let parsed = item.parse::<u64>().map_err(|_| {
                        invalid_native_prepare_policy(
                            "EZKL_NATIVE_LOOKUP_CARDINALITY_BOUNDS contains a non-u64 value",
                        )
                    })?;
                    if parsed == 0 {
                        return Err(invalid_native_prepare_policy(
                            "EZKL_NATIVE_LOOKUP_CARDINALITY_BOUNDS contains zero",
                        ));
                    }
                    usize::try_from(parsed).map_err(|_| {
                        invalid_native_prepare_policy(
                            "EZKL_NATIVE_LOOKUP_CARDINALITY_BOUNDS exceeds usize",
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Some(bounds)
        }
        Err(std::env::VarError::NotPresent) if !formal => None,
        Err(_) => {
            return Err(invalid_native_prepare_policy(
                "formal mv-lookup proving requires EZKL_NATIVE_LOOKUP_CARDINALITY_BOUNDS",
            ))
        }
    };
    Ok(NativePreparePolicy {
        python_budget_bytes,
        device_headroom_bytes,
        table_cardinality_bounds,
    })
}

fn native_prepare_available_bytes(device_headroom_bytes: u64) -> Result<u64, Error> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")
        .map_err(|error| Error::NativeStorage(io::Error::new(error.kind(), error)))?;
    let mut mem_available = None;
    let mut mem_total = None;
    for line in meminfo.lines() {
        let (key, value) = line.split_once(':').unwrap_or(("", ""));
        if key == "MemAvailable" || key == "MemTotal" {
            let kilobytes = value
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .ok_or_else(|| invalid_native_prepare_policy("/proc/meminfo has malformed memory values"))?;
            let bytes = kilobytes.checked_mul(1024).ok_or_else(|| {
                invalid_native_prepare_policy("/proc/meminfo memory value overflow")
            })?;
            if key == "MemAvailable" {
                mem_available = Some(bytes);
            } else {
                mem_total = Some(bytes);
            }
        }
    }
    let mut available = mem_available
        .ok_or_else(|| invalid_native_prepare_policy("/proc/meminfo lacks MemAvailable"))?;
    let mut limit = mem_total
        .ok_or_else(|| invalid_native_prepare_policy("/proc/meminfo lacks MemTotal"))?;

    let mut cgroup_paths = Vec::new();
    if let Ok(cgroup) = std::fs::read_to_string("/proc/self/cgroup") {
        for line in cgroup.lines() {
            let Some((hierarchy, tail)) = line.split_once(':') else {
                continue;
            };
            let Some((_controllers, relative)) = tail.split_once(':') else {
                continue;
            };
            if hierarchy == "0" {
                let mut path = PathBuf::from("/sys/fs/cgroup").join(relative.trim_start_matches('/'));
                loop {
                    if !cgroup_paths.contains(&path) {
                        cgroup_paths.push(path.clone());
                    }
                    if path == PathBuf::from("/sys/fs/cgroup") || !path.pop() {
                        break;
                    }
                }
                break;
            }
        }
    }
    let cgroup_root = PathBuf::from("/sys/fs/cgroup");
    if !cgroup_paths.contains(&cgroup_root) {
        cgroup_paths.push(cgroup_root);
    }
    for base in cgroup_paths {
        let maximum = match std::fs::read_to_string(base.join("memory.max")) {
            Ok(value) => value.trim().to_owned(),
            Err(_) => continue,
        };
        if maximum == "max" {
            continue;
        }
        let Ok(maximum) = maximum.parse::<u64>() else {
            return Err(invalid_native_prepare_policy(format!(
                "malformed cgroup memory.max at {}",
                base.display()
            )));
        };
        let current = match std::fs::read_to_string(base.join("memory.current"))
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
        {
            Some(value) => value,
            None => continue,
        };
        limit = limit.min(maximum);
        available = available.min(maximum.saturating_sub(current));
    }
    let reserve_percent = ((u128::from(limit) * 5 + 99) / 100)
        .try_into()
        .map_err(|_| invalid_native_prepare_policy("native memory reserve overflow"))?;
    let reserve = (2u64 * 1024 * 1024 * 1024).max(reserve_percent);
    Ok(available
        .saturating_sub(reserve)
        .saturating_sub(device_headroom_bytes))
}

/// Chooses a bounded SHPLONK contribution batch from the formal proof budget
/// and the current effective host availability. The batch is local to the
/// existing contribution loop and does not change Rayon global-pool settings.
pub(crate) fn native_shplonk_batch_width(
    requested_width: usize,
    per_task_bytes: u64,
    fixed_workspace_bytes: u64,
) -> Result<(usize, u64), Error> {
    if requested_width == 0 || per_task_bytes == 0 {
        return Err(invalid_native_prepare_policy(
            "native SHPLONK contribution batch has an empty width or zero task bound",
        ));
    }
    let formal = std::env::var_os("EZKL_GAT_RUN_DIR").is_some();
    let thread_width = crate::multicore::current_num_threads().max(1);
    let requested_width = requested_width.min(thread_width).max(1);
    let python_budget_bytes = match std::env::var("EZKL_NATIVE_PREPARE_BUDGET_BYTES") {
        Ok(value) => Some(value.parse::<u64>().map_err(|_| {
            invalid_native_prepare_policy("EZKL_NATIVE_PREPARE_BUDGET_BYTES is not a u64")
        })?),
        Err(std::env::VarError::NotPresent) if !formal => None,
        Err(_) => {
            return Err(invalid_native_prepare_policy(
                "formal SHPLONK proving requires EZKL_NATIVE_PREPARE_BUDGET_BYTES",
            ))
        }
    };
    let device_headroom_bytes = match std::env::var("EZKL_NATIVE_PREPARE_DEVICE_HEADROOM_BYTES") {
        Ok(value) => value.parse::<u64>().map_err(|_| {
            invalid_native_prepare_policy(
                "EZKL_NATIVE_PREPARE_DEVICE_HEADROOM_BYTES is not a u64",
            )
        })?,
        Err(std::env::VarError::NotPresent) if !formal => 0,
        Err(_) => {
            return Err(invalid_native_prepare_policy(
                "formal SHPLONK proving requires EZKL_NATIVE_PREPARE_DEVICE_HEADROOM_BYTES",
            ))
        }
    };
    if let Some(python_budget_bytes) = python_budget_bytes {
        let available = python_budget_bytes.min(native_prepare_available_bytes(
            device_headroom_bytes,
        )?);
        let task_envelope = available.saturating_sub(fixed_workspace_bytes);
        let width = task_envelope / per_task_bytes;
        if width == 0 {
            return Err(invalid_native_prepare_policy(format!(
                "native SHPLONK contribution needs {} fixed workspace bytes plus one {}-byte task; only {} bytes remain",
                fixed_workspace_bytes, per_task_bytes, available
            )));
        }
        Ok((
            usize::try_from(width)
                .unwrap_or(usize::MAX)
                .min(requested_width)
                .max(1),
            available,
        ))
    } else {
        Ok((requested_width, u64::MAX))
    }
}
use group::prime::PrimeCurveAffine;

const ASSIGNED_EVENT_BATCH_SIZE: usize = 8192;

struct AssignedEvent<F: PrimeField> {
    row: usize,
    value: Assigned<F>,
}

/// Holds only assigned cells, rather than a full `N`-element `Assigned<F>`
/// vector for each advice column. Assigned values stay in a bounded batch long
/// enough to use the native batch-inversion semantics; the private spill stores
/// only their evaluated field values and preserves order for last-write-wins.
struct AssignedEventColumn<F: PrimeField> {
    log: Option<crate::poly::SpillLog>,
    record: Vec<u8>,
    pending: Vec<AssignedEvent<F>>,
    event_count: u64,
    encoded_event_count: u64,
    encoded_bytes: u64,
    _marker: std::marker::PhantomData<F>,
}

impl<F: PrimeField> AssignedEventColumn<F> {
    fn new() -> Self {
        let repr_len = F::Repr::default().as_ref().len();
        Self {
            log: None,
            record: vec![0; 9 + repr_len],
            pending: Vec::new(),
            event_count: 0,
            encoded_event_count: 0,
            encoded_bytes: 0,
            _marker: std::marker::PhantomData,
        }
    }

    fn push(&mut self, row: usize, value: Assigned<F>) -> io::Result<()> {
        let row = u64::try_from(row).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "advice row exceeds spill format",
            )
        })?;
        let next_event_count = self.event_count.checked_add(1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "advice event count overflow")
        })?;
        self.pending.push(AssignedEvent {
            row: usize::try_from(row).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "advice row exceeds host index")
            })?,
            value,
        });
        self.event_count = next_event_count;
        if self.pending.len() == ASSIGNED_EVENT_BATCH_SIZE {
            self.flush_pending()?;
        }
        Ok(())
    }

    fn flush_pending(&mut self) -> io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }

        let mut inverse_denominators = self
            .pending
            .iter()
            .map(|event| event.value.denominator())
            .collect::<Vec<_>>();
        inverse_denominators
            .iter_mut()
            .filter_map(Option::as_mut)
            .batch_invert();

        if self.log.is_none() {
            self.log = Some(crate::poly::native_spill_log()?);
        }
        let repr_len = F::Repr::default().as_ref().len();
        if self.record.len() != 9 + repr_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "advice event writer buffer has an invalid record width",
            ));
        }
        for (event, inverse) in self.pending.drain(..).zip(inverse_denominators) {
            let row = u64::try_from(event.row).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "advice row exceeds spill format")
            })?;
            let value = event.value.numerator() * inverse.unwrap_or(F::ONE);
            self.record[..8].copy_from_slice(&row.to_le_bytes());
            let encoded_len = if value == F::ZERO {
                self.record[8] = 0;
                9
            } else {
                self.record[8] = 1;
                let repr = value.to_repr();
                self.record[9..9 + repr_len].copy_from_slice(repr.as_ref());
                9 + repr_len
            };
            let next_event_count = self.encoded_event_count.checked_add(1).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "encoded advice event count overflow")
            })?;
            let next_encoded_bytes = self
                .encoded_bytes
                .checked_add(u64::try_from(encoded_len).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "advice event width exceeds u64")
                })?)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "advice event byte length overflow")
                })?;
            self.log
                .as_mut()
                .expect("assigned event log was initialized")
                .append(&self.record[..encoded_len])?;
            self.encoded_event_count = next_event_count;
            self.encoded_bytes = next_encoded_bytes;
        }
        Ok(())
    }

    fn into_lagrange(mut self, domain_len: usize) -> io::Result<Polynomial<F, LagrangeCoeff>> {
        self.flush_pending()?;
        let Some(mut log) = self.log.take() else {
            if self.event_count != 0
                || self.encoded_event_count != 0
                || self.encoded_bytes != 0
                || !self.pending.is_empty()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "advice event counters are nonzero without a spill log",
                ));
            }
            return Polynomial::mapped_filled(domain_len, F::ZERO);
        };
        let expected_bytes = self.encoded_bytes;
        let minimum_bytes = self.event_count.checked_mul(9).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "advice event minimum byte length overflow",
            )
        })?;
        let maximum_bytes = self
            .event_count
            .checked_mul(self.record.len() as u64)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "advice event maximum byte length overflow",
                )
            })?;
        if log.bytes_written() != expected_bytes
            || self.event_count == 0
            || self.encoded_event_count != self.event_count
            || expected_bytes < minimum_bytes
            || expected_bytes > maximum_bytes
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "advice event log length does not match its authenticated in-memory count",
            ));
        }

        let mut latest = Polynomial::<u64, LagrangeCoeff>::mapped_filled(domain_len, 0)?;
        {
            let mut reader = log.reader()?;
            let mut remaining = self.event_count;
            let mut first_index = 0u64;
            let mut consumed_bytes = 0u64;
            while remaining > 0 {
                let count = usize::try_from(remaining.min(ASSIGNED_EVENT_BATCH_SIZE as u64))
                    .map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "advice batch length overflow")
                    })?;
                for offset in 0..count {
                    let (event, event_bytes) = read_assigned_event::<F, _>(&mut reader)?;
                    consumed_bytes = consumed_bytes
                        .checked_add(event_bytes as u64)
                        .ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "advice event byte count overflow",
                            )
                        })?;
                    if event.row >= domain_len {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "advice event row exceeds native polynomial domain",
                        ));
                    }
                    latest[event.row] = first_index
                        .checked_add(offset as u64)
                        .and_then(|index| index.checked_add(1))
                        .ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "advice event index overflow",
                            )
                        })?;
                }
                first_index = first_index.checked_add(count as u64).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "advice event index overflow")
                })?;
                remaining -= count as u64;
            }
            if consumed_bytes != expected_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "advice event decoder byte count disagrees with its writer count",
                ));
            }
            if reader.fill_buf()?.len() != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "advice event log has trailing bytes",
                ));
            }
        }

        let mut values = Polynomial::<F, LagrangeCoeff>::mapped_filled(domain_len, F::ZERO)?;
        {
            let mut reader = log.reader()?;
            let mut remaining = self.event_count;
            let mut first_index = 0u64;
            let mut consumed_bytes = 0u64;
            while remaining > 0 {
                let count = usize::try_from(remaining.min(ASSIGNED_EVENT_BATCH_SIZE as u64))
                    .map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "advice batch length overflow")
                    })?;
                let mut events = Vec::with_capacity(count);
                for _ in 0..count {
                    let (event, event_bytes) = read_assigned_event::<F, _>(&mut reader)?;
                    consumed_bytes = consumed_bytes
                        .checked_add(event_bytes as u64)
                        .ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "advice event byte count overflow",
                            )
                        })?;
                    if event.row >= domain_len {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "advice event row exceeds native polynomial domain",
                        ));
                    }
                    events.push(event);
                }
                let mut inverse_denominators = events
                    .iter()
                    .map(|event| event.value.denominator())
                    .collect::<Vec<_>>();
                inverse_denominators
                    .iter_mut()
                    .filter_map(Option::as_mut)
                    .batch_invert();
                for (offset, (event, inverse)) in
                    events.iter().zip(inverse_denominators).enumerate()
                {
                    let event_index = first_index
                        .checked_add(offset as u64)
                        .and_then(|index| index.checked_add(1))
                        .ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "advice event index overflow",
                            )
                        })?;
                    if latest[event.row] == event_index {
                        values[event.row] = event.value.numerator() * inverse.unwrap_or(F::ONE);
                    }
                }
                first_index = first_index.checked_add(count as u64).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "advice event index overflow")
                })?;
                remaining -= count as u64;
            }
            if consumed_bytes != expected_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "advice event decoder byte count disagrees with its writer count",
                ));
            }
            if reader.fill_buf()?.len() != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "advice event log has trailing bytes",
                ));
            }
        }
        drop(log);
        Ok(values)
    }
}

fn read_assigned_event<F: PrimeField, R: Read>(
    reader: &mut R,
) -> io::Result<(AssignedEvent<F>, usize)> {
    let repr_len = F::Repr::default().as_ref().len();
    let mut header = [0u8; 9];
    reader.read_exact(&mut header)?;
    let row = usize::try_from(u64::from_le_bytes(
        header[..8]
            .try_into()
            .expect("fixed advice event row width"),
    ))
    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "advice row exceeds host index"))?;
    let (value, encoded_len) = match header[8] {
        0 => (Assigned::Zero, 9),
        1 => {
            let mut numerator_repr = F::Repr::default();
            reader.read_exact(numerator_repr.as_mut())?;
            let numerator = Option::<F>::from(F::from_repr(numerator_repr)).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "advice numerator is not a field encoding",
                )
            })?;
            if numerator == F::ZERO {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "zero trivial advice event was not normalized",
                ));
            }
            (Assigned::Trivial(numerator), 9 + repr_len)
        }
        2 => {
            let mut numerator_repr = F::Repr::default();
            let mut denominator_repr = F::Repr::default();
            reader.read_exact(numerator_repr.as_mut())?;
            reader.read_exact(denominator_repr.as_mut())?;
            let numerator = Option::<F>::from(F::from_repr(numerator_repr)).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "advice numerator is not a field encoding",
                )
            })?;
            let denominator = Option::<F>::from(F::from_repr(denominator_repr)).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "advice denominator is not a field encoding",
                )
            })?;
            (
                Assigned::Rational(numerator, denominator),
                9 + 2 * repr_len,
            )
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "advice event has an invalid variant tag",
            ))
        }
    };
    Ok((AssignedEvent { row, value }, encoded_len))
}

/// This creates a proof for the provided `circuit` when given the public
/// parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
pub fn create_proof<
    'params,
    Scheme: CommitmentScheme,
    P: Prover<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore + Send + Sync,
    T: TranscriptWrite<Scheme::Curve, E>,
    ConcreteCircuit: Circuit<Scheme::Scalar>,
>(
    params: &'params Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Curve>,
    circuits: &[ConcreteCircuit],
    instances: &[&[&[Scheme::Scalar]]],
    mut rng: R,
    transcript: &mut T,
) -> Result<(), Error>
where
    Scheme::Scalar: PrimeField + WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
    Scheme::ParamsProver: Send + Sync,
{
    #[cfg(feature = "counter")]
    {
        use crate::{FFT_COUNTER, MSM_COUNTER};
        use std::collections::BTreeMap;

        // reset counters at the beginning of the prove
        *MSM_COUNTER.lock().unwrap() = BTreeMap::new();
        *FFT_COUNTER.lock().unwrap() = BTreeMap::new();
    }

    if circuits.len() != instances.len() {
        return Err(Error::InvalidInstances);
    }

    for instance in instances.iter() {
        if instance.len() != pk.vk.cs.num_instance_columns {
            return Err(Error::InvalidInstances);
        }
    }

    let start = Instant::now();
    // Hash verification key into transcript
    pk.vk.hash_into(transcript)?;
    log::trace!("Hashing verification key: {:?}", start.elapsed());

    let domain = &pk.vk.domain;
    let mut meta = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut meta, circuits[0].params());
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut meta);

    // Selector optimizations cannot be applied here; use the ConstraintSystem
    // from the verification key.
    let meta = &pk.vk.cs;

    struct InstanceSingle<C: CurveAffine> {
        pub instance_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
        pub instance_polys: Vec<Polynomial<C::Scalar, Coeff>>,
    }

    let start = Instant::now();
    let mut instance: Vec<InstanceSingle<Scheme::Curve>> = instances
        .iter()
        .map(|instance| -> Result<InstanceSingle<Scheme::Curve>, Error> {
            let instance_values = instance
                .iter()
                .map(|values| {
                    let mut poly = Polynomial::<Scheme::Scalar, LagrangeCoeff>::mapped_filled(params.n() as usize, Scheme::Scalar::ZERO).map_err(Error::NativeStorage)?;
                    assert_eq!(poly.len(), params.n() as usize);
                    if values.len() > (poly.len() - (meta.blinding_factors() + 1)) {
                        return Err(Error::InstanceTooLarge);
                    }
                    for (poly, value) in poly.iter_mut().zip(values.iter()) {
                        if !P::QUERY_INSTANCE {
                            transcript.common_scalar(*value)?;
                        }
                        *poly = *value;
                    }
                    Ok(poly)
                })
                .collect::<Result<Vec<_>, _>>()?;

            if P::QUERY_INSTANCE {
                let instance_commitments_projective: Vec<_> = instance_values
                    .iter()
                    .map(|poly| {
                        let _commitment_timer = crate::plonk::commitment_timing::instance_guard();
                        params.commit_lagrange(poly, Blind::default())
                    })
                    .collect();
                let mut instance_commitments =
                    vec![Scheme::Curve::identity(); instance_commitments_projective.len()];
                {
                    let _commitment_timer = crate::plonk::commitment_timing::normalization_guard();
                    <Scheme::Curve as CurveAffine>::CurveExt::batch_normalize(
                        &instance_commitments_projective,
                        &mut instance_commitments,
                    );
                }
                let instance_commitments = instance_commitments;
                drop(instance_commitments_projective);

                for commitment in &instance_commitments {
                    transcript.common_point(*commitment)?;
                }
            }

            Ok(InstanceSingle {
                instance_values,
                instance_polys: Vec::new(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    log::trace!("Instance preparation: {:?}", start.elapsed());

    #[derive(Clone)]
    struct AdviceSingle<C: CurveAffine, B: Basis> {
        pub advice_polys: Vec<Polynomial<C::Scalar, B>>,
        pub advice_blinds: Vec<Blind<C::Scalar>>,
    }

    struct WitnessCollection<'a, F: PrimeField> {
        k: u32,
        current_phase: sealed::Phase,
        advice: Vec<AssignedEventColumn<F>>,
        unblinded_advice: HashSet<usize>,
        challenges: &'a HashMap<usize, F>,
        instances: &'a [&'a [F]],
        usable_rows: RangeTo<usize>,
        _marker: std::marker::PhantomData<F>,
    }

    impl<'a, F: PrimeField> Assignment<F> for WitnessCollection<'a, F> {
        fn enter_region<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about regions in this context.
        }

        fn exit_region(&mut self) {
            // Do nothing; we don't care about regions in this context.
        }

        fn enable_selector<A, AR>(&mut self, _: A, _: &Selector, _: usize) -> Result<(), Error>
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // We only care about advice columns here

            Ok(())
        }

        fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // Do nothing
        }

        fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.k));
            }

            self.instances
                .get(column.index())
                .and_then(|column| column.get(row))
                .map(|v| Value::known(*v))
                .ok_or(Error::BoundsFailure)
        }

        fn assign_advice<V, VR, A, AR>(
            &mut self,
            _: A,
            column: Column<Advice>,
            row: usize,
            to: V,
        ) -> Result<(), Error>
        where
            V: FnOnce() -> Value<VR>,
            VR: Into<Assigned<F>>,
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            let target_phase = column.column_type().phase;
            if target_phase > self.current_phase {
                // A future-phase value can depend on challenges that have not
                // been sampled yet. Preserve the original phase boundary by
                // not evaluating its callback during earlier phases.
                return Ok(());
            }

            if target_phase < self.current_phase {
                // Re-synthesis in a later phase must still populate the
                // AssignedCell returned by Region, so downstream host-side
                // witness calculations can read this earlier-phase value.
                // Earlier-phase events are not part of this phase's witness.
                let value = to().into_field();
                if row == 0 {
                    let mut witness_known = false;
                    let _ = value.as_ref().map(|_| witness_known = true);
                    println!(
                        "EZKL_NATIVE_PHASE_ASSIGN current_phase={:?} target_phase={:?} column={} row={} witness_known={} persisted=false",
                        self.current_phase,
                        target_phase,
                        column.index(),
                        row,
                        witness_known,
                    );
                }
                return Ok(());
            }

            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.k));
            }

            let value = to().into_field();
            let mut witness_known = false;
            let _ = value.as_ref().map(|_| witness_known = true);
            if !witness_known {
                println!(
                    "EZKL_NATIVE_PHASE_ASSIGN current_phase={:?} target_phase={:?} column={} row={} witness_known=false persisted=false",
                    self.current_phase,
                    target_phase,
                    column.index(),
                    row,
                );
            }

            self.advice
                .get_mut(column.index())
                .ok_or(Error::BoundsFailure)?
                .push(row, value.assign()?)
                .map_err(Error::NativeStorage)?;

            Ok(())
        }

        fn assign_fixed<V, VR, A, AR>(
            &mut self,
            _: A,
            _: Column<Fixed>,
            _: usize,
            _: V,
        ) -> Result<(), Error>
        where
            V: FnOnce() -> Value<VR>,
            VR: Into<Assigned<F>>,
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // We only care about advice columns here

            Ok(())
        }

        fn copy(
            &mut self,
            _: Column<Any>,
            _: usize,
            _: Column<Any>,
            _: usize,
        ) -> Result<(), Error> {
            // We only care about advice columns here

            Ok(())
        }

        fn fill_from_row(
            &mut self,
            _: Column<Fixed>,
            _: usize,
            _: Value<Assigned<F>>,
        ) -> Result<(), Error> {
            Ok(())
        }

        fn get_challenge(&self, challenge: Challenge) -> Value<F> {
            self.challenges
                .get(&challenge.index())
                .cloned()
                .map(Value::known)
                .unwrap_or_else(Value::unknown)
        }

        fn push_namespace<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about namespaces in this context.
        }

        fn pop_namespace(&mut self, _: Option<String>) {
            // Do nothing; we don't care about namespaces in this context.
        }
    }

    let start = Instant::now();
    let (advice, challenges) = {
        let mut advice = vec![
            AdviceSingle::<Scheme::Curve, LagrangeCoeff> {
                advice_polys: (0..meta.num_advice_columns)
                    .map(|_| Polynomial::from_vec(Vec::new()))
                    .collect(),
                advice_blinds: vec![Blind::default(); meta.num_advice_columns],
            };
            instances.len()
        ];
        let s = FxBuildHasher;
        let mut challenges =
            HashMap::<usize, Scheme::Scalar>::with_capacity_and_hasher(meta.num_challenges, s);

        let unusable_rows_start = params.n() as usize - (meta.blinding_factors() + 1);
        let initial_advice_phase = meta
            .phases()
            .next()
            .expect("constraint system must expose its initial advice phase");
        for current_phase in pk.vk.cs.phases() {
            let _start = Instant::now();
            let column_indices = meta
                .advice_column_phase
                .iter()
                .enumerate()
                .filter_map(|(column_index, phase)| {
                    if current_phase == *phase {
                        Some(column_index)
                    } else {
                        None
                    }
                })
                .collect::<BTreeSet<_>>();

            for ((circuit, advice), instances) in
                circuits.iter().zip(advice.iter_mut()).zip(instances)
            {
                let _start = Instant::now();
                let mut witness = WitnessCollection {
                    k: params.k(),
                    current_phase,
                    advice: (0..meta.num_advice_columns)
                        .map(|_| AssignedEventColumn::<Scheme::Scalar>::new())
                        .collect(),
                    unblinded_advice: HashSet::from_iter(meta.unblinded_advice_columns.clone()),
                    instances,
                    challenges: &challenges,
                    // The prover will not be allowed to assign values to advice
                    // cells that exist within inactive rows, which include some
                    // number of blinding factors and an extra row for use in the
                    // permutation argument.
                    usable_rows: ..unusable_rows_start,
                    _marker: std::marker::PhantomData,
                };

                let _start = Instant::now();
                // Synthesize the circuit to obtain the witness and other information.
                ConcreteCircuit::FloorPlanner::synthesize(
                    &mut witness,
                    circuit,
                    config.clone(),
                    meta.constants.clone(),
                )?;

                let _start = Instant::now();
                // Finish each current-phase source log before choosing an
                // order, so encoded_bytes is the exact on-disk event length.
                // Descending source size drains the largest logs first while
                // the retained Lagrange outputs grow; tie-break by column
                // index for deterministic materialization order.
                let mut materialize_indices = Vec::with_capacity(column_indices.len());
                for column_index in &column_indices {
                    let source = &mut witness.advice[*column_index];
                    source.flush_pending().map_err(Error::NativeStorage)?;
                    // `drain(..)` preserves Vec capacity; no more assignments
                    // occur in this phase, so release that per-column batch
                    // buffer before materializing any full-domain polynomials.
                    source.pending = Vec::new();
                    materialize_indices.push(*column_index);
                }
                materialize_indices.sort_by(|left, right| {
                    witness.advice[*right]
                        .encoded_bytes
                        .cmp(&witness.advice[*left].encoded_bytes)
                        .then_with(|| left.cmp(right))
                });

                // Convert one assigned column at a time. Each source backing
                // is released before the next column is materialized, and
                // outputs remain stored at their original column indices.
                for column_index in materialize_indices {
                    let assigned = std::mem::replace(
                        &mut witness.advice[column_index],
                        AssignedEventColumn::<Scheme::Scalar>::new(),
                    );
                    let mut advice_values = assigned
                        .into_lagrange(params.n() as usize)
                        .map_err(Error::NativeStorage)?;
                    if !witness.unblinded_advice.contains(&column_index) {
                        for cell in &mut advice_values[unusable_rows_start..] {
                            *cell = Scheme::Scalar::random(&mut rng);
                        }
                    } else {
                        for cell in &mut advice_values[unusable_rows_start..] {
                            *cell = Blind::default().0;
                        }
                    }
                    advice.advice_polys[column_index] = advice_values;
                }

                let _start = Instant::now();
                // Compute commitments to advice column polynomials
                let blinds: Vec<_> = column_indices
                    .iter()
                    .map(|i| {
                        if witness.unblinded_advice.contains(i) {
                            Blind::default()
                        } else {
                            Blind(Scheme::Scalar::random(&mut rng))
                        }
                    })
                    .collect();
                let advice_commitments_projective: Vec<_> = column_indices
                    .iter()
                    .zip(blinds.iter())
                    .map(|(column_index, blind)| {
                        let poly = &advice.advice_polys[*column_index];
                        if current_phase == initial_advice_phase {
                            let _commitment_timer = crate::plonk::commitment_timing::advice_guard();
                            params.commit_lagrange(poly, *blind)
                        } else {
                            params.commit_lagrange(poly, *blind)
                        }
                    })
                    .collect();
                let mut advice_commitments =
                    vec![Scheme::Curve::identity(); advice_commitments_projective.len()];
                if current_phase == initial_advice_phase {
                    let _commitment_timer = crate::plonk::commitment_timing::normalization_guard();
                    <Scheme::Curve as CurveAffine>::CurveExt::batch_normalize(
                        &advice_commitments_projective,
                        &mut advice_commitments,
                    );
                } else {
                    <Scheme::Curve as CurveAffine>::CurveExt::batch_normalize(
                        &advice_commitments_projective,
                        &mut advice_commitments,
                    );
                }
                let advice_commitments = advice_commitments;
                drop(advice_commitments_projective);

                let _start = Instant::now();
                for commitment in &advice_commitments {
                    transcript.write_point(*commitment)?;
                }
                for (column_index, blind) in column_indices.iter().zip(blinds) {
                    advice.advice_blinds[*column_index] = blind;
                }
            }

            for (index, phase) in meta.challenge_phase.iter().enumerate() {
                if current_phase == *phase {
                    let existing =
                        challenges.insert(index, *transcript.squeeze_challenge_scalar::<()>());
                    assert!(existing.is_none());
                }
            }
        }

        assert_eq!(challenges.len(), meta.num_challenges);
        let challenges = (0..meta.num_challenges)
            .map(|index| challenges.remove(&index).unwrap())
            .collect::<Vec<_>>();

        (advice, challenges)
    };
    log::trace!("Advice preparation: {:?}", start.elapsed());

    // Sample theta challenge for keeping lookup columns linearly independent
    let start = Instant::now();
    let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();
    log::trace!("Theta challenge: {:?}", start.elapsed());

    let start = Instant::now();

    #[cfg(feature = "mv-lookup")]
    let prepare_policy = native_prepare_policy()?;

    #[cfg(feature = "mv-lookup")]
    let lookup_run_label = std::env::var("EZKL_GAT_RUN_DIR")
        .ok()
        .and_then(|path| PathBuf::from(path).file_name().map(|name| name.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "nonformal".to_owned());

    #[cfg(feature = "mv-lookup")]
    println!(
        "EZKL_NATIVE_PHASE START stage=mv_lookup_prepare dataset={} instances={} arguments={} domain={}",
        lookup_run_label,
        instance.len(),
        pk.vk.cs.lookups.len(),
        params.n(),
    );

    #[cfg(feature = "mv-lookup")]
    let prepare_rows = params.n() as u128;
    #[cfg(feature = "mv-lookup")]
    let prepare_sparse_count_bound = prepare_policy
        .table_cardinality_bounds
        .as_ref()
        .and_then(|bounds| bounds.iter().copied().max())
        .map(|bound| bound as u128)
        .unwrap_or(prepare_rows);
    #[cfg(feature = "mv-lookup")]
    let prepare_task_bytes = prepare_rows
        .checked_mul((2 * std::mem::size_of::<Scheme::Scalar>()) as u128)
        .and_then(|bytes| {
            // The codebook-sized atomics and their u64 result overlap while
            // converting the sparse count bank. The Q/table ID tapes are
            // already reserved as retained compact backing in the Python
            // envelope and are not charged again to this per-task window.
            bytes.checked_add(
                prepare_sparse_count_bound
                    .checked_mul(
                        (std::mem::size_of::<std::sync::atomic::AtomicU64>()
                            + std::mem::size_of::<u64>()) as u128,
                    )?,
            )
        })
        .and_then(|bytes| {
            // At most one Q polynomial is active per task. During expression
            // evaluation/spill its source Vec and mapped backing overlap.
            let polynomial_header = std::mem::size_of::<Polynomial<Scheme::Scalar, LagrangeCoeff>>();
            let transient_vector_headers =
                std::mem::size_of::<Vec<std::sync::atomic::AtomicU64>>()
                    + std::mem::size_of::<Vec<Scheme::Scalar>>();
            let prepared_header = std::mem::size_of::<lookup::prover::Prepared<Scheme::Curve>>();
            bytes.checked_add((polynomial_header + transient_vector_headers + prepared_header) as u128)
        })
        .and_then(|bytes| u64::try_from(bytes).ok())
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| invalid_native_prepare_policy("mv-lookup per-task byte bound is invalid"))?;

    #[cfg(feature = "mv-lookup")]
    let lookups: Vec<Vec<lookup::prover::Prepared<Scheme::Curve>>> = instance
        .par_iter()
        .zip(advice.par_iter())
        .map(|(instance, advice)| -> Result<Vec<_>, Error> {
            // Table expressions are identical for several independent lookup
            // arguments in a proof instance. Evaluate the compressed table and
            // its existing byte-key membership map once per exact expression
            // vector, scoped to this instance's advice and the current theta.
            let table_cache_started = Instant::now();
            println!(
                "EZKL_NATIVE_PHASE START stage=mv_lookup_table_cache dataset={} arguments={}",
                lookup_run_label,
                pk.vk.cs.lookups.len(),
            );
            let mut table_cache = Vec::new();
            let mut tables_for_lookup = Vec::with_capacity(pk.vk.cs.lookups.len());
            for lookup in &pk.vk.cs.lookups {
                let cached = table_cache
                    .iter()
                    .find(|(expressions, _)| expressions == &lookup.table_expressions)
                    .map(|(_, table)| std::sync::Arc::clone(table));
                let table = match cached {
                    Some(table) => table,
                    None => {
                        let table_group_index = table_cache.len();
                        let cardinality_upper = match &prepare_policy.table_cardinality_bounds {
                            Some(bounds) => *bounds.get(table_group_index).ok_or_else(|| {
                                invalid_native_prepare_policy(
                                    "native mv-lookup table groups exceed admitted cardinality bounds",
                                )
                            })?,
                            None => params.n() as usize,
                        };
                        let table = std::sync::Arc::new(lookup.prepare_table(
                            &pk.vk,
                            params,
                            domain,
                            theta,
                            &advice.advice_polys,
                            &pk.fixed_values,
                            &instance.instance_values,
                            &challenges,
                            cardinality_upper,
                        )?);
                        table_cache.push((lookup.table_expressions.clone(), table.clone()));
                        table
                    }
                };
                tables_for_lookup.push(table);
            }
            if prepare_policy
                .table_cardinality_bounds
                .as_ref()
                .is_some_and(|bounds| bounds.len() != table_cache.len())
            {
                return Err(invalid_native_prepare_policy(
                    "native mv-lookup table group count disagrees with admitted cardinality bounds",
                ));
            }
            let (table_map_entries, table_map_capacity) = table_cache
                .iter()
                .map(|(_, table)| table.membership_map_shape())
                .fold((0usize, 0usize), |(entries, capacity), shape| {
                    (entries.saturating_add(shape.0), capacity.saturating_add(shape.1))
                });
            println!(
                "EZKL_NATIVE_PHASE DONE stage=mv_lookup_table_cache dataset={} groups={} arguments={} map_entries={} map_capacity={} map_capacity_bytes_upper={} table_row_id_tape_bytes={} table_codebook_and_winner_bytes={} elapsed_ms={:.3}",
                lookup_run_label,
                table_cache.len(),
                pk.vk.cs.lookups.len(),
                table_map_entries,
                table_map_capacity,
                (table_map_capacity as u128)
                    .saturating_mul((native_mv_lookup_map_entry_bytes() + 1) as u128)
                    .saturating_add(16u128 * table_cache.len() as u128)
                    .saturating_add(
                        (table_map_entries as u128)
                            .saturating_mul(Scheme::Scalar::default().to_repr().as_ref().len() as u128),
                    ),
                table_cache
                    .iter()
                    .map(|(_, table)| {
                        let (active_rows, width_bytes, _, _, _) = table.compact_storage_shape();
                        (active_rows as u128).saturating_mul(width_bytes as u128)
                    })
                    .fold(0u128, u128::saturating_add),
                table_cache
                    .iter()
                    .map(|(_, table)| {
                        let (_, _, _, codebook_capacity, winner_capacity) =
                            table.compact_storage_shape();
                        (codebook_capacity as u128)
                            .saturating_mul(std::mem::size_of::<Scheme::Scalar>() as u128)
                            .saturating_add(
                                (winner_capacity as u128)
                                    .saturating_mul(std::mem::size_of::<usize>() as u128),
                            )
                    })
                    .fold(0u128, u128::saturating_add),
                table_cache_started.elapsed().as_secs_f64() * 1000.0,
            );

            // Query polynomials and each argument's multiplicity polynomial
            // remain distinct; only the exact table work above is shared.
            let mut prepared = Vec::with_capacity(pk.vk.cs.lookups.len());
            let mut lookup_start = 0usize;
            while lookup_start < pk.vk.cs.lookups.len() {
                let (batch_width, available_budget) = if let Some(python_budget) = prepare_policy.python_budget_bytes {
                    let runtime_budget = native_prepare_available_bytes(
                        prepare_policy.device_headroom_bytes,
                    )?;
                    let available = python_budget.min(runtime_budget);
                    let all_instances = u64::try_from(instances.len()).map_err(|_| {
                        invalid_native_prepare_policy("native proof instance count exceeds u64")
                    })?;
                    let per_instance_budget = available / all_instances.max(1);
                    let width = per_instance_budget / prepare_task_bytes;
                    if width == 0 {
                        return Err(invalid_native_prepare_policy(format!(
                            "native mv-lookup prepare requires at least one bounded task of {} bytes per proof instance; only {} bytes remain",
                            prepare_task_bytes, available
                        )));
                    }
                    (usize::try_from(width)
                        .unwrap_or(usize::MAX)
                        .min(pk.vk.cs.lookups.len() - lookup_start)
                        .max(1), available)
                } else {
                    (pk.vk.cs.lookups.len() - lookup_start, u64::MAX)
                };
                let lookup_end = lookup_start
                    .checked_add(batch_width)
                    .ok_or_else(|| invalid_native_prepare_policy("mv-lookup batch index overflow"))?;
                let batch_started = Instant::now();
                println!(
                    "EZKL_NATIVE_PHASE START stage=mv_lookup_prepare_batch dataset={} first_argument={} end_argument={} total_arguments={} batch_width={} per_task_bytes={} available_budget_bytes={}",
                    lookup_run_label,
                    lookup_start,
                    lookup_end,
                    pk.vk.cs.lookups.len(),
                    batch_width,
                    prepare_task_bytes,
                    available_budget,
                );
                let batch = pk.vk.cs.lookups[lookup_start..lookup_end]
                    .par_iter()
                    .zip(tables_for_lookup[lookup_start..lookup_end].par_iter())
                    .enumerate()
                    .map(|(batch_index, (lookup, table))| {
                        lookup.prepare(
                            &pk.vk,
                            params,
                            domain,
                            theta,
                            &advice.advice_polys,
                            &pk.fixed_values,
                            &instance.instance_values,
                            &challenges,
                            table,
                            lookup_start + batch_index,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                prepared.extend(batch);
                println!(
                    "EZKL_NATIVE_PHASE DONE stage=mv_lookup_prepare_batch dataset={} first_argument={} end_argument={} total_arguments={} elapsed_ms={:.3}",
                    lookup_run_label,
                    lookup_start,
                    lookup_end,
                    pk.vk.cs.lookups.len(),
                    batch_started.elapsed().as_secs_f64() * 1000.0,
                );
                lookup_start = lookup_end;
            }
            Ok(prepared)
        })
        .collect::<Result<Vec<_>, _>>()?;

    #[cfg(feature = "mv-lookup")]
    let (lookup_query_id_tape_bytes, lookup_sparse_m_count_entries) = lookups
        .iter()
        .flatten()
        .map(|lookup| lookup.compact_storage_shape())
        .fold((0u128, 0u128), |(tape_bytes, sparse_entries), (bytes, entries)| {
            (
                tape_bytes.saturating_add(bytes as u128),
                sparse_entries.saturating_add(entries as u128),
            )
        });

    #[cfg(feature = "mv-lookup")]
    println!(
        "EZKL_NATIVE_PHASE DONE stage=mv_lookup_prepare dataset={} instances={} arguments={} table_groups={} query_id_tape_bytes={} sparse_m_count_entries={} elapsed_ms={:.3}",
        lookup_run_label,
        lookups.len(),
        pk.vk.cs.lookups.len(),
        prepare_policy
            .table_cardinality_bounds
            .as_ref()
            .map(Vec::len)
            .unwrap_or(0),
        lookup_query_id_tape_bytes,
        lookup_sparse_m_count_entries,
        start.elapsed().as_secs_f64() * 1000.0,
    );

    #[cfg(feature = "mv-lookup")]
    {
        for lookups_ in &lookups {
            for lookup in lookups_.iter() {
                transcript.write_point(lookup.commitment)?;
            }
        }
    }

    #[cfg(not(feature = "mv-lookup"))]
    let lookups: Vec<Vec<lookup::prover::Permuted<Scheme::Curve>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Result<Vec<_>, Error> {
            // Construct and commit to permuted values for each lookup
            pk.vk
                .cs
                .lookups
                .iter()
                .map(|lookup| {
                    lookup.commit_permuted(
                        pk,
                        params,
                        domain,
                        theta,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                        &mut rng,
                        transcript,
                    )
                })
                .collect()
        })
        .collect::<Result<Vec<_>, _>>()?;
    log::trace!("Lookup preparation: {:?}", start.elapsed());

    // Sample beta challenge
    let start = Instant::now();
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();
    log::trace!("Beta challenge: {:?}", start.elapsed());

    // Sample gamma challenge
    let start = Instant::now();
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();
    log::trace!("Gamma challenge: {:?}", start.elapsed());

    // Commit to permutations.
    let start = Instant::now();
    let permutations: Vec<permutation::prover::Committed<Scheme::Curve>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| {
            pk.vk.cs.permutation.commit(
                params,
                pk,
                &pk.permutation,
                &advice.advice_polys,
                &pk.fixed_values,
                &instance.instance_values,
                beta,
                gamma,
                &mut rng,
                transcript,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    log::trace!("Permutation commitment: {:?}", start.elapsed());

    // preallocate the lookups

    #[cfg(feature = "mv-lookup")]
    let commit_active_rows = (params.n() as usize)
        .checked_sub(pk.vk.cs.blinding_factors() + 1)
        .ok_or_else(|| invalid_native_prepare_policy("mv-lookup active-row count underflow"))?;
    #[cfg(feature = "mv-lookup")]
    let commit_task_bytes = (params.n() as u128)
        .checked_add(commit_active_rows as u128)
        .and_then(|rows| rows.checked_mul(std::mem::size_of::<Scheme::Scalar>() as u128))
        .and_then(|bytes| {
            // A commit task keeps the R-row difference accumulator while the
            // N-row mapped phi polynomial is built. Compact tapes, codebooks,
            // inverse codebooks, sparse counts, and blinds are retained-source
            // backing and are reserved separately by the admission envelope.
            bytes.checked_add(
                (4 * std::mem::size_of::<Vec<Scheme::Scalar>>()) as u128,
            )
        })
        .and_then(|bytes| u64::try_from(bytes).ok())
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| invalid_native_prepare_policy("mv-lookup commit per-task byte bound is invalid"))?;

    #[cfg(feature = "mv-lookup")]
    let phi_blinds = (0..pk.vk.cs.blinding_factors())
        .map(|_| Scheme::Scalar::random(&mut rng))
        .collect::<Vec<_>>();

    #[cfg(feature = "mv-lookup")]
    let commit_lookups = || -> Result<Vec<Vec<lookup::prover::Committed<Scheme::Curve>>>, _> {
        let total_lookup_arguments = lookups.iter().try_fold(0usize, |count, instance| {
            count.checked_add(instance.len()).ok_or_else(|| {
                invalid_native_prepare_policy("mv-lookup total argument count overflow")
            })
        })?;
        let mut remaining_arguments = total_lookup_arguments;
        let mut committed_instances = Vec::with_capacity(lookups.len());
        println!(
            "EZKL_NATIVE_PHASE START stage=mv_lookup_commit dataset={} instances={} arguments={} per_task_bytes={} phi_mapped_bytes_per_task_upper={}",
            lookup_run_label,
            lookups.len(),
            total_lookup_arguments,
            commit_task_bytes,
            (params.n() as u128)
                .saturating_mul(std::mem::size_of::<Scheme::Scalar>() as u128),
        );
        for ((instance_values, advice_values), pending) in
            instance.iter().zip(advice.iter()).zip(lookups)
        {
            let advice_values = &advice_values.advice_polys;
            let instance_values = &instance_values.instance_values;
            let mut pending = std::collections::VecDeque::from(pending);
            let mut committed = Vec::with_capacity(pending.len());
            let mut first_argument = total_lookup_arguments - remaining_arguments;
            while !pending.is_empty() {
                let (batch_width, available_budget) = if let Some(python_budget) = prepare_policy.python_budget_bytes {
                    let available = python_budget.min(native_prepare_available_bytes(
                        prepare_policy.device_headroom_bytes,
                    )?);
                    let width = available / commit_task_bytes;
                    if width == 0 {
                        return Err(invalid_native_prepare_policy(format!(
                            "native mv-lookup commit requires one bounded task of {} bytes; only {} bytes remain",
                            commit_task_bytes, available
                        )));
                    }
                    (usize::try_from(width)
                        .unwrap_or(usize::MAX)
                        .min(pending.len())
                        .max(1), available)
                } else {
                    (pending.len(), u64::MAX)
                };
                let batch_started = Instant::now();
                let mut batch = Vec::with_capacity(batch_width);
                for _ in 0..batch_width {
                    if let Some(lookup) = pending.pop_front() {
                        batch.push(lookup);
                    }
                }
                let batch_end_argument = first_argument + batch.len();
                println!(
                    "EZKL_NATIVE_PHASE START stage=mv_lookup_commit_batch dataset={} first_argument={} end_argument={} total_arguments={} batch_width={} per_task_bytes={} available_budget_bytes={} phi_mapped_bytes_per_task_upper={}",
                    lookup_run_label,
                    first_argument,
                    batch_end_argument,
                    total_lookup_arguments,
                    batch.len(),
                    commit_task_bytes,
                    available_budget,
                    (params.n() as u128)
                        .saturating_mul(std::mem::size_of::<Scheme::Scalar>() as u128),
                );
                #[cfg(feature = "metal")]
                let batch_result = batch
                    .into_iter()
                    .map(|lookup| {
                        lookup.commit_grand_sum(
                            &pk.vk,
                            params,
                            beta,
                            advice_values,
                            &pk.fixed_values,
                            instance_values,
                            &challenges,
                            &phi_blinds,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                #[cfg(not(feature = "metal"))]
                let batch_result = batch
                    .into_par_iter()
                    .map(|lookup| {
                        lookup.commit_grand_sum(
                            &pk.vk,
                            params,
                            beta,
                            advice_values,
                            &pk.fixed_values,
                            instance_values,
                            &challenges,
                            &phi_blinds,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                committed.extend(batch_result);
                remaining_arguments = remaining_arguments
                    .checked_sub(batch_width)
                    .ok_or_else(|| invalid_native_prepare_policy("mv-lookup remaining argument underflow"))?;
                println!(
                    "EZKL_NATIVE_PHASE DONE stage=mv_lookup_commit_batch dataset={} first_argument={} end_argument={} total_arguments={} elapsed_ms={:.3}",
                    lookup_run_label,
                    first_argument,
                    batch_end_argument,
                    total_lookup_arguments,
                    batch_started.elapsed().as_secs_f64() * 1000.0,
                );
                first_argument = batch_end_argument;
            }
            committed_instances.push(committed);
        }
        println!(
            "EZKL_NATIVE_PHASE DONE stage=mv_lookup_commit dataset={} instances={} arguments={} phi_mapped_bytes_per_task_upper={}",
            lookup_run_label,
            committed_instances.len(),
            total_lookup_arguments,
            (params.n() as u128)
                .saturating_mul(std::mem::size_of::<Scheme::Scalar>() as u128),
        );
        Ok(committed_instances)
    };

    #[cfg(not(feature = "mv-lookup"))]
    let commit_lookups = || -> Result<Vec<Vec<lookup::prover::Committed<Scheme::Curve>>>, _> {
        lookups
            .into_iter()
            .map(|lookups| -> Result<Vec<_>, _> {
                // Construct and commit to products for each lookup
                lookups
                    .into_iter()
                    .map(|lookup| {
                        lookup.commit_product(pk, params, beta, gamma, &mut rng, transcript)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
    };

    let start = Instant::now();
    let lookups = commit_lookups()?;

    #[cfg(feature = "mv-lookup")]
    {
        for lookups_ in &lookups {
            for lookup in lookups_.iter() {
                transcript.write_point(lookup.commitment)?;
            }
        }
    }

    log::trace!("Lookup commitment: {:?}", start.elapsed());

    let start = Instant::now();
    let shuffles: Vec<Vec<shuffle::prover::Committed<Scheme::Curve>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Result<Vec<_>, _> {
            // Compress expressions for each shuffle
            pk.vk
                .cs
                .shuffles
                .iter()
                .map(|shuffle| {
                    shuffle.commit_product(
                        pk,
                        params,
                        domain,
                        theta,
                        gamma,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                        &mut rng,
                        transcript,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    log::trace!("Shuffle commitment: {:?}", start.elapsed());

    // Shuffle commitments are the final consumer of the Lagrange-form
    // instance columns. Reuse their backing for the coefficient form instead
    // of allocating and copying another full column before the iFFT.
    let start = Instant::now();
    for proof_instance in &mut instance {
        proof_instance.instance_polys = std::mem::take(&mut proof_instance.instance_values)
            .into_iter()
            .map(|poly| domain.lagrange_to_coeff(poly))
            .collect();
    }
    log::trace!("Instance coefficient conversion: {:?}", start.elapsed());

    let start = Instant::now();
    // Commit to the vanishing argument's random polynomial for blinding h(x_3)
    let vanishing = vanishing::Argument::commit(params, domain, &mut rng, transcript)?;
    log::trace!("Vanishing commitment: {:?}", start.elapsed());

    // Obtain challenge for keeping all separate gates linearly independent
    let start = Instant::now();
    let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();
    log::trace!("Y challenge: {:?}", start.elapsed());

    // Calculate the advice polys
    let start = Instant::now();
    let advice: Vec<AdviceSingle<Scheme::Curve, Coeff>> = advice
        .into_iter()
        .map(|AdviceSingle {
                 advice_polys,
                 advice_blinds,
             }| {
                AdviceSingle {
                    advice_polys: advice_polys
                        .into_iter()
                        .map(|poly| domain.lagrange_to_coeff(poly))
                        .collect::<Vec<_>>(),
                    advice_blinds,
                }
            })
        .collect();
    log::trace!("Advice calculation: {:?}", start.elapsed());

    // Evaluate the h(X) polynomial
    let start = Instant::now();
    let h_poly = pk.ev.evaluate_h(
        pk,
        &advice
            .iter()
            .map(|a| a.advice_polys.as_slice())
            .collect::<Vec<_>>(),
        &instance
            .iter()
            .map(|i| i.instance_polys.as_slice())
            .collect::<Vec<_>>(),
        &challenges,
        *y,
        *beta,
        *gamma,
        *theta,
        &lookups,
        &shuffles,
        &permutations,
    )?;
    log::trace!("H(X) evaluation: {:?}", start.elapsed());

    // Construct the vanishing argument's h(X) commitments
    let start = Instant::now();
    let vanishing = vanishing.construct(params, domain, h_poly, &mut rng, transcript)?;
    log::trace!("Vanishing construction: {:?}", start.elapsed());

    let start = Instant::now();
    let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
    let xn = x.pow([params.n()]);
    log::trace!("X challenge: {:?}", start.elapsed());

    let start = Instant::now();
    if P::QUERY_INSTANCE {
        // Compute and hash instance evals for each circuit instance
        for instance in instance.iter() {
            // Evaluate polynomials at omega^i x
            let instance_evals: Vec<_> = meta
                .instance_queries
                .iter()
                .map(|&(column, at)| {
                    eval_polynomial(
                        &instance.instance_polys[column.index()],
                        domain.rotate_omega(*x, at),
                    )
                })
                .collect();

            // Hash each instance column evaluation
            for eval in instance_evals.iter() {
                transcript.write_scalar(*eval)?;
            }
        }
    }
    log::trace!("Instance evaluation: {:?}", start.elapsed());

    let start = Instant::now();
    // Compute and hash advice evals for each circuit instance
    let mut advice_evals_by_circuit = Vec::with_capacity(advice.len());
    for advice in advice.iter() {
        // Evaluate polynomials at omega^i x
        let advice_evals: Vec<_> = meta
            .advice_queries
            .iter()
            .map(|&(column, at)| {
                eval_polynomial(
                    &advice.advice_polys[column.index()],
                    domain.rotate_omega(*x, at),
                )
            })
            .collect();

        // Hash each advice column evaluation
        for eval in advice_evals.iter() {
            transcript.write_scalar(*eval)?;
        }
        advice_evals_by_circuit.push(advice_evals);
    }
    log::trace!("Advice evaluation: {:?}", start.elapsed());

    let start = Instant::now();
    // Compute and hash fixed evals (shared across all circuit instances)
    let mut fixed_queries_by_column = vec![Vec::new(); pk.fixed_values.len()];
    for (query_index, &(column, at)) in meta.fixed_queries.iter().enumerate() {
        fixed_queries_by_column[column.index()].push((query_index, at));
    }
    let mut fixed_evals = vec![Scheme::Scalar::ZERO; meta.fixed_queries.len()];
    for (column_index, queries) in fixed_queries_by_column.iter().enumerate() {
        if queries.is_empty() {
            continue;
        }
        let source = &pk.fixed_values[column_index];
        if let Some(value) = source.constant_value() {
            assert_eq!(source.len(), domain.get_n() as usize);
            for &(query_index, _) in queries {
                fixed_evals[query_index] = value;
            }
        } else {
            let lagrange = source
                .materialize_lagrange()
                .map_err(Error::NativeStorage)?;
            let coeff = domain
                .lagrange_to_coeff(lagrange)
                .spill()
                .map_err(Error::NativeStorage)?;
            for &(query_index, at) in queries {
                fixed_evals[query_index] = eval_polynomial(&coeff, domain.rotate_omega(*x, at));
            }
            drop(coeff);
        }
    }
    log::trace!("Fixed evaluation: {:?}", start.elapsed());

    // Hash each fixed column evaluation
    let start = Instant::now();
    for eval in fixed_evals.iter() {
        transcript.write_scalar(*eval)?;
    }
    log::trace!("Fixed evaluation hashing: {:?}", start.elapsed());

    let start = Instant::now();
    let vanishing = vanishing.evaluate(x, xn, domain, transcript)?;
    log::trace!("Vanishing evaluation: {:?}", start.elapsed());

    // Evaluate common permutation data
    let start = Instant::now();
    let permutation_omega_powers = permutation::omega_powers(domain)
        .map_err(Error::NativeStorage)?;
    let permutation_evaluations =
        pk.permutation
            .evaluate(x, domain, &permutation_omega_powers[..], transcript)?;
    log::trace!("Permutation evaluation: {:?}", start.elapsed());

    // Evaluate the permutations, if any, at omega^i x.
    let start = Instant::now();
    let permutations: Vec<permutation::prover::Evaluated<Scheme::Curve>> = permutations
        .into_iter()
        .map(|permutation| -> Result<_, _> { permutation.construct().evaluate(pk, x, transcript) })
        .collect::<Result<Vec<_>, _>>()?;
    log::trace!("Permutation evaluation: {:?}", start.elapsed());

    // Evaluate the lookups, if any, at omega^i x.

    let start = Instant::now();

    let lookups: Vec<Vec<lookup::prover::Evaluated<Scheme::Curve>>> = lookups
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            lookups
                .into_iter()
                .map(|p| {
                    #[cfg(not(feature = "mv-lookup"))]
                    let res = { p.evaluate(pk, x, transcript) };
                    #[cfg(feature = "mv-lookup")]
                    let res = { p.evaluate(&pk.vk, x, transcript) };
                    res
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    log::trace!("Lookup evaluation: {:?}", start.elapsed());

    // Evaluate the shuffles, if any, at omega^i x.
    let start = Instant::now();
    let shuffles: Vec<Vec<shuffle::prover::Evaluated<Scheme::Curve>>> = shuffles
        .into_iter()
        .map(|shuffles| -> Result<Vec<_>, _> {
            shuffles
                .into_iter()
                .map(|p| p.evaluate(pk, x, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    log::trace!("Shuffle evaluation: {:?}", start.elapsed());

    let start = Instant::now();
    let instances = instance
        .iter()
        .zip(advice.iter())
        .zip(advice_evals_by_circuit.iter())
        .zip(permutations.iter())
        .zip(lookups.iter())
        .zip(shuffles.iter())
        .flat_map(|(((((instance, advice), advice_evals), permutation), lookups), shuffles)| {
            iter::empty()
                .chain(
                    P::QUERY_INSTANCE
                        .then_some(pk.vk.cs.instance_queries.iter().map(move |&(column, at)| ProverQuery::new(domain.rotate_omega(*x, at), &instance.instance_polys[column.index()], Blind::default())))
                        .into_iter()
                        .flatten(),
                )
                .chain(
                    pk.vk
                        .cs
                        .advice_queries
                        .iter()
                        .zip(advice_evals.iter())
                        .map(move |(&(column, at), &eval)| ProverQuery::new_with_eval(domain.rotate_omega(*x, at), &advice.advice_polys[column.index()], eval, advice.advice_blinds[column.index()])),
                )
                .chain(permutation.open(pk, x))
                .chain(lookups.iter().flat_map(move |p| p.open(pk, x)))
                .chain(shuffles.iter().flat_map(move |p| p.open(pk, x)))
        })
        .chain(pk.vk.cs.fixed_queries.iter().zip(fixed_evals.iter().copied()).map(
            |(&(column, at), eval)| {
                ProverQuery::new_fixed_column(
                    domain.rotate_omega(*x, at),
                    &pk.fixed_values[column.index()],
                    domain,
                    eval,
                )
            },
        ))
        .chain(pk.permutation.open(
            x,
            domain,
            &permutation_omega_powers[..],
            &permutation_evaluations[..],
        ))
        // We query the h(X) polynomial at x
        .chain(vanishing.open(x));
    log::trace!("Open queries: {:?}", start.elapsed());

    #[cfg(feature = "counter")]
    {
        use crate::{FFT_COUNTER, MSM_COUNTER};
        use std::collections::BTreeMap;
        log::debug!("MSM_COUNTER: {:?}", MSM_COUNTER.lock().unwrap());
        log::debug!("FFT_COUNTER: {:?}", *FFT_COUNTER.lock().unwrap());

        // reset counters at the end of the proving
        *MSM_COUNTER.lock().unwrap() = BTreeMap::new();
        *FFT_COUNTER.lock().unwrap() = BTreeMap::new();
    }

    let prover = P::new(params);
    prover
        .create_proof(rng, transcript, instances)
        .map_err(|_| Error::ConstraintSystemFailure)
}

#[test]
fn test_create_proof() {
    use crate::{
        circuit::{SimpleFloorPlanner, Value},
        plonk::{
            keygen_pk, keygen_vk, verify_proof, Advice, Column, ConstraintSystem, Fixed,
            FixedColumnSource, PackedIdTape, PackedIdWidth, Selector,
        },
        poly::kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
            strategy::SingleStrategy,
        },
        poly::Rotation,
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
    };
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[derive(Clone, Copy)]
    struct MyCircuit([u64; 2]);

    impl<F: Field + From<u64>> Circuit<F> for MyCircuit {
        type Config = (Selector, Column<Advice>, Column<Fixed>);
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            MyCircuit([0, 0])
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let instance = meta.instance_column();
            let advice = meta.advice_column();
            let fixed = meta.fixed_column();
            let q_instance = meta.selector();
            meta.create_gate("advice follows public instance at cur and next", |meta| {
                let q_instance = meta.query_selector(q_instance);
                let instance_cur = meta.query_instance(instance, Rotation::cur());
                let instance_next = meta.query_instance(instance, Rotation::next());
                let advice_cur = meta.query_advice(advice, Rotation::cur());
                let advice_next = meta.query_advice(advice, Rotation::next());
                let fixed_cur = meta.query_fixed(fixed, Rotation::cur());
                vec![
                    q_instance.clone() * (advice_cur - instance_cur - fixed_cur),
                    q_instance * (advice_next - instance_next),
                ]
            });
            (q_instance, advice, fixed)
        }

        fn synthesize(
            &self,
            (q_instance, advice, _fixed): Self::Config,
            mut layouter: impl crate::circuit::Layouter<F>,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "public instance constraint",
                |mut region| {
                    q_instance.enable(&mut region, 0)?;
                    region.assign_advice(|| "advice value cur", advice, 0, || Value::known(F::from(self.0[0])))?;
                    region.assign_advice(|| "advice value next", advice, 1, || Value::known(F::from(self.0[1])))?;
                    Ok(())
                },
            )?;
            Ok(())
        }
    }

    let params: ParamsKZG<Bn256> = ParamsKZG::setup(3, ChaCha20Rng::from_seed([0x11; 32]));
    let keygen_circuit = MyCircuit([7, 9]);
    let vk = keygen_vk(&params, &keygen_circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &keygen_circuit).expect("keygen_pk should not fail");

    let opening_poly = Polynomial::<Fr, Coeff>::from_vec((0..params.n()).map(|index| Fr::from(index as u64 + 1)).collect());
    let opening_points = [Fr::from(13), Fr::from(17)];
    let opening_evals = opening_points.map(|point| eval_polynomial(&opening_poly[..], point));
    let opening_prover = ProverSHPLONK::new(&params);
    let mut legacy_opening = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    opening_prover
        .create_proof(ChaCha20Rng::from_seed([0x44; 32]), &mut legacy_opening, [ProverQuery::new(opening_points[0], &opening_poly, Blind::default()), ProverQuery::new(opening_points[1], &opening_poly, Blind::default())])
        .expect("legacy dense opening should succeed");
    let mut cached_opening = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    opening_prover
        .create_proof(ChaCha20Rng::from_seed([0x44; 32]), &mut cached_opening, [ProverQuery::new_with_eval(opening_points[0], &opening_poly, opening_evals[0], Blind::default()), ProverQuery::new_with_eval(opening_points[1], &opening_poly, opening_evals[1], Blind::default())])
        .expect("cached-evaluation opening should succeed");
    assert_eq!(legacy_opening.finalize(), cached_opening.finalize());

    // Compare the prior copy-then-iFFT path with consuming the original mapped
    // Lagrange backing. The move path must retain that same allocation.
    let domain = &pk.get_vk().domain;
    let fixed_domain = domain;
    let mut instance_values =
        Polynomial::<Fr, LagrangeCoeff>::mapped_filled(params.n() as usize, Fr::ZERO)
            .expect("create mapped instance values");
    for (index, value) in instance_values.iter_mut().enumerate() {
        *value = Fr::from(index as u64 + 1);
    }
    let mut copied_values =
        Polynomial::<Fr, LagrangeCoeff>::mapped_filled(params.n() as usize, Fr::ZERO)
            .expect("create legacy copied instance values");
    copied_values.copy_from_slice(&instance_values[..]);
    let expected_coefficients = domain.lagrange_to_coeff(copied_values);
    let instance_backing = instance_values.as_ptr();
    let actual_coefficients = domain.lagrange_to_coeff(instance_values);
    assert_eq!(
        actual_coefficients.as_ptr(),
        instance_backing,
        "basis conversion should preserve the consumed mapped backing"
    );
    assert_eq!(&actual_coefficients[..], &expected_coefficients[..]);

    // Constant compact fixed columns use the exact coefficient representation
    // of the old materialize-plus-iFFT path; nonconstant columns keep that path.
    let fixed_len = params.n() as usize;
    let fixed_cases = [
        vec![Fr::ZERO; fixed_len],
        vec![Fr::from(5); fixed_len],
        (0..fixed_len).map(|row| Fr::from(row as u64)).collect(),
    ];
    for (case_index, values) in fixed_cases.iter().cloned().enumerate() {
        let source = FixedColumnSource::from_polynomial(Polynomial::<Fr, LagrangeCoeff>::from_vec(values))
            .expect("build compact fixed source");
        assert_eq!(source.constant_value().is_some(), case_index < 2);
        let old_coefficients = fixed_domain
            .lagrange_to_coeff(source.materialize_lagrange().expect("materialize fixed source"));
        let new_coefficients = ProverQuery::<G1Affine>::new_fixed_column(
            Fr::from(3),
            &source,
            fixed_domain,
            Fr::ZERO,
        )
        .poly
        .materialize()
        .expect("materialize fixed query");
        assert_eq!(&new_coefficients[..], &old_coefficients[..]);
    }

    let invalid_id_width = PackedIdWidth::for_cardinality(3).expect("three-value ID width");
    let mut invalid_ids = PackedIdTape::zeroed(fixed_len, invalid_id_width).expect("allocate ID tape");
    invalid_ids.write(0, 3).expect("encode out-of-range ID");
    assert!(FixedColumnSource::from_compact(vec![Fr::ZERO; 3], invalid_ids).is_err());

    let short_ids = PackedIdTape::zeroed(fixed_len - 1, PackedIdWidth::for_cardinality(1).unwrap())
        .expect("allocate short ID tape");
    let short_source = FixedColumnSource::from_compact(vec![Fr::from(5)], short_ids)
        .expect("construct short constant source");
    let short_query = ProverQuery::<G1Affine>::new_fixed_column(Fr::from(3), &short_source, fixed_domain, Fr::from(5));
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| short_query.poly.materialize())).is_err());

    // Create proof with wrong number of instances
    let mut invalid_transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    let proof = create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[MyCircuit([7, 9]), MyCircuit([8, 10])],
        &[],
        ChaCha20Rng::from_seed([0x22; 32]),
        &mut invalid_transcript,
    );
    assert!(matches!(proof.unwrap_err(), Error::InvalidInstances));

    // Create proof with correct number of instances
    let public_a = [Fr::from(7), Fr::from(9)];
    let public_b = [Fr::from(8), Fr::from(10)];
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[MyCircuit([7, 9]), MyCircuit([8, 10])],
        &[&[&public_a[..]], &[&public_b[..]]],
        ChaCha20Rng::from_seed([0x33; 32]),
        &mut transcript,
    )
    .expect("proof generation should not fail");

    let proof = transcript.finalize();
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    verify_proof::<KZGCommitmentScheme<_>, VerifierSHPLONK<_>, _, _, SingleStrategy<_>>(
        &params,
        pk.get_vk(),
        SingleStrategy::new(&params),
        &[&[&public_a[..]], &[&public_b[..]]],
        &mut transcript,
        params.n(),
    )
    .expect("proof with nonempty public instances should verify independently");
}

#[test]
fn packed_selector_count_vk_roundtrip_and_proof_verify() {
    use crate::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            census_native_shape, count_compressed_selector_columns, keygen_pk, keygen_vk, verify_proof, Advice,
            Circuit, Column, ConstraintSystem, Error, Selector,
        },
        poly::{
            kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::{ProverSHPLONK, VerifierSHPLONK},
                strategy::SingleStrategy,
            },
            Rotation as PolyRotation,
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
        SerdeFormat,
    };
    use ff::Field;
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use rand_core::OsRng;

    #[derive(Clone, Copy)]
    struct SelectorCircuit;

    #[derive(Clone)]
    struct Config {
        advice: Column<Advice>,
        first: Selector,
        second: Selector,
    }

    impl Circuit<Fr> for SelectorCircuit {
        type Config = Config;
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            *self
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let advice = meta.advice_column();
            meta.enable_equality(advice);
            let first = meta.selector();
            let second = meta.selector();
            meta.create_gate("two disjoint selector activations", |meta| {
                let value = meta.query_advice(advice, PolyRotation::cur());
                vec![
                    meta.query_selector(first) * value.clone(),
                    meta.query_selector(second) * value,
                ]
            });
            Config {
                advice,
                first,
                second,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "two selector rows",
                |mut region| {
                    let first = region.assign_advice(
                        || "first row value",
                        config.advice,
                        0,
                        || Value::known(Fr::ZERO),
                    )?;
                    let second = region.assign_advice(
                        || "second row value",
                        config.advice,
                        1,
                        || Value::known(Fr::ZERO),
                    )?;
                    region.constrain_equal(first.cell(), second.cell())?;
                    config.first.enable(&mut region, 0)?;
                    config.second.enable(&mut region, 1)?;
                    Ok(())
                },
            )?;
            Ok(())
        }
    }

    let circuit = SelectorCircuit;
    let params: ParamsKZG<Bn256> = ParamsKZG::setup(3, OsRng);
    let census = census_native_shape::<G1Affine, _>(3, &circuit).expect("native shape census");
    assert_eq!(census.selector_fixed_column_count, 1);
    assert_eq!(census.permutation_copy_call_count, 1);
    let selector_columns =
        count_compressed_selector_columns::<G1Affine, _>(3, &circuit).expect("selector census");
    assert_eq!(selector_columns, 1);

    let vk = keygen_vk(&params, &circuit).expect("keygen_vk should succeed");
    assert_eq!(vk.fixed_commitments.len(), selector_columns);
    let vk_bytes = vk.to_bytes(SerdeFormat::RawBytes);
    assert_eq!(
        &vk_bytes[vk_bytes.len() - 2..],
        &[0b0000_0001, 0b0000_0010],
        "selector wire bits retain the legacy little-endian packing"
    );
    #[cfg(feature = "circuit-params")]
    let decoded_vk = super::VerifyingKey::<G1Affine>::from_bytes::<SelectorCircuit>(&vk_bytes, SerdeFormat::RawBytes, ());
    #[cfg(not(feature = "circuit-params"))]
    let decoded_vk = super::VerifyingKey::<G1Affine>::from_bytes::<SelectorCircuit>(&vk_bytes, SerdeFormat::RawBytes);
    let decoded_vk = decoded_vk.expect("packed selector VK should roundtrip");
    assert_eq!(decoded_vk.transcript_repr(), vk.transcript_repr());

    let pk = keygen_pk(&params, vk, &circuit).expect("keygen_pk should succeed");
    // Reload the compact sparse mapping and exercise the supported deferred
    // sigma source through proof creation and independent verification.
    let pk_bytes = pk.to_bytes(SerdeFormat::RawBytes);
    #[cfg(feature = "circuit-params")]
    let pk = super::ProvingKey::from_bytes::<SelectorCircuit>(
        &pk_bytes,
        SerdeFormat::RawBytes,
        (),
    )
    .expect("packed selector PK should reload");
    #[cfg(not(feature = "circuit-params"))]
    let pk = super::ProvingKey::from_bytes::<SelectorCircuit>(&pk_bytes, SerdeFormat::RawBytes)
        .expect("packed selector PK should reload");
    let mut writer = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[circuit],
        &[&[]],
        OsRng,
        &mut writer,
    )
    .expect("packed selector proof should be created");
    let proof = writer.finalize();

    let mut reader = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    verify_proof::<KZGCommitmentScheme<_>, VerifierSHPLONK<_>, _, _, SingleStrategy<_>>(
        &params,
        &decoded_vk,
        SingleStrategy::new(&params),
        &[&[]],
        &mut reader,
        params.n(),
    )
    .expect("proof should verify with the VK loaded from packed selector bytes");
}

#[test]
fn initial_advice_commitment_phase_is_phase_zero_only() {
    use super::circuit::sealed::SealedPhase;
    use halo2curves::bn256::Fr;

    let mut cs = ConstraintSystem::<Fr>::default();
    cs.advice_column_in::<super::circuit::FirstPhase>(super::circuit::FirstPhase);
    cs.advice_column_in::<super::circuit::SecondPhase>(super::circuit::SecondPhase);

    let mut phases = cs.phases();
    let initial_advice_phase = phases
        .next()
        .expect("constraint system must expose an initial advice phase");
    let challenge_dependent_phase = phases
        .next()
        .expect("second-phase advice column must expose its phase");
    assert_eq!(initial_advice_phase, super::circuit::FirstPhase.to_sealed());
    assert_ne!(challenge_dependent_phase, initial_advice_phase);
    assert!(phases.next().is_none());
}

#[test]
fn earlier_phase_values_feed_later_phase_without_evaluating_future_callbacks() {
    use crate::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column, ConstraintSystem, Error,
            Expression, FirstPhase, SecondPhase, Selector, ThirdPhase,
        },
        poly::{
            kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::{ProverSHPLONK, VerifierSHPLONK},
                strategy::SingleStrategy,
            },
            Rotation as PolyRotation,
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
    };
    use ff::Field;
    use halo2curves::bn256::{Bn256, Fr};
    use rand_core::OsRng;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FUTURE_PHASE_CALLBACKS: AtomicUsize = AtomicUsize::new(0);

    #[derive(Clone, Copy)]
    struct MultiPhaseCircuit;

    #[derive(Clone)]
    struct Config {
        early: Column<Advice>,
        late: Column<Advice>,
        future: Column<Advice>,
        q_late: Selector,
    }

    impl Circuit<Fr> for MultiPhaseCircuit {
        type Config = Config;
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            *self
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let early = meta.advice_column_in::<FirstPhase>(FirstPhase);
            let late = meta.advice_column_in::<SecondPhase>(SecondPhase);
            let future = meta.advice_column_in::<ThirdPhase>(ThirdPhase);
            let q_late = meta.selector();
            meta.create_gate("later phase consumes earlier value", |meta| {
                let q = meta.query_selector(q_late);
                let early = meta.query_advice(early, PolyRotation::cur());
                let late = meta.query_advice(late, PolyRotation::cur());
                vec![q * (late - early - Expression::Constant(Fr::ONE))]
            });
            Config {
                early,
                late,
                future,
                q_late,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "phase-aware witness propagation",
                |mut region| {
                    let early = region.assign_advice(
                        || "early phase input",
                        config.early,
                        0,
                        || Value::known(Fr::from(5)),
                    )?;
                    let _late = region.assign_advice(
                        || "late phase derived value",
                        config.late,
                        0,
                        || early.value().map(|value| *value + Fr::ONE),
                    )?;
                    let _future = region.assign_advice(
                        || "future phase value",
                        config.future,
                        0,
                        || {
                            FUTURE_PHASE_CALLBACKS.fetch_add(1, Ordering::SeqCst);
                            Value::known(Fr::from(9))
                        },
                    )?;
                    config.q_late.enable(&mut region, 0)?;
                    Ok(())
                },
            )?;
            Ok(())
        }
    }

    let params: ParamsKZG<Bn256> = ParamsKZG::setup(3, OsRng);
    let vk = keygen_vk(&params, &MultiPhaseCircuit).expect("keygen_vk should succeed");
    let pk = keygen_pk(&params, vk, &MultiPhaseCircuit).expect("keygen_pk should succeed");

    FUTURE_PHASE_CALLBACKS.store(0, Ordering::SeqCst);
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[MultiPhaseCircuit],
        &[&[]],
        OsRng,
        &mut transcript,
    )
    .expect("later-phase operation must receive the retained earlier-phase value");
    assert_eq!(
        FUTURE_PHASE_CALLBACKS.load(Ordering::SeqCst),
        1,
        "future-phase callback must run only in its declared phase"
    );
    let proof = transcript.finalize();

    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    verify_proof::<KZGCommitmentScheme<_>, VerifierSHPLONK<_>, _, _, SingleStrategy<_>>(
        &params,
        pk.get_vk(),
        SingleStrategy::new(&params),
        &[&[]],
        &mut transcript,
        params.n(),
    )
    .expect("multi-phase proof should verify");
}

#[cfg(test)]
mod assigned_event_tests {
    use super::{Assigned, AssignedEventColumn};
    use crate::poly::native_spill_scope;
    use ff::Field;
    use halo2curves::bn256::Fr;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn assigned_events_preserve_zero_rational_and_last_write() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before Unix epoch")
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "halo2-assigned-events-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir(&base).expect("create test scratch parent");
        let spill = native_spill_scope(&base).expect("create scoped spill directory");

        let mut assigned = AssignedEventColumn::<Fr>::new();
        assigned.push(1, Assigned::Zero).expect("append zero event");
        assigned
            .push(2, Assigned::Trivial(Fr::from(9)))
            .expect("append initial duplicate value");
        assigned
            .push(2, Assigned::Rational(Fr::from(20), Fr::from(4)))
            .expect("append rational duplicate value");
        assigned
            .push(3, Assigned::Rational(Fr::from(7), Fr::ZERO))
            .expect("append zero-denominator rational");
        assigned
            .push(4, Assigned::Rational(Fr::from(21), Fr::from(3)))
            .expect("append rational value");

        let values = assigned.into_lagrange(6).expect("decode assigned events");
        let expected = [
            Fr::ZERO,
            Fr::ZERO,
            Fr::from(5),
            Fr::ZERO,
            Fr::from(7),
            Fr::ZERO,
        ];
        assert!(values
            .iter()
            .zip(expected)
            .all(|(actual, expected)| *actual == expected));

        let empty = AssignedEventColumn::<Fr>::new()
            .into_lagrange(4)
            .expect("empty advice column remains zero");
        assert!(empty.iter().all(|value| *value == Fr::ZERO));

        let mut out_of_domain = AssignedEventColumn::<Fr>::new();
        out_of_domain
            .push(4, Assigned::Trivial(Fr::ONE))
            .expect("append out-of-domain event for conversion check");
        assert!(out_of_domain.into_lagrange(4).is_err());

        let unknown = crate::circuit::Value::<Fr>::unknown().into_field().assign();
        assert!(matches!(unknown, Err(crate::plonk::Error::Synthesis)));

        drop(empty);
        drop(values);
        drop(spill);
        std::fs::remove_dir_all(base).expect("remove test scratch parent");
    }
}
