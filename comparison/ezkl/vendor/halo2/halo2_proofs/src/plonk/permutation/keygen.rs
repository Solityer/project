use ff::{Field, PrimeField};
use group::Curve;

use super::{Argument, ProvingKey, VerifyingKey};
use crate::{
    arithmetic::{parallelize, CurveAffine},
    plonk::{Any, Column, Error},
    poly::{
        commitment::{Blind, Params},
        EvaluationDomain,
    },
};

use maybe_rayon::prelude::{IndexedParallelIterator, ParallelIterator};

#[cfg(feature = "thread-safe-region")]
use maybe_rayon::prelude::{IntoParallelIterator, IntoParallelRefMutIterator};

#[cfg(not(feature = "thread-safe-region"))]
use maybe_rayon::prelude::IntoParallelIterator;

#[cfg(feature = "thread-safe-region")]
use std::collections::BTreeSet;

#[cfg(feature = "thread-safe-region")]
use rustc_hash::FxHashMap as HashMap;

#[cfg(not(feature = "thread-safe-region"))]
/// Struct that accumulates all the necessary data in order to construct the permutation argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assembly {
    /// Columns that participate on the copy permutation argument.
    columns: Vec<Column<Any>>,
    /// Non-identity successor edges; every omitted cell maps to itself.
    mapping: std::collections::HashMap<usize, usize>,
    /// Non-default cycle representatives; every omitted cell represents itself.
    aux: std::collections::HashMap<usize, usize>,
    /// Non-singleton cycle sizes; every omitted representative has size one.
    sizes: std::collections::HashMap<usize, usize>,
    col_len: usize,
}

#[cfg(not(feature = "thread-safe-region"))]
impl Assembly {
    pub(crate) fn new(n: usize, p: &Argument) -> Self {
        Assembly {
            columns: p.columns.clone(),
            mapping: std::collections::HashMap::new(),
            aux: std::collections::HashMap::new(),
            sizes: std::collections::HashMap::new(),
            col_len: n,
        }
    }

    fn cell_id(&self, column: usize, row: usize) -> usize {
        assert!(
            column < self.columns.len() && row < self.col_len,
            "permutation cell is outside its declared shape"
        );
        column
            .checked_mul(self.col_len)
            .and_then(|base| base.checked_add(row))
            .expect("permutation cell identifier overflow")
    }

    fn cell_at(&self, cell: usize) -> (usize, usize) {
        (cell / self.col_len, cell % self.col_len)
    }

    fn mapping_at(&self, cell: usize) -> usize {
        self.mapping.get(&cell).copied().unwrap_or(cell)
    }

    fn mapping_at_idx(&self, column: usize, row: usize) -> (usize, usize) {
        self.cell_at(self.mapping_at(self.cell_id(column, row)))
    }

    fn set_mapping(&mut self, cell: usize, successor: usize) {
        if cell == successor {
            self.mapping.remove(&cell);
        } else {
            self.mapping.insert(cell, successor);
        }
    }

    fn representative(&self, cell: usize) -> usize {
        self.aux.get(&cell).copied().unwrap_or(cell)
    }

    fn cycle_size(&self, representative: usize) -> usize {
        self.sizes.get(&representative).copied().unwrap_or(1)
    }

    fn set_representative(&mut self, cell: usize, representative: usize) {
        if cell == representative {
            self.aux.remove(&cell);
        } else {
            self.aux.insert(cell, representative);
        }
    }

    pub(crate) fn copy(
        &mut self,
        left_column: Column<Any>,
        left_row: usize,
        right_column: Column<Any>,
        right_row: usize,
    ) -> Result<(), Error> {
        let left_column = self
            .columns
            .iter()
            .position(|c| c == &left_column)
            .ok_or(Error::ColumnNotInPermutation(left_column))?;
        let right_column = self
            .columns
            .iter()
            .position(|c| c == &right_column)
            .ok_or(Error::ColumnNotInPermutation(right_column))?;

        if left_row >= self.col_len || right_row >= self.col_len {
            return Err(Error::BoundsFailure);
        }

        // See book/src/design/permutation.md for a description of this algorithm.

        let left_cell = self.cell_id(left_column, left_row);
        let right_cell = self.cell_id(right_column, right_row);
        let mut left_cycle = self.representative(left_cell);
        let mut right_cycle = self.representative(right_cell);

        // If left and right are in the same cycle, do nothing.
        if left_cycle == right_cycle {
            return Ok(());
        }

        let mut left_size = self.cycle_size(left_cycle);
        let mut right_size = self.cycle_size(right_cycle);
        if left_size < right_size {
            std::mem::swap(&mut left_cycle, &mut right_cycle);
            std::mem::swap(&mut left_size, &mut right_size);
        }

        self.sizes.insert(
            left_cycle,
            left_size
                .checked_add(right_size)
                .expect("permutation cycle size overflow"),
        );
        self.sizes.remove(&right_cycle);
        let mut i = right_cycle;
        loop {
            self.set_representative(i, left_cycle);
            i = self.mapping_at(i);
            if i == right_cycle {
                break;
            }
        }

        let left_successor = self.mapping_at(left_cell);
        let right_successor = self.mapping_at(right_cell);
        self.set_mapping(left_cell, right_successor);
        self.set_mapping(right_cell, left_successor);

        Ok(())
    }

    pub(crate) fn build_vk<'params, C: CurveAffine, P: Params<'params, C>>(
        self,
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        p: &Argument,
    ) -> VerifyingKey<C> {
        build_vk(params, domain, p, move |i, j| self.mapping_at_idx(i, j))
    }

    pub(crate) fn build_pk<'params, C: CurveAffine, P: Params<'params, C>>(
        self,
        _params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        p: &Argument,
    ) -> Result<ProvingKey<C>, Error> {
        let Assembly {
            columns, mapping, ..
        } = self;
        if columns != p.get_columns() {
            return Err(Error::Synthesis);
        }
        let mut successors = mapping
            .into_iter()
            .map(|(source, target)| (source as u64, target as u64))
            .collect::<Vec<_>>();
        successors.sort_unstable_by_key(|entry| entry.0);
        ProvingKey::from_successors(domain.get_n(), columns, successors).map_err(Error::Transcript)
    }

    /// Returns columns that participate in the permutation argument.
    pub fn columns(&self) -> &[Column<Any>] {
        &self.columns
    }

    /// Returns mappings of the copies.
    pub fn mapping(
        &self,
    ) -> impl Iterator<Item = impl IndexedParallelIterator<Item = (usize, usize)> + '_> {
        (0..self.columns.len()).map(move |column| {
            (0..self.col_len)
                .into_par_iter()
                .map(move |row| self.mapping_at_idx(column, row))
        })
    }
}

#[cfg(feature = "thread-safe-region")]
/// Struct that accumulates all the necessary data in order to construct the permutation argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assembly {
    /// Columns that participate on the copy permutation argument.
    columns: Vec<Column<Any>>,
    /// Mapping of the actual copies done.
    cycles: Vec<Vec<(usize, usize)>>,
    /// Mapping of the actual copies done.
    ordered_cycles: Vec<BTreeSet<(usize, usize)>>,
    /// Mapping of the actual copies done.
    aux: HashMap<(usize, usize), usize>,
    /// total length of a column
    col_len: usize,
    /// number of columns
    num_cols: usize,
}

#[cfg(feature = "thread-safe-region")]
impl Assembly {
    pub(crate) fn new(n: usize, p: &Argument) -> Self {
        Assembly {
            columns: p.columns.clone(),
            cycles: Vec::with_capacity(n),
            ordered_cycles: Vec::with_capacity(n),
            aux: HashMap::default(),
            col_len: n,
            num_cols: p.columns.len(),
        }
    }

    pub(crate) fn copy(
        &mut self,
        left_column: Column<Any>,
        left_row: usize,
        right_column: Column<Any>,
        right_row: usize,
    ) -> Result<(), Error> {
        let left_column = self
            .columns
            .iter()
            .position(|c| c == &left_column)
            .ok_or(Error::ColumnNotInPermutation(left_column))?;
        let right_column = self
            .columns
            .iter()
            .position(|c| c == &right_column)
            .ok_or(Error::ColumnNotInPermutation(right_column))?;

        // Check bounds
        if left_row >= self.col_len || right_row >= self.col_len {
            return Err(Error::BoundsFailure);
        }

        let left_cycle = self.aux.get(&(left_column, left_row));
        let right_cycle = self.aux.get(&(right_column, right_row));

        // extract cycle elements
        let right_cycle_elems = match right_cycle {
            Some(i) => {
                let entry = self.cycles[*i].clone();
                self.cycles[*i] = vec![];
                entry
            }
            None => [(right_column, right_row)].into(),
        };

        assert!(right_cycle_elems.contains(&(right_column, right_row)));

        // merge cycles
        let cycle_idx = match left_cycle {
            Some(i) => {
                let entry = &mut self.cycles[*i];
                entry.extend(right_cycle_elems.clone());
                *i
            }
            // if they were singletons -- create a new cycle entry
            None => {
                let mut set: Vec<(usize, usize)> = right_cycle_elems.clone();
                set.push((left_column, left_row));
                self.cycles.push(set);
                let cycle_idx = self.cycles.len() - 1;
                self.aux.insert((left_column, left_row), cycle_idx);
                cycle_idx
            }
        };

        let index_updates = vec![cycle_idx; right_cycle_elems.len()].into_iter();
        let updates = right_cycle_elems.into_iter().zip(index_updates);

        self.aux.extend(updates);

        Ok(())
    }

    /// Builds the ordered mapping of the cycles.
    /// This will only get executed once.
    pub fn build_ordered_mapping(&mut self) {
        // will only get called once
        if self.ordered_cycles.is_empty() && !self.cycles.is_empty() {
            self.ordered_cycles = self
                .cycles
                .par_iter_mut()
                .map(|col| {
                    let mut set = BTreeSet::new();
                    set.extend(col.clone());
                    // free up memory
                    *col = vec![];
                    set
                })
                .collect();
        }
    }

    fn mapping_at_idx(&self, col: usize, row: usize) -> (usize, usize) {
        assert!(
            !self.ordered_cycles.is_empty() || self.cycles.is_empty(),
            "cycles have not been ordered"
        );

        if let Some(cycle_idx) = self.aux.get(&(col, row)) {
            let cycle = &self.ordered_cycles[*cycle_idx];
            let mut cycle_iter = cycle.range((
                std::ops::Bound::Excluded((col, row)),
                std::ops::Bound::Unbounded,
            ));
            // point to the next node in the cycle
            match cycle_iter.next() {
                Some((i, j)) => (*i, *j),
                // wrap back around to the first element which SHOULD exist
                None => *(cycle.iter().next().unwrap()),
            }
        // is a singleton
        } else {
            (col, row)
        }
    }

    pub(crate) fn build_vk<'params, C: CurveAffine, P: Params<'params, C>>(
        &mut self,
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        p: &Argument,
    ) -> VerifyingKey<C> {
        self.build_ordered_mapping();
        build_vk(params, domain, p, |i, j| self.mapping_at_idx(i, j))
    }

    pub(crate) fn build_pk<'params, C: CurveAffine, P: Params<'params, C>>(
        mut self,
        _params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        p: &Argument,
    ) -> Result<ProvingKey<C>, Error> {
        self.build_ordered_mapping();
        if self.columns != p.get_columns() {
            return Err(Error::Synthesis);
        }
        let mut successors = Vec::new();
        for cycle in &self.ordered_cycles {
            if cycle.len() < 2 {
                continue;
            }
            let members = cycle.iter().copied().collect::<Vec<_>>();
            for (index, &(column, row)) in members.iter().enumerate() {
                let (next_column, next_row) = members[(index + 1) % members.len()];
                let source = (column as u64)
                    .checked_mul(self.col_len as u64)
                    .and_then(|base| base.checked_add(row as u64))
                    .ok_or(Error::BoundsFailure)?;
                let target = (next_column as u64)
                    .checked_mul(self.col_len as u64)
                    .and_then(|base| base.checked_add(next_row as u64))
                    .ok_or(Error::BoundsFailure)?;
                successors.push((source, target));
            }
        }
        successors.sort_unstable_by_key(|entry| entry.0);
        ProvingKey::from_successors(domain.get_n(), p.columns.clone(), successors)
            .map_err(Error::Transcript)
    }

    /// Returns columns that participate in the permutation argument.
    pub fn columns(&self) -> &[Column<Any>] {
        &self.columns
    }

    /// Returns mappings of the copies.
    pub fn mapping(
        &self,
    ) -> impl Iterator<Item = impl IndexedParallelIterator<Item = (usize, usize)> + '_> {
        (0..self.num_cols).map(move |i| {
            (0..self.col_len)
                .into_par_iter()
                .map(move |j| self.mapping_at_idx(i, j))
        })
    }
}

pub(crate) fn build_vk<'params, C: CurveAffine, P: Params<'params, C>>(
    params: &P,
    domain: &EvaluationDomain<C::Scalar>,
    p: &Argument,
    mapping: impl Fn(usize, usize) -> (usize, usize) + Sync,
) -> VerifyingKey<C> {
    // Compute [omega^0, omega^1, ..., omega^{params.n - 1}]
    let mut omega_powers = vec![C::Scalar::ZERO; params.n() as usize];
    {
        let omega = domain.get_omega();
        parallelize(&mut omega_powers, |o, start| {
            let mut cur = omega.pow_vartime([start as u64]);
            for v in o.iter_mut() {
                *v = cur;
                cur *= &omega;
            }
        })
    }

    let mut delta_powers = vec![C::Scalar::ONE; p.columns.len()];
    for index in 1..delta_powers.len() {
        delta_powers[index] = delta_powers[index - 1] * C::Scalar::DELTA;
    }

    // The verification key retains only sigma commitments, so construct and
    // commit one dense sigma column at a time.
    let mut commitments = Vec::with_capacity(p.columns.len());
    for column in 0..p.columns.len() {
        let mut permutation = domain.empty_lagrange();
        parallelize(&mut permutation, |values, start| {
            for (offset, value) in values.iter_mut().enumerate() {
                let (permuted_column, permuted_row) = mapping(column, start + offset);
                *value = delta_powers[permuted_column] * omega_powers[permuted_row];
            }
        });
        commitments.push(
            params
                .commit_lagrange(&permutation, Blind::default())
                .to_affine(),
        );
    }

    VerifyingKey { commitments }
}

#[cfg(all(test, not(feature = "thread-safe-region")))]
mod sparse_assembly_tests {
    use super::{build_vk, Assembly, ProvingKey};
    use crate::{
        halo2curves::bn256::{Bn256, Fr, G1Affine},
        plonk::{permutation::Argument, Any, Column},
        poly::{commitment::{Blind, Params}, kzg::commitment::ParamsKZG, EvaluationDomain},
        SerdeFormat,
    };
    use ff::{Field, PrimeField};
    use group::Curve;
    use rand_core::OsRng;
    use std::io::Cursor;

    #[test]
    fn touched_sparse_successors_match_dense_assembly_and_vk_commitments() {
        let n = 8usize;
        let columns = (0..3)
            .map(|i| Column::new(i, Any::advice()))
            .collect::<Vec<_>>();
        let mut argument = Argument::new();
        for column in &columns {
            argument.add_column(*column);
        }

        let mut sparse = Assembly::new(n, &argument);
        let mut dense_mapping = (0..columns.len())
            .map(|column| (0..n).map(|row| (column, row)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut dense_aux = dense_mapping.clone();
        let mut dense_sizes = vec![vec![1usize; n]; columns.len()];
        let operations = [
            (0, 0, 1, 4), // join two untouched singleton cycles
            (2, 6, 0, 0), // size-balanced swap: the right cycle is larger
            (1, 4, 2, 6), // duplicate relation inside an existing cycle
            (0, 1, 1, 4),
            (2, 2, 0, 1), // cross-column cycle extension
            (2, 2, 2, 2), // self-copy is a no-op
            (1, 0, 2, 6),
            (0, 5, 0, 1),
            (1, 1, 2, 5),
            (1, 1, 1, 1),
        ];

        for (left_col, left_row, right_col, right_row) in operations {
            sparse
                .copy(columns[left_col], left_row, columns[right_col], right_row)
                .unwrap();

            let mut left_cycle = dense_aux[left_col][left_row];
            let mut right_cycle = dense_aux[right_col][right_row];
            if left_cycle != right_cycle {
                if dense_sizes[left_cycle.0][left_cycle.1]
                    < dense_sizes[right_cycle.0][right_cycle.1]
                {
                    std::mem::swap(&mut left_cycle, &mut right_cycle);
                }
                dense_sizes[left_cycle.0][left_cycle.1] +=
                    dense_sizes[right_cycle.0][right_cycle.1];
                let mut cell = right_cycle;
                loop {
                    dense_aux[cell.0][cell.1] = left_cycle;
                    cell = dense_mapping[cell.0][cell.1];
                    if cell == right_cycle {
                        break;
                    }
                }
                let next = dense_mapping[left_col][left_row];
                dense_mapping[left_col][left_row] = dense_mapping[right_col][right_row];
                dense_mapping[right_col][right_row] = next;
            }

            let delta = Fr::DELTA;
            let omega = Fr::ROOT_OF_UNITY.pow_vartime([1u64 << (Fr::S - 3)]);
            for column in 0..columns.len() {
                for row in 0..n {
                    let sparse_target = sparse.mapping_at_idx(column, row);
                    assert_eq!(
                        sparse_target, dense_mapping[column][row],
                        "successor mismatch after copy operation"
                    );
                    let sparse_cell = sparse.cell_id(column, row);
                    let sparse_root = sparse.representative(sparse_cell);
                    let dense_root = dense_aux[column][row];
                    assert_eq!(
                        sparse.cell_at(sparse_root),
                        dense_root,
                        "cycle representative mismatch after copy operation"
                    );
                    assert_eq!(
                        sparse.cycle_size(sparse_root),
                        dense_sizes[dense_root.0][dense_root.1],
                        "cycle size mismatch after copy operation"
                    );
                    let sparse_sigma = delta.pow_vartime([sparse_target.0 as u64])
                        * omega.pow_vartime([sparse_target.1 as u64]);
                    let dense_target = dense_mapping[column][row];
                    let dense_sigma = delta.pow_vartime([dense_target.0 as u64])
                        * omega.pow_vartime([dense_target.1 as u64]);
                    assert_eq!(
                        sparse_sigma, dense_sigma,
                        "sigma value mismatch after copy operation"
                    );
                }
            }
        }

        let params = ParamsKZG::<Bn256>::setup(3, OsRng);
        let domain = EvaluationDomain::<Fr>::new(3, 3);
        let sparse_vk = sparse.clone().build_vk(&params, &domain, &argument);
        let dense_vk = build_vk(&params, &domain, &argument, |column, row| {
            dense_mapping[column][row]
        });
        assert_eq!(
            sparse_vk.commitments(),
            dense_vk.commitments(),
            "verification-key sigma commitments differ"
        );

        // Exercise the serialized production PK path, including reconstruction
        // of coefficient-basis sigma polynomials. Compare every loaded native
        // sigma row and its KZG commitment against the same dense native map
        // used to construct the VK above.
        let native_pk = sparse
            .build_pk(&params, &domain, &argument)
            .expect("construct sparse permutation proving key");
        let mut encoded = Vec::new();
        native_pk
            .write(&mut encoded, SerdeFormat::RawBytes)
            .expect("serialize sparse permutation proving key");
        let mut reader = Cursor::new(encoded.clone());
        let loaded = ProvingKey::<G1Affine>::read(
            &mut reader,
            SerdeFormat::RawBytes,
            &domain,
            &argument.get_columns(),
        )
        .expect("reload sparse permutation proving key");
        let mut roundtripped = Vec::new();
        loaded
            .write(&mut roundtripped, SerdeFormat::RawBytes)
            .expect("reserialize sparse permutation proving key");
        assert_eq!(roundtripped, encoded, "raw successor payload is canonical");
        assert_eq!(loaded.polys().len(), columns.len());

        let mut omega_powers = vec![Fr::ONE; n];
        for row in 1..n {
            omega_powers[row] = omega_powers[row - 1] * domain.get_omega();
        }
        for column in 0..columns.len() {
            let sigma = loaded
                .sigma_lagrange(&domain, column, &omega_powers)
                .expect("rehydrate native sigma in Lagrange basis");
            for row in 0..n {
                let (target_column, target_row) = dense_mapping[column][row];
                let expected = Fr::DELTA.pow_vartime([target_column as u64])
                    * omega_powers[target_row];
                assert_eq!(sigma[row], expected, "rehydrated sigma ({column}, {row})");
                assert_eq!(
                    crate::arithmetic::eval_polynomial(&loaded.polys()[column], omega_powers[row]),
                    expected,
                    "rehydrated coefficient sigma ({column}, {row})",
                );
            }
            assert_eq!(
                params
                    .commit_lagrange(&sigma, Blind::default())
                    .to_affine(),
                sparse_vk.commitments()[column],
                "rehydrated sigma KZG commitment for column {column}",
            );
        }
    }
}
