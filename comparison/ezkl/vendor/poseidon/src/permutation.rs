use halo2curves::group::ff::PrimeField;

use crate::spec::{Spec, State};

impl<F: PrimeField, const T: usize, const RATE: usize> Spec<F, T, RATE> {
    /// Applies the Poseidon permutation to the given state
    pub fn permute(&self, state: &mut State<F, T>) {
        let r_f = self.r_f / 2;

        // First half of the full rounds
        {
            state.add_constants(&self.constants.start[0]);
            for round_constants in self.constants.start.iter().skip(1).take(r_f - 1) {
                state.sbox_full();
                state.add_constants(round_constants);
                self.mds_matrices.mds.apply(state);
            }
            state.sbox_full();
            state.add_constants(self.constants.start.last().unwrap());
            self.mds_matrices.pre_sparse_mds.apply(state)
        }

        // Partial rounds
        {
            for (round_constant, sparse_mds) in self
                .constants
                .partial
                .iter()
                .zip(self.mds_matrices.sparse_matrices.iter())
            {
                state.sbox_part();
                state.add_constant(round_constant);
                sparse_mds.apply(state);
            }
        }

        // Second half of the full rounds
        {
            for round_constants in self.constants.end.iter() {
                state.sbox_full();
                state.add_constants(round_constants);
                self.mds_matrices.mds.apply(state);
            }
            state.sbox_full();
            self.mds_matrices.mds.apply(state);
        }
    }
}
