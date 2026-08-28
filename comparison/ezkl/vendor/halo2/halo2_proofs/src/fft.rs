//! This is a module for dispatching between different FFT implementations at runtime based on environment variable `FFT`.

use ff::Field;

use self::recursive::FFTData;
use crate::arithmetic::FftGroup;

pub mod baseline;
pub mod parallel;
pub mod recursive;

/// Runtime dispatcher to concrete FFT implementation
pub fn fft<Scalar: Field, G: FftGroup<Scalar>>(
    a: &mut [G],
    omega: Scalar,
    log_n: u32,
    data: &FFTData<Scalar>,
    inverse: bool,
) {
    // Empirically, the parallel implementation requires less memory bandwidth, which is more performant on x86_64.
    #[cfg(target_arch = "x86_64")]
    parallel::fft(a, omega, log_n, data, inverse);
    #[cfg(not(target_arch = "x86_64"))]
    recursive::fft(a, omega, log_n, data, inverse)
}
