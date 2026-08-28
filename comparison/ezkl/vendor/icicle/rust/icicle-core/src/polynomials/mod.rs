use crate::traits::{FieldConfig, FieldImpl};
use icicle_runtime::memory::HostOrDeviceSlice;

pub trait UnivariatePolynomial: Clone + Sized
where
    Self::Field: FieldImpl,
    Self::FieldConfig: FieldConfig,
{
    type Field;
    type FieldConfig;

    fn from_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(coeffs: &S, size: usize) -> Self;
    fn from_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(evals: &S, size: usize) -> Self;
    fn divide(&self, denominator: &Self) -> (Self, Self)
    where
        Self: Sized;
    fn div_by_vanishing(&self, degree: u64) -> Self;
    fn add_monomial_inplace(&mut self, monomial_coeff: &Self::Field, monomial: u64);
    fn sub_monomial_inplace(&mut self, monomial_coeff: &Self::Field, monomial: u64);
    fn slice(&self, offset: u64, stride: u64, size: u64) -> Self;
    fn even(&self) -> Self;
    fn odd(&self) -> Self;
    fn eval(&self, x: &Self::Field) -> Self::Field;
    fn degree(&self) -> i64;
    fn eval_on_domain<
        D: HostOrDeviceSlice<Self::Field> + ?Sized,
        E: HostOrDeviceSlice<Self::Field> + ?Sized,
    >(
        &self,
        domain: &D,
        evals: &mut E,
    );
    fn eval_on_rou_domain<E: HostOrDeviceSlice<Self::Field> + ?Sized>(
        &self,
        domain_log_size: u64,
        evals: &mut E,
    );
    fn get_nof_coeffs(&self) -> u64;
    fn get_coeff(&self, idx: u64) -> Self::Field;
    fn copy_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(
        &self,
        start_idx: u64,
        coeffs: &mut S,
    );
    fn add(&self, rhs: &Self) -> Self;
    fn add_assign(&mut self, rhs: &Self);
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
    fn mul_by_scalar(&self, rhs: &Self::Field) -> Self;
}

#[macro_export]
macro_rules! impl_univariate_polynomial_api {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_cfg:ident
    ) => {
        use icicle_core::{polynomials::UnivariatePolynomial, traits::FieldImpl};
        use icicle_runtime::memory::{DeviceSlice, HostOrDeviceSlice};
        use std::{
            clone, cmp,
            ffi::c_void,
            ops::{Add, AddAssign, Div, Mul, Rem, Sub},
            ptr, slice,
        };

        type PolynomialHandle = *const c_void;

        extern "C" {
            #[link_name = concat!($field_prefix, "_polynomial_create_from_coefficients")]
            fn create_from_coeffs(coeffs: *const $field, size: usize) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_create_from_rou_evaluations")]
            fn create_from_rou_evals(coeffs: *const $field, size: usize) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_clone")]
            fn clone(p: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_delete")]
            fn delete(ptr: PolynomialHandle);

            #[link_name = concat!($field_prefix, "_polynomial_print")]
            fn print(ptr: PolynomialHandle);

            #[link_name = concat!($field_prefix, "_polynomial_add")]
            fn add(a: PolynomialHandle, b: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_add_inplace")]
            fn add_inplace(a: PolynomialHandle, b: PolynomialHandle) -> c_void;

            #[link_name = concat!($field_prefix, "_polynomial_subtract")]
            fn subtract(a: PolynomialHandle, b: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_multiply")]
            fn multiply(a: PolynomialHandle, b: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_multiply_by_scalar")]
            fn multiply_by_scalar(a: PolynomialHandle, b: &$field) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_quotient")]
            fn quotient(a: PolynomialHandle, b: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_remainder")]
            fn remainder(a: PolynomialHandle, b: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_division")]
            fn divide(
                a: PolynomialHandle,
                b: PolynomialHandle,
                q: *mut PolynomialHandle,
                r: *mut PolynomialHandle,
            );

            #[link_name = concat!($field_prefix, "_polynomial_divide_by_vanishing")]
            fn div_by_vanishing(a: PolynomialHandle, deg: u64) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_add_monomial_inplace")]
            fn add_monomial_inplace(
                a: PolynomialHandle,
                monomial_coeff: &$field,
                monomial: u64,
            ) -> c_void;

            #[link_name = concat!($field_prefix, "_polynomial_sub_monomial_inplace")]
            fn sub_monomial_inplace(
                a: PolynomialHandle,
                monomial_coeff: &$field,
                monomial: u64,
            ) -> c_void;

            #[link_name = concat!($field_prefix, "_polynomial_slice")]
            fn slice(a: PolynomialHandle, offset: u64, stride: u64, size: u64) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_even")]
            fn even(a: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_odd")]
            fn odd(a: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_evaluate_on_domain")]
            fn eval_on_domain(
                a: PolynomialHandle,
                domain: *const $field,
                domain_size: u64,
                evals: *mut $field,
            );

            #[link_name = concat!($field_prefix, "_polynomial_evaluate_on_rou_domain")]
            fn eval_on_rou_domain(a: PolynomialHandle, domain_log_size: u64, evals: *mut $field);

            #[link_name = concat!($field_prefix, "_polynomial_degree")]
            fn degree(a: PolynomialHandle) -> i64;

            #[link_name = concat!($field_prefix, "_polynomial_copy_coeffs_range")]
            fn copy_coeffs(
                a: PolynomialHandle,
                host_coeffs: *mut $field,
                start_idx: u64,
                end_idx: u64,
            ) -> u64;

            #[link_name = concat!($field_prefix, "_polynomial_get_coeffs_raw_ptr")]
            fn get_coeffs_ptr(a: PolynomialHandle, len: *mut u64) -> *mut $field;
        }

        pub struct DensePolynomial {
            handle: PolynomialHandle,
        }

        impl DensePolynomial {
            // TODO Yuval: implement Display trait
            pub fn print(&self) {
                unsafe {
                    print(self.handle);
                }
            }

            pub fn coeffs_mut_slice(&mut self) -> &mut DeviceSlice<$field> {
                unsafe {
                    let mut len: u64 = 0;
                    let mut coeffs_mut = get_coeffs_ptr(self.handle, &mut len);
                    let s = slice::from_raw_parts_mut(coeffs_mut, len as usize);
                    DeviceSlice::from_mut_slice(s)
                }
            }
        }

        impl UnivariatePolynomial for DensePolynomial {
            type Field = $field;
            type FieldConfig = $field_cfg;

            fn from_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(
                coeffs: &S,
                size: usize,
            ) -> Self {
                unsafe {
                    DensePolynomial {
                        handle: create_from_coeffs(coeffs.as_ptr(), size),
                    }
                }
            }

            fn from_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(
                evals: &S,
                size: usize,
            ) -> Self {
                unsafe {
                    Self {
                        handle: create_from_rou_evals(evals.as_ptr(), size),
                    }
                }
            }

            fn divide(&self, denominator: &Self) -> (Self, Self) {
                let mut q_handle: PolynomialHandle = std::ptr::null_mut();
                let mut r_handle: PolynomialHandle = std::ptr::null_mut();
                unsafe {
                    divide(
                        self.handle,
                        denominator.handle,
                        &mut q_handle,
                        &mut r_handle,
                    );
                }
                (Self { handle: q_handle }, Self { handle: r_handle })
            }

            fn div_by_vanishing(&self, degree: u64) -> Self {
                unsafe {
                    Self {
                        handle: div_by_vanishing(self.handle, degree),
                    }
                }
            }

            fn add_monomial_inplace(&mut self, monomial_coeff: &Self::Field, monomial: u64) {
                unsafe {
                    add_monomial_inplace(self.handle, monomial_coeff, monomial);
                }
            }

            fn sub_monomial_inplace(&mut self, monomial_coeff: &Self::Field, monomial: u64) {
                unsafe {
                    sub_monomial_inplace(self.handle, monomial_coeff, monomial);
                }
            }

            fn slice(&self, offset: u64, stride: u64, size: u64) -> Self {
                unsafe {
                    Self {
                        handle: slice(self.handle, offset, stride, size),
                    }
                }
            }

            fn even(&self) -> Self {
                unsafe {
                    Self {
                        handle: even(self.handle),
                    }
                }
            }

            fn odd(&self) -> Self {
                unsafe {
                    Self {
                        handle: odd(self.handle),
                    }
                }
            }

            fn eval(&self, x: &Self::Field) -> Self::Field {
                let mut eval = Self::Field::zero();
                unsafe {
                    eval_on_domain(self.handle, x, 1, &mut eval);
                }
                eval
            }

            fn eval_on_domain<
                D: HostOrDeviceSlice<Self::Field> + ?Sized,
                E: HostOrDeviceSlice<Self::Field> + ?Sized,
            >(
                &self,
                domain: &D,
                evals: &mut E,
            ) {
                assert!(
                    domain.len() <= evals.len(),
                    "eval_on_domain(): eval size must not be smaller then domain"
                );
                unsafe {
                    eval_on_domain(
                        self.handle,
                        domain.as_ptr(),
                        domain.len() as u64,
                        evals.as_mut_ptr(),
                    );
                }
            }

            fn eval_on_rou_domain<E: HostOrDeviceSlice<Self::Field> + ?Sized>(
                &self,
                domain_log_size: u64,
                evals: &mut E,
            ) {
                assert!(
                    evals.len() >= 1 << domain_log_size,
                    "eval_on_rou_domain(): eval size must not be smaller than domain"
                );
                unsafe {
                    eval_on_rou_domain(self.handle, domain_log_size, evals.as_mut_ptr());
                }
            }

            fn get_nof_coeffs(&self) -> u64 {
                unsafe {
                    // returns total #coeffs. Not copying when null
                    let nof_coeffs = copy_coeffs(self.handle, std::ptr::null_mut(), 0, 0);
                    nof_coeffs
                }
            }

            fn get_coeff(&self, idx: u64) -> Self::Field {
                let mut coeff: Self::Field = Self::Field::zero();
                unsafe { copy_coeffs(self.handle, &mut coeff, idx, idx) };
                coeff
            }

            fn copy_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(
                &self,
                start_idx: u64,
                coeffs: &mut S,
            ) {
                let coeffs_len = coeffs.len() as u64;
                let nof_coeffs = self.get_nof_coeffs();
                let end_idx = cmp::min(nof_coeffs - 1, start_idx + coeffs_len - 1);

                unsafe {
                    copy_coeffs(self.handle, coeffs.as_mut_ptr(), start_idx, end_idx);
                }
            }

            fn degree(&self) -> i64 {
                unsafe { degree(self.handle) }
            }

            fn add(&self, rhs: &Self) -> Self {
                unsafe {
                    Self {
                        handle: add(self.handle, rhs.handle),
                    }
                }
            }

            fn add_assign(&mut self, rhs: &Self) {
                unsafe { add_inplace(self.handle, rhs.handle) };
            }

            fn sub(&self, rhs: &Self) -> Self {
                unsafe {
                    Self {
                        handle: subtract(self.handle, rhs.handle),
                    }
                }
            }

            fn mul(&self, rhs: &Self) -> Self {
                unsafe {
                    Self {
                        handle: multiply(self.handle, rhs.handle),
                    }
                }
            }

            //poly * scalar
            fn mul_by_scalar(&self, rhs: &Self::Field) -> Self {
                unsafe {
                    Self {
                        handle: multiply_by_scalar(self.handle, rhs),
                    }
                }
            }
        }

        impl Drop for DensePolynomial {
            fn drop(&mut self) {
                unsafe {
                    delete(self.handle);
                }
            }
        }

        impl Clone for DensePolynomial {
            fn clone(&self) -> Self {
                unsafe {
                    DensePolynomial {
                        handle: clone(self.handle),
                    }
                }
            }
        }

        impl Add for &DensePolynomial {
            type Output = DensePolynomial;

            fn add(self: Self, rhs: Self) -> Self::Output {
                unsafe {
                    DensePolynomial {
                        handle: add(self.handle, rhs.handle),
                    }
                }
            }
        }

        impl AddAssign<&DensePolynomial> for DensePolynomial {
            fn add_assign(&mut self, other: &DensePolynomial) {
                unsafe { add_inplace(self.handle, other.handle) };
            }
        }

        impl Sub for &DensePolynomial {
            type Output = DensePolynomial;

            fn sub(self: Self, rhs: Self) -> Self::Output {
                unsafe {
                    DensePolynomial {
                        handle: subtract(self.handle, rhs.handle),
                    }
                }
            }
        }

        impl Mul for &DensePolynomial {
            type Output = DensePolynomial;

            fn mul(self: Self, rhs: Self) -> Self::Output {
                unsafe {
                    DensePolynomial {
                        handle: multiply(self.handle, rhs.handle),
                    }
                }
            }
        }

        // poly * scalar
        impl Mul<&$field> for &DensePolynomial {
            type Output = DensePolynomial;

            fn mul(self: Self, rhs: &$field) -> Self::Output {
                unsafe {
                    DensePolynomial {
                        handle: multiply_by_scalar(self.handle, rhs),
                    }
                }
            }
        }

        // scalar * poly
        impl Mul<&DensePolynomial> for &$field {
            type Output = DensePolynomial;

            fn mul(self, rhs: &DensePolynomial) -> Self::Output {
                unsafe {
                    DensePolynomial {
                        handle: multiply_by_scalar(rhs.handle, self),
                    }
                }
            }
        }

        impl Div for &DensePolynomial {
            type Output = DensePolynomial;

            fn div(self: Self, rhs: Self) -> Self::Output {
                unsafe {
                    DensePolynomial {
                        handle: quotient(self.handle, rhs.handle),
                    }
                }
            }
        }

        impl Rem for &DensePolynomial {
            type Output = DensePolynomial;

            fn rem(self: Self, rhs: Self) -> Self::Output {
                unsafe {
                    DensePolynomial {
                        handle: remainder(self.handle, rhs.handle),
                    }
                }
            }
        }
    };
}
