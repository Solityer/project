use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use group::ff::Field;

/// A value assigned to a cell within a circuit.
///
/// Stored as a fraction, so the backend can use batch inversion.
///
/// A denominator of zero maps to an assigned value of zero.
#[derive(Clone, Copy, Debug)]
pub enum Assigned<F> {
    /// The field element zero.
    Zero,
    /// A value that does not require inversion to evaluate.
    Trivial(F),
    /// A value stored as a fraction to enable batch inversion.
    Rational(F, F),
}

impl<F: Field> From<&Assigned<F>> for Assigned<F> {
    fn from(val: &Assigned<F>) -> Self {
        *val
    }
}

impl<F: Field> From<&F> for Assigned<F> {
    fn from(numerator: &F) -> Self {
        Assigned::Trivial(*numerator)
    }
}

impl<F: Field> From<F> for Assigned<F> {
    fn from(numerator: F) -> Self {
        Assigned::Trivial(numerator)
    }
}

impl<F: Field> From<(F, F)> for Assigned<F> {
    fn from((numerator, denominator): (F, F)) -> Self {
        Assigned::Rational(numerator, denominator)
    }
}

impl<F: Field> PartialEq for Assigned<F> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            // At least one side is directly zero.
            (Self::Zero, Self::Zero) => true,
            (Self::Zero, x) | (x, Self::Zero) => x.is_zero_vartime(),

            // One side is x/0 which maps to zero.
            (Self::Rational(_, denominator), x) | (x, Self::Rational(_, denominator))
                if denominator.is_zero_vartime() =>
            {
                x.is_zero_vartime()
            }

            // Okay, we need to do some actual math...
            (Self::Trivial(lhs), Self::Trivial(rhs)) => lhs == rhs,
            (Self::Trivial(x), Self::Rational(numerator, denominator))
            | (Self::Rational(numerator, denominator), Self::Trivial(x)) => {
                &(*x * denominator) == numerator
            }
            (
                Self::Rational(lhs_numerator, lhs_denominator),
                Self::Rational(rhs_numerator, rhs_denominator),
            ) => *lhs_numerator * rhs_denominator == *lhs_denominator * rhs_numerator,
        }
    }
}

impl<F: Field> Eq for Assigned<F> {}

impl<F: Field> Neg for Assigned<F> {
    type Output = Assigned<F>;
    fn neg(self) -> Self::Output {
        match self {
            Self::Zero => Self::Zero,
            Self::Trivial(numerator) => Self::Trivial(-numerator),
            Self::Rational(numerator, denominator) => Self::Rational(-numerator, denominator),
        }
    }
}

impl<F: Field> Neg for &Assigned<F> {
    type Output = Assigned<F>;
    fn neg(self) -> Self::Output {
        -*self
    }
}

impl<F: Field> Add for Assigned<F> {
    type Output = Assigned<F>;
    fn add(self, rhs: Assigned<F>) -> Assigned<F> {
        match (self, rhs) {
            // One side is directly zero.
            (Self::Zero, _) => rhs,
            (_, Self::Zero) => self,

            // One side is x/0 which maps to zero.
            (Self::Rational(_, denominator), other) | (other, Self::Rational(_, denominator))
                if denominator.is_zero_vartime() =>
            {
                other
            }

            // Okay, we need to do some actual math...
            (Self::Trivial(lhs), Self::Trivial(rhs)) => Self::Trivial(lhs + rhs),
            (Self::Rational(numerator, denominator), Self::Trivial(other))
            | (Self::Trivial(other), Self::Rational(numerator, denominator)) => {
                Self::Rational(numerator + denominator * other, denominator)
            }
            (
                Self::Rational(lhs_numerator, lhs_denominator),
                Self::Rational(rhs_numerator, rhs_denominator),
            ) => Self::Rational(
                lhs_numerator * rhs_denominator + lhs_denominator * rhs_numerator,
                lhs_denominator * rhs_denominator,
            ),
        }
    }
}

impl<F: Field> Add<F> for Assigned<F> {
    type Output = Assigned<F>;
    fn add(self, rhs: F) -> Assigned<F> {
        self + Self::Trivial(rhs)
    }
}

impl<F: Field> Add<F> for &Assigned<F> {
    type Output = Assigned<F>;
    fn add(self, rhs: F) -> Assigned<F> {
        *self + rhs
    }
}

impl<F: Field> Add<&Assigned<F>> for Assigned<F> {
    type Output = Assigned<F>;
    fn add(self, rhs: &Self) -> Assigned<F> {
        self + *rhs
    }
}

impl<F: Field> Add<Assigned<F>> for &Assigned<F> {
    type Output = Assigned<F>;
    fn add(self, rhs: Assigned<F>) -> Assigned<F> {
        *self + rhs
    }
}

impl<F: Field> Add<&Assigned<F>> for &Assigned<F> {
    type Output = Assigned<F>;
    fn add(self, rhs: &Assigned<F>) -> Assigned<F> {
        *self + *rhs
    }
}

impl<F: Field> AddAssign for Assigned<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Field> AddAssign<&Assigned<F>> for Assigned<F> {
    fn add_assign(&mut self, rhs: &Self) {
        *self = *self + rhs;
    }
}

impl<F: Field> Sub for Assigned<F> {
    type Output = Assigned<F>;
    fn sub(self, rhs: Assigned<F>) -> Assigned<F> {
        self + (-rhs)
    }
}

impl<F: Field> Sub<F> for Assigned<F> {
    type Output = Assigned<F>;
    fn sub(self, rhs: F) -> Assigned<F> {
        self + (-rhs)
    }
}

impl<F: Field> Sub<F> for &Assigned<F> {
    type Output = Assigned<F>;
    fn sub(self, rhs: F) -> Assigned<F> {
        *self - rhs
    }
}

impl<F: Field> Sub<&Assigned<F>> for Assigned<F> {
    type Output = Assigned<F>;
    fn sub(self, rhs: &Self) -> Assigned<F> {
        self - *rhs
    }
}

impl<F: Field> Sub<Assigned<F>> for &Assigned<F> {
    type Output = Assigned<F>;
    fn sub(self, rhs: Assigned<F>) -> Assigned<F> {
        *self - rhs
    }
}

impl<F: Field> Sub<&Assigned<F>> for &Assigned<F> {
    type Output = Assigned<F>;
    fn sub(self, rhs: &Assigned<F>) -> Assigned<F> {
        *self - *rhs
    }
}

impl<F: Field> SubAssign for Assigned<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Field> SubAssign<&Assigned<F>> for Assigned<F> {
    fn sub_assign(&mut self, rhs: &Self) {
        *self = *self - rhs;
    }
}

impl<F: Field> Mul for Assigned<F> {
    type Output = Assigned<F>;
    fn mul(self, rhs: Assigned<F>) -> Assigned<F> {
        match (self, rhs) {
            (Self::Zero, _) | (_, Self::Zero) => Self::Zero,
            (Self::Trivial(lhs), Self::Trivial(rhs)) => Self::Trivial(lhs * rhs),
            (Self::Rational(numerator, denominator), Self::Trivial(other))
            | (Self::Trivial(other), Self::Rational(numerator, denominator)) => {
                Self::Rational(numerator * other, denominator)
            }
            (
                Self::Rational(lhs_numerator, lhs_denominator),
                Self::Rational(rhs_numerator, rhs_denominator),
            ) => Self::Rational(
                lhs_numerator * rhs_numerator,
                lhs_denominator * rhs_denominator,
            ),
        }
    }
}

impl<F: Field> Mul<F> for Assigned<F> {
    type Output = Assigned<F>;
    fn mul(self, rhs: F) -> Assigned<F> {
        self * Self::Trivial(rhs)
    }
}

impl<F: Field> Mul<F> for &Assigned<F> {
    type Output = Assigned<F>;
    fn mul(self, rhs: F) -> Assigned<F> {
        *self * rhs
    }
}

impl<F: Field> Mul<&Assigned<F>> for Assigned<F> {
    type Output = Assigned<F>;
    fn mul(self, rhs: &Assigned<F>) -> Assigned<F> {
        self * *rhs
    }
}

impl<F: Field> MulAssign for Assigned<F> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Field> MulAssign<&Assigned<F>> for Assigned<F> {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = *self * rhs;
    }
}

impl<F: Field> Assigned<F> {
    /// Returns the numerator.
    pub fn numerator(&self) -> F {
        match self {
            Self::Zero => F::ZERO,
            Self::Trivial(x) => *x,
            Self::Rational(numerator, _) => *numerator,
        }
    }

    /// Returns the denominator, if non-trivial.
    pub fn denominator(&self) -> Option<F> {
        match self {
            Self::Zero => None,
            Self::Trivial(_) => None,
            Self::Rational(_, denominator) => Some(*denominator),
        }
    }

    /// Returns true iff this element is zero.
    pub fn is_zero_vartime(&self) -> bool {
        match self {
            Self::Zero => true,
            Self::Trivial(x) => x.is_zero_vartime(),
            // Assigned maps x/0 -> 0.
            Self::Rational(numerator, denominator) => {
                numerator.is_zero_vartime() || denominator.is_zero_vartime()
            }
        }
    }

    /// Doubles this element.
    #[must_use]
    pub fn double(&self) -> Self {
        match self {
            Self::Zero => Self::Zero,
            Self::Trivial(x) => Self::Trivial(x.double()),
            Self::Rational(numerator, denominator) => {
                Self::Rational(numerator.double(), *denominator)
            }
        }
    }

    /// Squares this element.
    #[must_use]
    pub fn square(&self) -> Self {
        match self {
            Self::Zero => Self::Zero,
            Self::Trivial(x) => Self::Trivial(x.square()),
            Self::Rational(numerator, denominator) => {
                Self::Rational(numerator.square(), denominator.square())
            }
        }
    }

    /// Cubes this element.
    #[must_use]
    pub fn cube(&self) -> Self {
        self.square() * self
    }

    /// Inverts this assigned value (taking the inverse of zero to be zero).
    pub fn invert(&self) -> Self {
        match self {
            Self::Zero => Self::Zero,
            Self::Trivial(x) => Self::Rational(F::ONE, *x),
            Self::Rational(numerator, denominator) => Self::Rational(*denominator, *numerator),
        }
    }

    /// Evaluates this assigned value directly, performing an unbatched inversion if
    /// necessary.
    ///
    /// If the denominator is zero, this returns zero.
    pub fn evaluate(self) -> F {
        match self {
            Self::Zero => F::ZERO,
            Self::Trivial(x) => x,
            Self::Rational(numerator, denominator) => {
                if denominator == F::ONE {
                    numerator
                } else {
                    numerator * denominator.invert().unwrap_or(F::ZERO)
                }
            }
        }
    }
}
