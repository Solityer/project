use anyhow::{bail, ensure, Result};
use blst::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct Scalar(pub [u8; 32]);

impl Serialize for Scalar {
    fn serialize<S:serde::Serializer>(&self,serializer:S)->std::result::Result<S::Ok,S::Error>{serializer.serialize_bytes(&self.0)}
}
impl<'de> Deserialize<'de> for Scalar {
    fn deserialize<D:serde::Deserializer<'de>>(deserializer:D)->std::result::Result<Self,D::Error>{
        let bytes:Vec<u8>=Deserialize::deserialize(deserializer)?;if bytes.len()!=32{return Err(serde::de::Error::custom("BLS12-381 scalar must be 32 bytes"));}
        let mut canonical=[0u8;32];canonical.copy_from_slice(&bytes);let mut raw=blst_scalar::default();unsafe{blst_scalar_from_be_bytes(&mut raw,canonical.as_ptr(),32)};let mut roundtrip=[0u8;32];unsafe{blst_bendian_from_scalar(roundtrip.as_mut_ptr(),&raw)};if roundtrip!=canonical{return Err(serde::de::Error::custom("non-canonical BLS12-381 scalar"));}Ok(Self(canonical))
    }
}

impl fmt::Debug for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{}", hex::encode(self.0))
    }
}

impl Scalar {
    pub fn zero() -> Self { Self([0; 32]) }
    pub fn one() -> Self { Self::from_u64(1) }
    pub fn from_u64(value: u64) -> Self {
        let words = [value, 0, 0, 0];
        let mut fr = blst_fr::default();
        unsafe { blst_fr_from_uint64(&mut fr, words.as_ptr()) };
        Self::from_fr(fr)
    }
    pub fn from_i64(value: i64) -> Self {
        if value >= 0 { return Self::from_u64(value as u64); }
        -Self::from_u64(value.unsigned_abs())
    }
    pub fn from_canonical_bytes(canonical: [u8; 32]) -> Result<Self> {
        let mut raw = blst_scalar::default();
        unsafe { blst_scalar_from_be_bytes(&mut raw, canonical.as_ptr(), canonical.len()) };
        let mut roundtrip = [0u8; 32];
        unsafe { blst_bendian_from_scalar(roundtrip.as_mut_ptr(), &raw) };
        ensure!(roundtrip == canonical, "non-canonical BLS12-381 scalar returned by backend");
        Ok(Self(canonical))
    }
    pub fn from_be_hex(value:&str)->Result<Self>{
        let text=value.strip_prefix("0x").unwrap_or(value);let bytes=hex::decode(text)?;ensure!(bytes.len()<=32,"scalar hex exceeds 32 bytes");let mut canonical=[0u8;32];canonical[32-bytes.len()..].copy_from_slice(&bytes);let mut raw=blst_scalar::default();unsafe{blst_scalar_from_be_bytes(&mut raw,canonical.as_ptr(),32)};let mut roundtrip=[0u8;32];unsafe{blst_bendian_from_scalar(roundtrip.as_mut_ptr(),&raw)};ensure!(roundtrip==canonical,"non-canonical BLS12-381 scalar");Ok(Self(canonical))
    }
    pub fn random() -> Result<Self> {
        let mut wide = [0u8; 64];
        getrandom::getrandom(&mut wide).map_err(|e| anyhow::anyhow!("OS randomness failed: {e}"))?;
        Ok(Self::hash_to_field(b"zkDL/random-scalar/v1", &wide))
    }
    pub fn hash_to_field(domain: &[u8], message: &[u8]) -> Self {
        let mut h0 = Sha256::new();
        h0.update(domain);
        h0.update([0]);
        h0.update(message);
        h0.update([0]);
        let a = h0.finalize();
        let mut h1 = Sha256::new();
        h1.update(domain);
        h1.update([0]);
        h1.update(message);
        h1.update([1]);
        let b = h1.finalize();
        let mut wide = [0u8; 64];
        wide[..32].copy_from_slice(&a);
        wide[32..].copy_from_slice(&b);
        // blst accepts a wide big-endian input and returns a canonical scalar.
        let mut raw = blst_scalar::default();
        let ok = unsafe { blst_scalar_from_be_bytes(&mut raw, wide.as_ptr(), wide.len()) };
        if !ok {
            // The return value means the reduced scalar is zero. Domain-separated
            // rejection sampling avoids a zero Fiat--Shamir challenge.
            for counter in 0u32.. {
                let mut h = Sha256::new();
                h.update(domain);
                h.update([0]);
                h.update(message);
                h.update(counter.to_be_bytes());
                let candidate = h.finalize();
                if unsafe { blst_scalar_from_be_bytes(&mut raw, candidate.as_ptr(), 32) } {
                    break;
                }
            }
        }
        let mut fr = blst_fr::default();
        unsafe { blst_fr_from_scalar(&mut fr, &raw) };
        Self::from_fr(fr)
    }
    fn to_fr(self) -> blst_fr {
        let mut raw = blst_scalar::default();
        unsafe { blst_scalar_from_be_bytes(&mut raw, self.0.as_ptr(), 32) };
        let mut canonical=[0u8;32]; unsafe{blst_bendian_from_scalar(canonical.as_mut_ptr(),&raw)};
        debug_assert_eq!(canonical,self.0,"canonical scalar invariant");
        let mut fr = blst_fr::default();
        unsafe { blst_fr_from_scalar(&mut fr, &raw) };
        fr
    }
    fn from_fr(fr: blst_fr) -> Self {
        let mut raw = blst_scalar::default();
        unsafe { blst_scalar_from_fr(&mut raw, &fr) };
        let mut canonical_be = [0u8; 32];
        unsafe { blst_bendian_from_scalar(canonical_be.as_mut_ptr(), &raw) };
        Self(canonical_be)
    }
    pub fn inverse(self) -> Result<Self> {
        ensure!(self != Self::zero(), "attempted field inversion of zero");
        let a = self.to_fr();
        let mut out = blst_fr::default();
        unsafe { blst_fr_inverse(&mut out, &a) };
        Ok(Self::from_fr(out))
    }
    pub fn pow_u64(self, mut e: u64) -> Self {
        let mut base = self;
        let mut out = Self::one();
        while e != 0 {
            if e & 1 == 1 { out = out * base; }
            e >>= 1;
            if e != 0 { base = base * base; }
        }
        out
    }
}

impl std::ops::Add for Scalar { type Output = Self; fn add(self, rhs: Self) -> Self {
    let (a,b)=(self.to_fr(),rhs.to_fr()); let mut o=blst_fr::default(); unsafe{blst_fr_add(&mut o,&a,&b)}; Self::from_fr(o)
}}
impl std::ops::Sub for Scalar { type Output = Self; fn sub(self, rhs: Self) -> Self {
    let (a,b)=(self.to_fr(),rhs.to_fr()); let mut o=blst_fr::default(); unsafe{blst_fr_sub(&mut o,&a,&b)}; Self::from_fr(o)
}}
impl std::ops::Mul for Scalar { type Output = Self; fn mul(self, rhs: Self) -> Self {
    let (a,b)=(self.to_fr(),rhs.to_fr()); let mut o=blst_fr::default(); unsafe{blst_fr_mul(&mut o,&a,&b)}; Self::from_fr(o)
}}
impl std::ops::Neg for Scalar { type Output = Self; fn neg(self) -> Self {
    let a=self.to_fr(); let mut o=blst_fr::default(); unsafe{blst_fr_cneg(&mut o,&a,true)}; Self::from_fr(o)
}}
impl std::ops::AddAssign for Scalar { fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; } }
impl std::ops::SubAssign for Scalar { fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; } }
impl std::ops::MulAssign for Scalar { fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; } }

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Point(pub [u8; 48]);

impl Serialize for Point {
    fn serialize<S: serde::Serializer>(&self, serializer:S)->std::result::Result<S::Ok,S::Error>{serializer.serialize_bytes(&self.0)}
}
impl<'de> Deserialize<'de> for Point {
    fn deserialize<D:serde::Deserializer<'de>>(deserializer:D)->std::result::Result<Self,D::Error>{
        let bytes:Vec<u8>=Deserialize::deserialize(deserializer)?;
        if bytes.len()!=48{return Err(serde::de::Error::custom("BLS12-381 G1 encoding must be 48 bytes"));}
        let mut out=[0u8;48];out.copy_from_slice(&bytes);Ok(Self(out))
    }
}

impl fmt::Debug for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "0x{}", hex::encode(self.0)) }
}

impl Point {
    pub fn identity() -> Self {
        let p = blst_p1::default();
        Self::from_p1(&p)
    }
    pub fn hash_to_curve(domain: &[u8], message: &[u8]) -> Self {
        let mut p = blst_p1::default();
        unsafe { blst_hash_to_g1(&mut p, message.as_ptr(), message.len(), domain.as_ptr(), domain.len(), std::ptr::null(), 0) };
        Self::from_p1(&p)
    }
    fn from_p1(p: &blst_p1) -> Self {
        let mut out = [0u8; 48];
        unsafe { blst_p1_compress(out.as_mut_ptr(), p) };
        Self(out)
    }
    pub(crate) fn to_raw_p1(self) -> Result<[u8; 144]> {
        let point = self.to_p1()?;
        ensure!(std::mem::size_of::<blst_p1>() == 144, "unexpected blst_p1 ABI");
        let mut raw = [0u8; 144];
        unsafe { std::ptr::copy_nonoverlapping((&point as *const blst_p1).cast::<u8>(), raw.as_mut_ptr(), raw.len()) };
        Ok(raw)
    }
    pub(crate) fn from_raw_p1(raw: &[u8; 144]) -> Result<Self> {
        ensure!(std::mem::size_of::<blst_p1>() == raw.len(), "unexpected blst_p1 ABI");
        let mut point = blst_p1::default();
        unsafe { std::ptr::copy_nonoverlapping(raw.as_ptr(), (&mut point as *mut blst_p1).cast::<u8>(), raw.len()) };
        let encoded = Self::from_p1(&point);
        // Round-trip through the canonical compressed parser to reject any
        // malformed/off-curve/subgroup-invalid result returned by the device.
        encoded.to_p1()?;
        Ok(encoded)
    }
    fn to_p1(self) -> Result<blst_p1> {
        let mut affine = blst_p1_affine::default();
        let err = unsafe { blst_p1_uncompress(&mut affine, self.0.as_ptr()) };
        if err != BLST_ERROR::BLST_SUCCESS { bail!("non-canonical compressed BLS12-381 G1 point"); }
        ensure!(unsafe { blst_p1_affine_on_curve(&affine) }, "G1 point not on curve");
        ensure!(unsafe { blst_p1_affine_in_g1(&affine) }, "G1 point not in prime subgroup");
        let mut out = blst_p1::default();
        unsafe { blst_p1_from_affine(&mut out, &affine) };
        Ok(out)
    }
    pub fn checked(self) -> Result<Self> {
        let p = self.to_p1()?;
        ensure!(!unsafe { blst_p1_is_inf(&p) }, "identity is forbidden in commitment parameters");
        Ok(self)
    }
    pub fn add(self, rhs: Self) -> Result<Self> {
        let (a,b)=(self.to_p1()?,rhs.to_p1()?); let mut o=blst_p1::default(); unsafe{blst_p1_add_or_double(&mut o,&a,&b)}; Ok(Self::from_p1(&o))
    }
    pub fn neg(self) -> Result<Self> {
        let mut p=self.to_p1()?; unsafe{blst_p1_cneg(&mut p,true)}; Ok(Self::from_p1(&p))
    }
    pub fn sub(self, rhs: Self) -> Result<Self> { self.add(rhs.neg()?) }
    pub fn mul(self, scalar: Scalar) -> Result<Self> {
        let p=self.to_p1()?;
        let mut raw=blst_scalar::default();
        unsafe { blst_scalar_from_be_bytes(&mut raw, scalar.0.as_ptr(), 32) };
        let mut canonical=[0u8;32];unsafe{blst_bendian_from_scalar(canonical.as_mut_ptr(),&raw)};ensure!(canonical==scalar.0,"non-canonical scalar");
        let mut o=blst_p1::default(); unsafe{blst_p1_mult(&mut o,&p,raw.b.as_ptr(),255)}; Ok(Self::from_p1(&o))
    }
    pub fn eq_checked(self, rhs: Self) -> Result<bool> { Ok(unsafe{blst_p1_is_equal(&self.to_p1()?,&rhs.to_p1()?)} ) }
}

pub fn msm(points: &[Point], scalars: &[Scalar]) -> Result<Point> {
    ensure!(points.len() == scalars.len(), "MSM length mismatch");
    let mut out = Point::identity();
    for (&p, &s) in points.iter().zip(scalars) { out = out.add(p.mul(s)?)?; }
    Ok(out)
}

pub fn inner_product(a: &[Scalar], b: &[Scalar]) -> Result<Scalar> {
    ensure!(a.len()==b.len(), "inner product length mismatch");
    Ok(a.iter().zip(b).fold(Scalar::zero(), |acc,(&x,&y)| acc + x*y))
}

pub fn interpolate_at(values: &[Scalar], x: Scalar) -> Result<Scalar> {
    ensure!(!values.is_empty(), "cannot interpolate empty polynomial");
    let mut out=Scalar::zero();
    for (i,&yi) in values.iter().enumerate() {
        let xi=Scalar::from_u64(i as u64); let mut num=Scalar::one(); let mut den=Scalar::one();
        for j in 0..values.len() { if i!=j { let xj=Scalar::from_u64(j as u64); num*=x-xj; den*=xi-xj; } }
        out += yi*num*den.inverse()?;
    }
    Ok(out)
}

pub fn eq_weights(point: &[Scalar]) -> Vec<Scalar> {
    let mut out=vec![Scalar::one()];
    for &r in point {
        let old=out; out=Vec::with_capacity(old.len()*2);
        for &v in &old { out.push(v*(Scalar::one()-r)); }
        for &v in &old { out.push(v*r); }
    }
    out
}
