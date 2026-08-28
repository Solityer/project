use crate::bls12381::fq::Fq;
use crate::bls12381::fq2::Fq2;
use crate::bls12381::fr::Fr;
use crate::ff::WithSmallOrderMulGroup;
use crate::ff::{Field, PrimeField};
use crate::ff_ext::ExtField;
use crate::group::Curve;
use crate::group::{cofactor::CofactorGroup, prime::PrimeCurveAffine, Group, GroupEncoding};
use crate::serde::{Compressed, CompressedFlagConfig};
use crate::{
    impl_binops_additive, impl_binops_additive_specify_output, impl_binops_multiplicative,
    impl_binops_multiplicative_mixed, new_curve_impl,
};
use crate::{Coordinates, CurveAffine, CurveExt};
use core::cmp;
use core::fmt::Debug;
use core::iter::Sum;
use core::ops::{Add, Mul, Neg, Sub};
use rand::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

const G2_GENERATOR_X: Fq2 = Fq2 {
    c0: Fq([
        0xf5f2_8fa2_0294_0a10,
        0xb3f5_fb26_87b4_961a,
        0xa1a8_93b5_3e2a_e580,
        0x9894_999d_1a3c_aee9,
        0x6f67_b763_1863_366b,
        0x0581_9192_4350_bcd7,
    ]),
    c1: Fq([
        0xa5a9_c075_9e23_f606,
        0xaaa0_c59d_bccd_60c3,
        0x3bb1_7e18_e286_7806,
        0x1b1a_b6cc_8541_b367,
        0xc2b6_ed0e_f215_8547,
        0x1192_2a09_7360_edf3,
    ]),
};

const G2_GENERATOR_Y: Fq2 = Fq2 {
    c0: Fq([
        0x4c73_0af8_6049_4c4a,
        0x597c_fa1f_5e36_9c5a,
        0xe7e6_856c_aa0a_635a,
        0xbbef_b5e9_6e0d_495f,
        0x07d3_a975_f0ef_25a2,
        0x0083_fd8e_7e80_dae5,
    ]),
    c1: Fq([
        0xadc0_fc92_df64_b05d,
        0x18aa_270a_2b14_61dc,
        0x86ad_ac6a_3be4_eba0,
        0x7949_5c4e_c93d_a33a,
        0xe717_5850_a43c_caed,
        0x0b2b_c2a1_63de_1bf2,
    ]),
};

const G2_B: Fq2 = Fq2 {
    c0: Fq::from_raw([4, 0, 0, 0, 0, 0]),
    c1: Fq::from_raw([4, 0, 0, 0, 0, 0]),
};

const G2_A: Fq2 = Fq2::ZERO;

new_curve_impl!(
    (pub),
    G2,
    G2Affine,
    Fq2,
    Fr,
    (G2_GENERATOR_X, G2_GENERATOR_Y),
    G2_A,
    G2_B,
    "bls12381_g2",
    |domain_prefix| hash_to_curve(domain_prefix, hash_to_curve_suite(b"BLS12381G2_XMD:SHA-256_SSWU_RO_")),
    crate::serde::CompressedFlagConfig::ThreeSpare

);

impl crate::serde::endian::EndianRepr for Fq2 {
    const ENDIAN: crate::serde::endian::Endian = Fq::ENDIAN;

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> subtle::CtOption<Self> {
        Fq2::from_bytes(bytes[..Fq2::SIZE].try_into().unwrap())
    }
}

impl Compressed<G2Affine> for G2Compressed {
    const CONFIG: CompressedFlagConfig = CompressedFlagConfig::ThreeSpare;
    fn sign(c: &G2Affine) -> Choice {
        c.y.lexicographically_largest() & !c.is_identity()
    }
    fn resolve(x: Fq2, sign_set: Choice) -> CtOption<G2Affine> {
        G2Affine::y2(x).sqrt().map(|y| {
            let y = Fq2::conditional_select(&y, &-y, sign_set ^ y.lexicographically_largest());
            G2Affine { x, y }
        })
    }
}

impl group::cofactor::CofactorGroup for G2 {
    type Subgroup = G2;

    /// Clears the cofactor, using [Budroni-Pintore](https://ia.cr/2017/419).
    /// This is equivalent to multiplying by $h\_\textrm{eff} = 3(z^2 - 1) \cdot
    /// h_2$, where $h_2$ is the cofactor of $\mathbb{G}\_2$ and $z$ is the
    /// parameter of BLS12-381.
    fn clear_cofactor(&self) -> G2 {
        let t1 = self.mul_by_x(); // [x] P
        let t2 = self.psi(); // psi(P)

        self.double().psi2() // psi^2(2P)
            + (t1 + t2).mul_by_x() // psi^2(2P) + [x^2] P + [x] psi(P)
            - t1 // psi^2(2P) + [x^2 - x] P + [x] psi(P)
            - t2 // psi^2(2P) + [x^2 - x] P + [x - 1] psi(P)
            - self // psi^2(2P) + [x^2 - x - 1] P + [x - 1] psi(P)
    }

    fn into_subgroup(self) -> CtOption<Self::Subgroup> {
        CtOption::new(self, 1.into())
    }

    /// Returns true if this point is free of an $h$-torsion component, and so it
    /// exists within the $q$-order subgroup $\mathbb{G}_2$. This should always return true
    /// unless an "unchecked" API was used.
    fn is_torsion_free(&self) -> Choice {
        // Algorithm from Section 4 of https://eprint.iacr.org/2021/1130
        // Updated proof of correctness in https://eprint.iacr.org/2022/352
        //
        // Check that psi(P) == [x] P
        self.psi().ct_eq(&self.mul_by_x())
    }
}

impl G2 {
    /// Multiply `self` by `crate::BLS_X`, using double and add.

    fn mul_by_x(&self) -> G2 {
        let mut acc = G2::identity();
        for (i, b) in super::BLS_X.into_iter().enumerate() {
            (i != 0).then(|| acc = acc.double());
            (b == 1).then(|| acc += self);
        }
        acc.neg()
    }

    fn psi(&self) -> G2 {
        // 1 / ((u+1) ^ ((q-1)/3))
        let psi_coeff_x = Fq2 {
            c0: Fq::zero(),
            c1: Fq([
                0x890dc9e4867545c3,
                0x2af322533285a5d5,
                0x50880866309b7e2c,
                0xa20d1b8c7e881024,
                0x14e4f04fe2db9068,
                0x14e56d3f1564853a,
            ]),
        };
        // 1 / ((u+1) ^ (p-1)/2)
        let psi_coeff_y = Fq2 {
            c0: Fq([
                0x3e2f585da55c9ad1,
                0x4294213d86c18183,
                0x382844c88b623732,
                0x92ad2afd19103e18,
                0x1d794e4fac7cf0b9,
                0x0bd592fc7d825ec8,
            ]),
            c1: Fq([
                0x7bcfa7a25aa30fda,
                0xdc17dec12a927e7c,
                0x2f088dd86b4ebef1,
                0xd1ca2087da74d4a7,
                0x2da2596696cebc1d,
                0x0e2b7eedbbfd87d2,
            ]),
        };

        // x = frobenius(x)/((u+1)^((p-1)/3))
        let mut x = self.x;
        x.frobenius_map(1);
        x.mul_assign(&psi_coeff_x);

        // y = frobenius(y)/(u+1)^((p-1)/2)
        let mut y = self.y;
        y.frobenius_map(1);
        y.mul_assign(&psi_coeff_y);

        // z = frobenius(z)
        let mut z = self.z;
        z.frobenius_map(1);

        G2 { x, y, z }
    }

    fn psi2(&self) -> G2 {
        // 1 / 2 ^ ((q-1)/3)
        let psi2_coeff_x = Fq2 {
            c0: Fq([
                0xcd03c9e48671f071,
                0x5dab22461fcda5d2,
                0x587042afd3851b95,
                0x8eb60ebe01bacb9e,
                0x03f97d6e83d050d2,
                0x18f0206554638741,
            ]),
            c1: Fq::zero(),
        };

        G2 {
            // x = frobenius^2(x)/2^((p-1)/3); note that q^2 is the order of the field.
            x: self.x * psi2_coeff_x,
            // y = -frobenius^2(y); note that q^2 is the order of the field.
            y: self.y.neg(),
            // z = z
            z: self.z,
        }
    }
}

fn hash_to_curve_suite(domain: &[u8]) -> crate::hash_to_curve::Suite<G2, sha2::Sha256, 128> {
    const SSWU_Z: Fq2 = Fq2 {
        c0: Fq([
            0x87eb_ffff_fff9_555c,
            0x656f_ffe5_da8f_fffa,
            0x0fd0_7493_45d3_3ad2,
            0xd951_e663_0665_76f4,
            0xde29_1a3d_41e9_80d3,
            0x0815_664c_7dfe_040d,
        ]),
        c1: Fq([
            0x43f5_ffff_fffc_aaae,
            0x32b7_fff2_ed47_fffd,
            0x07e8_3a49_a2e9_9d69,
            0xeca8_f331_8332_bb7a,
            0xef14_8d1e_a0f4_c069,
            0x040a_b326_3eff_0206,
        ]),
    };

    const ISO_A: Fq2 = Fq2 {
        c0: Fq::zero(),
        c1: Fq([
            0xe53a_0000_0313_5242,
            0x0108_0c0f_def8_0285,
            0xe788_9edb_e340_f6bd,
            0x0b51_3751_2631_0601,
            0x02d6_9857_17c7_44ab,
            0x1220_b4e9_79ea_5467,
        ]),
    };

    const ISO_B: Fq2 = Fq2 {
        c0: Fq([
            0x22ea_0000_0cf8_9db2,
            0x6ec8_32df_7138_0aa4,
            0x6e1b_9440_3db5_a66e,
            0x75bf_3c53_a794_73ba,
            0x3dd3_a569_412c_0a34,
            0x125c_db5e_74dc_4fd1,
        ]),
        c1: Fq([
            0x22ea_0000_0cf8_9db2,
            0x6ec8_32df_7138_0aa4,
            0x6e1b_9440_3db5_a66e,
            0x75bf_3c53_a794_73ba,
            0x3dd3_a569_412c_0a34,
            0x125c_db5e_74dc_4fd1,
        ]),
    };
    let iso_map = crate::hash_to_curve::Iso {
        a: ISO_A,
        b: ISO_B,
        map: Box::new(iso_map),
    };

    crate::hash_to_curve::Suite::new(domain, SSWU_Z, crate::hash_to_curve::Method::SSWU(iso_map))
}

/// Maps an iso-G1 point to a G1 point.
fn iso_map(x: Fq2, y: Fq2, z: Fq2) -> G2 {
    const COEFFS: [&[Fq2]; 4] = [&ISO3_XNUM, &ISO3_XDEN, &ISO3_YNUM, &ISO3_YDEN];

    // xnum, xden, ynum, yden
    let mut mapvals = [Fq2::ZERO; 4];

    // compute powers of z
    let zsq = z.square();
    let zpows = [z, zsq, zsq * z];

    // compute map value by Horner's rule
    for idx in 0..4 {
        let coeff = COEFFS[idx];
        let clast = coeff.len() - 1;
        mapvals[idx] = coeff[clast];
        for jdx in 0..clast {
            mapvals[idx] = mapvals[idx] * x + zpows[jdx] * coeff[clast - 1 - jdx];
        }
    }

    // x denominator is order 1 less than x numerator, so we need an extra factor of z
    mapvals[1] *= z;

    // multiply result of Y map by the y-coord, y / z
    mapvals[2] *= y;
    mapvals[3] *= z;

    G2 {
        x: mapvals[0] * mapvals[3], // xnum * yden,
        y: mapvals[2] * mapvals[1], // ynum * xden,
        z: mapvals[1] * mapvals[3], // xden * yden
    }
}

#[allow(clippy::type_complexity)]
pub(crate) fn hash_to_curve<'a>(
    domain_prefix: &'a str,
    suite: crate::hash_to_curve::Suite<G2, sha2::Sha256, 128>,
) -> Box<dyn Fn(&[u8]) -> G2 + 'a> {
    Box::new(move |message| suite.hash_to_curve(domain_prefix, message).clear_cofactor())
}

/// Coefficients of the 3-isogeny x map's numerator
const ISO3_XNUM: [Fq2; 4] = [
    Fq2 {
        c0: Fq([
            0x47f6_71c7_1ce0_5e62,
            0x06dd_5707_1206_393e,
            0x7c80_cd2a_f3fd_71a2,
            0x0481_03ea_9e6c_d062,
            0xc545_16ac_c8d0_37f6,
            0x1380_8f55_0920_ea41,
        ]),
        c1: Fq([
            0x47f6_71c7_1ce0_5e62,
            0x06dd_5707_1206_393e,
            0x7c80_cd2a_f3fd_71a2,
            0x0481_03ea_9e6c_d062,
            0xc545_16ac_c8d0_37f6,
            0x1380_8f55_0920_ea41,
        ]),
    },
    Fq2 {
        c0: Fq::zero(),
        c1: Fq([
            0x5fe5_5555_554c_71d0,
            0x873f_ffdd_236a_aaa3,
            0x6a6b_4619_b26e_f918,
            0x21c2_8884_0887_4945,
            0x2836_cda7_028c_abc5,
            0x0ac7_3310_a7fd_5abd,
        ]),
    },
    Fq2 {
        c0: Fq([
            0x0a0c_5555_5559_71c3,
            0xdb0c_0010_1f9e_aaae,
            0xb1fb_2f94_1d79_7997,
            0xd396_0742_ef41_6e1c,
            0xb700_40e2_c205_56f4,
            0x149d_7861_e581_393b,
        ]),
        c1: Fq([
            0xaff2_aaaa_aaa6_38e8,
            0x439f_ffee_91b5_5551,
            0xb535_a30c_d937_7c8c,
            0x90e1_4442_0443_a4a2,
            0x941b_66d3_8146_55e2,
            0x0563_9988_53fe_ad5e,
        ]),
    },
    Fq2 {
        c0: Fq([
            0x40aa_c71c_71c7_25ed,
            0x1909_5555_7a84_e38e,
            0xd817_050a_8f41_abc3,
            0xd864_85d4_c87f_6fb1,
            0x696e_b479_f885_d059,
            0x198e_1a74_3280_02d2,
        ]),
        c1: Fq::zero(),
    },
];

/// Coefficients of the 3-isogeny x map's denominator
const ISO3_XDEN: [Fq2; 3] = [
    Fq2 {
        c0: Fq::zero(),
        c1: Fq([
            0x1f3a_ffff_ff13_ab97,
            0xf25b_fc61_1da3_ff3e,
            0xca37_57cb_3819_b208,
            0x3e64_2736_6f8c_ec18,
            0x0397_7bc8_6095_b089,
            0x04f6_9db1_3f39_a952,
        ]),
    },
    Fq2 {
        c0: Fq([
            0x4476_0000_0027_552e,
            0xdcb8_009a_4348_0020,
            0x6f7e_e9ce_4a6e_8b59,
            0xb103_30b7_c0a9_5bc6,
            0x6140_b1fc_fb1e_54b7,
            0x0381_be09_7f0b_b4e1,
        ]),
        c1: Fq([
            0x7588_ffff_ffd8_557d,
            0x41f3_ff64_6e0b_ffdf,
            0xf7b1_e8d2_ac42_6aca,
            0xb374_1acd_32db_b6f8,
            0xe9da_f5b9_482d_581f,
            0x167f_53e0_ba74_31b8,
        ]),
    },
    Fq2::one(),
];

/// Coefficients of the 3-isogeny y map's numerator
const ISO3_YNUM: [Fq2; 4] = [
    Fq2 {
        c0: Fq([
            0x96d8_f684_bdfc_77be,
            0xb530_e4f4_3b66_d0e2,
            0x184a_88ff_3796_52fd,
            0x57cb_23ec_fae8_04e1,
            0x0fd2_e39e_ada3_eba9,
            0x08c8_055e_31c5_d5c3,
        ]),
        c1: Fq([
            0x96d8_f684_bdfc_77be,
            0xb530_e4f4_3b66_d0e2,
            0x184a_88ff_3796_52fd,
            0x57cb_23ec_fae8_04e1,
            0x0fd2_e39e_ada3_eba9,
            0x08c8_055e_31c5_d5c3,
        ]),
    },
    Fq2 {
        c0: Fq::zero(),
        c1: Fq([
            0xbf0a_71c7_1c91_b406,
            0x4d6d_55d2_8b76_38fd,
            0x9d82_f98e_5f20_5aee,
            0xa27a_a27b_1d1a_18d5,
            0x02c3_b2b2_d293_8e86,
            0x0c7d_1342_0b09_807f,
        ]),
    },
    Fq2 {
        c0: Fq([
            0xd7f9_5555_5553_1c74,
            0x21cf_fff7_48da_aaa8,
            0x5a9a_d186_6c9b_be46,
            0x4870_a221_0221_d251,
            0x4a0d_b369_c0a3_2af1,
            0x02b1_ccc4_29ff_56af,
        ]),
        c1: Fq([
            0xe205_aaaa_aaac_8e37,
            0xfcdc_0007_6879_5556,
            0x0c96_011a_8a15_37dd,
            0x1c06_a963_f163_406e,
            0x010d_f44c_82a8_81e6,
            0x174f_4526_0f80_8feb,
        ]),
    },
    Fq2 {
        c0: Fq([
            0xa470_bda1_2f67_f35c,
            0xc0fe_38e2_3327_b425,
            0xc9d3_d0f2_c6f0_678d,
            0x1c55_c993_5b5a_982e,
            0x27f6_c0e2_f074_6764,
            0x117c_5e6e_28aa_9054,
        ]),
        c1: Fq::zero(),
    },
];

/// Coefficients of the 3-isogeny y map's denominator
const ISO3_YDEN: [Fq2; 4] = [
    Fq2 {
        c0: Fq([
            0x0162_ffff_fa76_5adf,
            0x8f7b_ea48_0083_fb75,
            0x561b_3c22_59e9_3611,
            0x11e1_9fc1_a9c8_75d5,
            0xca71_3efc_0036_7660,
            0x03c6_a03d_41da_1151,
        ]),
        c1: Fq([
            0x0162_ffff_fa76_5adf,
            0x8f7b_ea48_0083_fb75,
            0x561b_3c22_59e9_3611,
            0x11e1_9fc1_a9c8_75d5,
            0xca71_3efc_0036_7660,
            0x03c6_a03d_41da_1151,
        ]),
    },
    Fq2 {
        c0: Fq::zero(),
        c1: Fq([
            0x5db0_ffff_fd3b_02c5,
            0xd713_f523_58eb_fdba,
            0x5ea6_0761_a84d_161a,
            0xbb2c_75a3_4ea6_c44a,
            0x0ac6_7359_21c1_119b,
            0x0ee3_d913_bdac_fbf6,
        ]),
    },
    Fq2 {
        c0: Fq([
            0x66b1_0000_003a_ffc5,
            0xcb14_00e7_64ec_0030,
            0xa73e_5eb5_6fa5_d106,
            0x8984_c913_a0fe_09a9,
            0x11e1_0afb_78ad_7f13,
            0x0542_9d0e_3e91_8f52,
        ]),
        c1: Fq([
            0x534d_ffff_ffc4_aae6,
            0x5397_ff17_4c67_ffcf,
            0xbff2_73eb_870b_251d,
            0xdaf2_8271_5287_0915,
            0x393a_9cba_ca9e_2dc3,
            0x14be_74db_faee_5748,
        ]),
    },
    Fq2::one(),
];
