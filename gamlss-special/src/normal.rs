//! Normal density, stable tails, quantiles and Owen's T.

use crate::HALF_LOG_2_PI;
use crate::probability::clamp_probability;
use crate::quadrature::integrate_finite;

#[inline]
fn polynomial_ascending(value: f64, coefficients: &[f64]) -> f64 {
    // AS241 coefficients below are stored from constant term to highest degree.
    coefficients
        .iter()
        .rev()
        .fold(0.0, |accumulator, coefficient| {
            accumulator * value + coefficient
        })
}

#[inline]
fn polynomial_ascending_with_constant_one(value: f64, coefficients: &[f64]) -> f64 {
    value.mul_add(polynomial_ascending(value, coefficients), 1.0)
}

#[inline]
fn polynomial_descending(value: f64, coefficients: &[f64]) -> f64 {
    // Cody/Cephes coefficients below are stored from highest degree to constant term.
    coefficients.iter().fold(0.0, |accumulator, coefficient| {
        accumulator * value + coefficient
    })
}

#[inline]
fn polynomial_descending_with_implicit_leading_one(value: f64, coefficients: &[f64]) -> f64 {
    let Some((&first, rest)) = coefficients.split_first() else {
        return value;
    };

    rest.iter().fold(value + first, |accumulator, coefficient| {
        accumulator * value + coefficient
    })
}

/// Standard normal log-density.
#[must_use]
#[inline]
pub fn unit_normal_log_pdf(z: f64) -> f64 {
    (0.5 * z).mul_add(-z, -HALF_LOG_2_PI)
}

/// Standard normal CDF.
#[must_use]
#[inline]
pub fn unit_normal_cdf(z: f64) -> f64 {
    unit_normal_sf(-z)
}

/// Standard normal survival function `1 - Phi(z)`.
#[must_use]
#[inline]
pub fn unit_normal_sf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z == f64::NEG_INFINITY {
        return 1.0;
    }
    if z == f64::INFINITY {
        return 0.0;
    }

    clamp_probability(0.5 * unit_normal_erfc(z * std::f64::consts::FRAC_1_SQRT_2))
}

/// Natural logarithm of the standard normal survival function.
#[must_use]
#[inline]
pub fn unit_normal_log_sf(z: f64) -> f64 {
    log_ndtr(-z)
}

fn unit_normal_erfc(x: f64) -> f64 {
    const ERF_NUMERATOR: [f64; 5] = [
        9.604_973_739_870_516,
        90.026_019_720_384_27,
        2_232.005_345_946_843,
        7_003.325_141_128_051,
        55_592.301_301_039_49,
    ];
    const ERF_DENOMINATOR: [f64; 5] = [
        33.561_714_164_750_31,
        521.357_949_780_152_7,
        4_594.323_829_709_801,
        22_629.000_061_389_09,
        49_267.394_260_863_59,
    ];
    const ERFC_NUMERATOR: [f64; 9] = [
        2.461_969_814_735_305e-10,
        0.564_189_564_831_068_8,
        7.463_210_564_422_699,
        48.637_197_098_568_14,
        196.520_832_956_077_1,
        526.445_194_995_477_3,
        934.528_527_171_957_6,
        1_027.551_886_895_157,
        557.535_335_369_399_4,
    ];
    const ERFC_DENOMINATOR: [f64; 8] = [
        13.228_195_115_474_499,
        86.707_214_088_598_97,
        354.937_778_887_819_9,
        975.708_501_743_205_5,
        1_823.909_166_879_097_3,
        2_246.337_608_187_109_7,
        1_656.663_091_941_613_5,
        557.535_340_817_727_7,
    ];
    const ERFC_TAIL_NUMERATOR: [f64; 6] = [
        0.564_189_583_547_755_1,
        1.275_366_707_599_781,
        5.019_050_422_511_805,
        6.160_210_979_930_536,
        7.409_742_699_504_489,
        2.978_866_653_721_002,
    ];
    const ERFC_TAIL_DENOMINATOR: [f64; 6] = [
        2.260_528_632_201_172_6,
        9.396_035_249_380_014,
        12.048_953_980_809_665,
        17.081_445_074_756_59,
        9.608_968_090_632_859,
        3.369_076_451_000_815,
    ];
    const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const ERFC_UNDERFLOW_X: f64 = 27.3;

    if x.abs() <= SQRT_HALF {
        let square = x * x;
        let erf = x * polynomial_descending(square, &ERF_NUMERATOR)
            / polynomial_descending_with_implicit_leading_one(square, &ERF_DENOMINATOR);
        return 1.0 - erf;
    }

    let abs_x = x.abs();
    if abs_x >= ERFC_UNDERFLOW_X {
        return if x < 0.0 { 2.0 } else { 0.0 };
    }

    let (numerator, denominator) = if abs_x < 8.0 {
        (
            polynomial_descending(abs_x, &ERFC_NUMERATOR),
            polynomial_descending_with_implicit_leading_one(abs_x, &ERFC_DENOMINATOR),
        )
    } else {
        (
            polynomial_descending(abs_x, &ERFC_TAIL_NUMERATOR),
            polynomial_descending_with_implicit_leading_one(abs_x, &ERFC_TAIL_DENOMINATOR),
        )
    };

    let erfc = (-abs_x * abs_x).exp() * numerator / denominator;
    if x < 0.0 { 2.0 - erfc } else { erfc }
}

/// Natural logarithm of the standard normal CDF.
#[must_use]
#[inline]
pub fn log_ndtr(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if z == f64::INFINITY {
        return 0.0;
    }
    if z <= -10.0 {
        return log_ndtr_left_tail(z);
    }
    if z > 5.0 {
        return (-unit_normal_sf(z)).ln_1p();
    }

    unit_normal_cdf(z).ln()
}

fn log_ndtr_left_tail(z: f64) -> f64 {
    let x = -z;
    let inv2 = 1.0 / (x * x);
    let correction = normal_left_tail_correction(inv2);

    unit_normal_log_pdf(z) - x.ln() + correction.max(f64::MIN_POSITIVE).ln()
}

#[inline]
fn normal_left_tail_correction(inv2: f64) -> f64 {
    polynomial_descending(
        inv2,
        &[
            -34_459_425.0,
            2_027_025.0,
            -135_135.0,
            10_395.0,
            -945.0,
            105.0,
            -15.0,
            3.0,
            -1.0,
            1.0,
        ],
    )
}

/// Standard normal Mills ratio `phi(z) / Phi(z)`.
///
/// Infinite tails map to positive infinity and zero, respectively; `NaN`
/// propagates.
#[must_use]
#[inline]
pub fn normal_mills_ratio(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z == f64::NEG_INFINITY {
        return f64::INFINITY;
    }
    if z == f64::INFINITY {
        return 0.0;
    }

    if z <= -10.0 {
        let x = -z;
        let correction = normal_left_tail_correction(1.0 / (x * x));
        return x / correction.max(f64::MIN_POSITIVE);
    }

    let log_ratio = unit_normal_log_pdf(z) - log_ndtr(z);
    if log_ratio.is_nan() && z < 0.0 {
        // Both log terms may underflow to `-inf` for extreme finite left-tail
        // inputs. The leading asymptotic term remains representable.
        -z
    } else {
        log_ratio.exp()
    }
}

/// Owen's T function `T(h, a)`.
///
/// This is primarily used for skew-normal CDF evaluation. Arguments with
/// `|a| > 1` are reduced by Owen's reciprocal identity, leaving adaptive
/// Simpson integration on a bounded interval no wider than one.
#[must_use]
pub fn owens_t(h: f64, a: f64) -> f64 {
    if !h.is_finite() || !a.is_finite() {
        return f64::NAN;
    }
    if a == 0.0 {
        return 0.0;
    }

    let sign = a.signum();
    let upper = a.abs();
    if upper > 1.0 {
        return sign * owens_t_reciprocal_reduction(h.abs(), upper);
    }

    let h2 = h * h;

    #[allow(clippy::suboptimal_flops)]
    let integral = integrate_finite(0.0, upper, |x| {
        (-0.5 * h2 * (1.0 + x * x)).exp() / (1.0 + x * x)
    });
    sign * integral / (2.0 * std::f64::consts::PI)
}

fn owens_t_reciprocal_reduction(h: f64, a: f64) -> f64 {
    let scaled_h = h * a;
    let cdf_h = unit_normal_cdf(h);
    let cdf_scaled = unit_normal_cdf(scaled_h);
    let reduced = if scaled_h.is_finite() {
        owens_t(scaled_h, 1.0 / a)
    } else {
        0.0
    };
    let value = cdf_h.mul_add(-cdf_scaled, f64::midpoint(cdf_h, cdf_scaled)) - reduced;
    value.clamp(0.0, 0.25)
}

/// Standard normal quantile using Wichura's AS241 rational approximation.
#[must_use]
#[inline]
#[allow(clippy::float_cmp)]
pub fn unit_normal_quantile(p: f64) -> f64 {
    const CENTER_NUMERATOR: [f64; 8] = [
        3.387_132_872_796_366_5,
        133.141_667_891_784_38,
        1_971.590_950_306_551_3,
        13_731.693_765_509_46,
        45_921.953_931_549_87,
        67_265.770_927_008_7,
        33_430.575_583_588_13,
        2_509.080_928_730_122_7,
    ];
    const CENTER_DENOMINATOR: [f64; 7] = [
        42.313_330_701_600_91,
        687.187_007_492_057_9,
        5_394.196_021_424_751,
        21_213.794_301_586_597,
        39_307.895_800_092_71,
        28_729.085_735_721_943,
        5_226.495_278_852_855,
    ];
    const TAIL_NUMERATOR: [f64; 8] = [
        1.423_437_110_749_683_5,
        4.630_337_846_156_545,
        5.769_497_221_460_691,
        3.647_848_324_763_204_5,
        1.270_458_252_452_368_4,
        0.241_780_725_177_450_6,
        0.022_723_844_989_269_184,
        0.000_774_545_014_278_341_4,
    ];
    const TAIL_DENOMINATOR: [f64; 7] = [
        2.053_191_626_637_759,
        1.676_384_830_183_803_8,
        0.689_767_334_985_1,
        0.148_103_976_427_480_08,
        0.015_198_666_563_616_457,
        0.000_547_593_808_499_534_5,
        1.050_750_071_644_416_8e-9,
    ];
    const FAR_TAIL_NUMERATOR: [f64; 8] = [
        6.657_904_643_501_104,
        5.463_784_911_164_114,
        1.784_826_539_917_291_3,
        0.296_560_571_828_504_87,
        0.026_532_189_526_576_124,
        0.001_242_660_947_388_078_4,
        0.000_027_115_555_687_434_876,
        0.000_000_201_033_439_929_228_82,
    ];
    const FAR_TAIL_DENOMINATOR: [f64; 7] = [
        0.599_832_206_555_887_9,
        0.136_929_880_922_735_8,
        0.014_875_361_290_850_615,
        0.000_786_869_131_145_613_3,
        0.000_018_463_183_175_100_547,
        0.000_000_142_151_175_831_644_6,
        2.044_263_103_389_939_7e-15,
    ];

    if p < 0.0 || !p.is_finite() || p > 1.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }

    let centered = p - 0.5;
    if centered.abs() <= 0.425 {
        let r = centered.mul_add(-centered, 0.180_625);
        centered * polynomial_ascending(r, &CENTER_NUMERATOR)
            / polynomial_ascending_with_constant_one(r, &CENTER_DENOMINATOR)
    } else {
        let tail_probability = if centered < 0.0 { p } else { 1.0 - p };
        let mut r = (-tail_probability.ln()).sqrt();
        let quantile = if r <= 5.0 {
            r -= 1.6;
            polynomial_ascending(r, &TAIL_NUMERATOR)
                / polynomial_ascending_with_constant_one(r, &TAIL_DENOMINATOR)
        } else {
            r -= 5.0;
            polynomial_ascending(r, &FAR_TAIL_NUMERATOR)
                / polynomial_ascending_with_constant_one(r, &FAR_TAIL_DENOMINATOR)
        };
        if centered < 0.0 { -quantile } else { quantile }
    }
}
