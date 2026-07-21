//! Dimension-independent sinh-arcsinh transformation helpers.

#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub asinh_x: f64,
    pub h: f64,
    pub latent: f64,
}

#[inline]
pub fn transform_standardized(x: f64, nu: f64, tau: f64) -> Transform {
    let asinh_x = x.asinh();
    let h = tau.mul_add(asinh_x, -nu.ln());
    Transform {
        asinh_x,
        h,
        latent: h.sinh(),
    }
}

#[inline]
pub fn inverse_standardized(latent: f64, nu: f64, tau: f64) -> f64 {
    ((latent.asinh() + nu.ln()) / tau).sinh()
}

#[inline]
pub fn log_cosh(value: f64) -> f64 {
    let absolute = value.abs();
    absolute + (-2.0 * absolute).exp().ln_1p() - std::f64::consts::LN_2
}
