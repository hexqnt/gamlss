use gamlss_core::{Family, HasQuantile, SimulationError};

#[inline]
pub fn open_unit<Rng>(rng: &mut Rng) -> f64
where
    Rng: rand::Rng,
{
    rand_distr::Distribution::sample(&rand_distr::Open01, rng)
}

#[inline]
pub fn standard_normal<Rng>(rng: &mut Rng) -> f64
where
    Rng: rand::Rng,
{
    rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng)
}

#[inline]
pub fn fair_sign<Rng>(rng: &mut Rng) -> f64
where
    Rng: rand::Rng,
{
    if open_unit(rng) < 0.5 { 1.0 } else { -1.0 }
}

#[inline]
pub const fn ensure_finite(sample: f64, detail: &'static str) -> Result<f64, SimulationError> {
    if sample.is_finite() {
        Ok(sample)
    } else {
        Err(SimulationError::NumericalFailure(detail))
    }
}

#[inline]
pub fn try_sample_quantile<Rng, F>(
    rng: &mut Rng,
    family: &F,
    theta: &F::Theta,
    detail: &'static str,
) -> Result<f64, SimulationError>
where
    Rng: rand::Rng,
    F: HasQuantile + for<'obs> Family<Observation<'obs> = f64>,
{
    ensure_finite(family.quantile(open_unit(rng), theta), detail)
}
