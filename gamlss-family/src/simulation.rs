use gamlss_core::{Family, HasQuantile};

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
    let positive = rand_distr::Distribution::sample(
        &rand_distr::Bernoulli::new(0.5).expect("fair Bernoulli probability must construct"),
        rng,
    );
    if positive { 1.0 } else { -1.0 }
}

#[inline]
pub fn sample_quantile<Rng, F>(rng: &mut Rng, family: &F, theta: &F::Theta) -> f64
where
    Rng: rand::Rng,
    F: HasQuantile + for<'obs> Family<Observation<'obs> = f64>,
{
    family.quantile(open_unit(rng), theta)
}
