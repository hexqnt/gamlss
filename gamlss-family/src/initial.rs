use gamlss_core::{Family, ObservationView};

pub const POSITIVE_FLOOR: f64 = 1.0e-6;
pub const PROBABILITY_FLOOR: f64 = 1.0e-6;
pub const VARIANCE_FLOOR: f64 = 1.0e-12;
pub const LARGE_SHAPE: f64 = 1.0e6;

#[derive(Debug, Clone, Copy)]
pub struct WeightedSummary {
    pub(crate) mean: f64,
    pub(crate) variance: f64,
}

pub fn weighted_values<'obs, F, Obs, P>(obs: &'obs Obs, mut valid: P) -> Vec<(f64, f64)>
where
    F: Family<Observation<'obs> = f64>,
    Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    P: FnMut(f64) -> Option<f64>,
{
    let mut values = Vec::with_capacity(obs.len());
    for row in 0..obs.len() {
        let weight = obs.weight_at(row);
        if weight <= 0.0 || !weight.is_finite() {
            continue;
        }
        let y = obs.observation_at(row);
        if let Some(value) = valid(y)
            && value.is_finite()
        {
            values.push((value, weight));
        }
    }
    values
}

pub fn weighted_summary(values: &[(f64, f64)]) -> Option<WeightedSummary> {
    let weight = values.iter().map(|(_, weight)| weight).sum::<f64>();
    if weight <= 0.0 || !weight.is_finite() {
        return None;
    }

    let mean = values
        .iter()
        .map(|(value, value_weight)| value * value_weight)
        .sum::<f64>()
        / weight;
    if !mean.is_finite() {
        return None;
    }

    let variance = values
        .iter()
        .map(|(value, value_weight)| value_weight * (value - mean) * (value - mean))
        .sum::<f64>()
        / weight;

    Some(WeightedSummary {
        mean,
        variance: variance.max(0.0),
    })
}

pub fn weighted_mean(values: &[(f64, f64)]) -> Option<f64> {
    weighted_summary(values).map(|summary| summary.mean)
}

pub fn weighted_quantile(values: &[(f64, f64)], probability: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=1.0).contains(&probability) {
        return None;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.0.total_cmp(&right.0));
    let total_weight = sorted.iter().map(|(_, weight)| weight).sum::<f64>();
    if total_weight <= 0.0 || !total_weight.is_finite() {
        return None;
    }

    let target = probability * total_weight;
    let mut cumulative = 0.0;
    for (value, weight) in sorted.iter().copied() {
        cumulative += weight;
        if cumulative >= target {
            return Some(value);
        }
    }

    sorted.last().map(|(value, _)| *value)
}

pub fn weighted_median(values: &[(f64, f64)]) -> Option<f64> {
    weighted_quantile(values, 0.5)
}

pub fn robust_location_scale(values: &[(f64, f64)]) -> Option<(f64, f64)> {
    let location = weighted_median(values)?;
    let q1 = weighted_quantile(values, 0.25).unwrap_or(location);
    let q3 = weighted_quantile(values, 0.75).unwrap_or(location);
    let iqr_scale = (q3 - q1).abs() / 1.349;
    let deviations = values
        .iter()
        .map(|(value, weight)| ((value - location).abs(), *weight))
        .collect::<Vec<_>>();
    let mad_scale = weighted_median(&deviations).unwrap_or(0.0) * 1.4826;
    let moment_scale = weighted_summary(values).map_or(0.0, |summary| summary.variance.sqrt());

    let scale = positive_floor(
        [mad_scale, iqr_scale, moment_scale, POSITIVE_FLOOR]
            .into_iter()
            .find(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(POSITIVE_FLOOR),
    );
    Some((location, scale))
}

pub fn positive_floor(value: f64) -> f64 {
    if value.is_finite() && value > POSITIVE_FLOOR {
        value
    } else {
        POSITIVE_FLOOR
    }
}

pub fn probability_floor(value: f64) -> f64 {
    value.clamp(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
}
