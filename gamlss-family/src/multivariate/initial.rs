#![allow(clippy::suboptimal_flops)]

//! Distribution-independent initial values for shared covariance parameterizations.

use gamlss_core::{InitialEtaFromTheta, ObservationView};

pub(super) fn location_cholesky<
    'obs,
    const D: usize,
    LocationLink,
    DiagonalLink,
    OffDiagonalLink,
    Obs,
>(
    obs: &'obs Obs,
) -> ([f64; D], [[f64; D]; D])
where
    LocationLink: InitialEtaFromTheta<f64>,
    DiagonalLink: InitialEtaFromTheta<f64>,
    OffDiagonalLink: InitialEtaFromTheta<f64>,
    Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
{
    let mut weight_sum = [0.0; D];
    let mut location = [0.0; D];
    for row in 0..obs.len() {
        let weight = obs.weight_at(row);
        if weight == 0.0 {
            continue;
        }
        let observation = obs.observation_at(row);
        for component in 0..D {
            let value = observation[component];
            if value.is_finite() {
                weight_sum[component] += weight;
                location[component] += weight * value;
            }
        }
    }
    for component in 0..D {
        if weight_sum[component] > 0.0 {
            location[component] /= weight_sum[component];
        }
    }

    let mut variances = [0.0; D];
    for row in 0..obs.len() {
        let weight = obs.weight_at(row);
        if weight == 0.0 {
            continue;
        }
        let observation = obs.observation_at(row);
        for component in 0..D {
            let value = observation[component];
            if value.is_finite() && weight_sum[component] > 0.0 {
                let residual = value - location[component];
                variances[component] += weight * residual * residual;
            }
        }
    }

    let mut location_eta = [0.0; D];
    let mut lower_eta = [[OffDiagonalLink::initial_eta_from_theta(0.0); D]; D];
    for component in 0..D {
        location_eta[component] = LocationLink::initial_eta_from_theta(location[component]);
        let scale = if weight_sum[component] > 0.0 {
            (variances[component] / weight_sum[component])
                .sqrt()
                .max(1.0e-6)
        } else {
            1.0
        };
        lower_eta[component][component] = DiagonalLink::initial_eta_from_theta(scale);
    }

    (location_eta, lower_eta)
}

#[allow(clippy::needless_range_loop)]
pub(super) fn location_scale_partial_correlation<
    'obs,
    const D: usize,
    LocationLink,
    ScaleLink,
    Obs,
>(
    obs: &'obs Obs,
) -> ([f64; D], [f64; D], [[f64; D]; D])
where
    LocationLink: InitialEtaFromTheta<f64>,
    ScaleLink: InitialEtaFromTheta<f64>,
    Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
{
    let mut weight_sum = 0.0;
    let mut location = [0.0; D];
    for row in 0..obs.len() {
        let weight = obs.weight_at(row);
        let value = obs.observation_at(row);
        if weight > 0.0 && value.iter().all(|entry| entry.is_finite()) {
            weight_sum += weight;
            for component in 0..D {
                location[component] += weight * value[component];
            }
        }
    }
    if weight_sum > 0.0 {
        for mean in &mut location {
            *mean /= weight_sum;
        }
    }

    let mut covariance = [[0.0; D]; D];
    for row in 0..obs.len() {
        let weight = obs.weight_at(row);
        let value = obs.observation_at(row);
        if weight > 0.0 && value.iter().all(|entry| entry.is_finite()) {
            for i in 0..D {
                for j in 0..=i {
                    covariance[i][j] +=
                        weight * (value[i] - location[i]) * (value[j] - location[j]);
                }
            }
        }
    }
    if weight_sum > 0.0 {
        for i in 0..D {
            for j in 0..=i {
                covariance[i][j] /= weight_sum;
                covariance[j][i] = covariance[i][j];
            }
        }
    }

    let scale = std::array::from_fn(|index| covariance[index][index].sqrt().max(1.0e-6));
    let mut correlation = [[0.0; D]; D];
    for i in 0..D {
        correlation[i][i] = 1.0;
        for j in 0..i {
            let empirical = covariance[i][j] / (scale[i] * scale[j]);
            correlation[i][j] = 0.8 * empirical.clamp(-0.99, 0.99);
            correlation[j][i] = correlation[i][j];
        }
    }

    let mut cholesky = [[0.0; D]; D];
    for row in 0..D {
        for col in 0..=row {
            let correction = (0..col)
                .map(|k| cholesky[row][k] * cholesky[col][k])
                .sum::<f64>();
            cholesky[row][col] = if row == col {
                (correlation[row][row] - correction).max(1.0e-8).sqrt()
            } else {
                (correlation[row][col] - correction) / cholesky[col][col]
            };
        }
    }

    let mut partial_eta = [[0.0; D]; D];
    for row in 1..D {
        let mut prefix = 1.0;
        for col in 0..row {
            let partial = (cholesky[row][col] / prefix).clamp(-0.95, 0.95);
            partial_eta[row][col] = partial.atanh();
            prefix *= (1.0 - partial * partial).sqrt();
        }
    }

    (
        location.map(LocationLink::initial_eta_from_theta),
        scale.map(ScaleLink::initial_eta_from_theta),
        partial_eta,
    )
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{Identity, Log};

    use super::{location_cholesky, location_scale_partial_correlation};

    #[test]
    fn cholesky_initialization_is_parameterization_not_distribution_specific() {
        let observations = [[0.0, 2.0], [2.0, 4.0]];
        let obs = observations.as_slice();
        let (location, lower) = location_cholesky::<2, Identity, Log, Identity, _>(&obs);

        for (actual, expected) in location.into_iter().zip([1.0, 3.0]) {
            assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
        }
        assert_relative_eq!(lower[0][0], 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(lower[1][0], 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(lower[1][1], 0.0, epsilon = 1.0e-12);
    }

    #[test]
    fn partial_correlation_initialization_returns_shared_geometry() {
        let observations = [[0.0, 2.0], [2.0, 4.0]];
        let obs = observations.as_slice();
        let (location, scale, partial_corr) =
            location_scale_partial_correlation::<2, Identity, Log, _>(&obs);

        for (actual, expected) in location.into_iter().zip([1.0, 3.0]) {
            assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
        }
        assert_relative_eq!(scale[0], 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(scale[1], 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(partial_corr[1][0], 0.792_f64.atanh(), epsilon = 1.0e-12);
    }
}
