#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::suboptimal_flops
)]

use gamlss_core::{Identity, Log};

pub use dynamic::{DynMvNormalCholesky, DynMvNormalCholeskyEta, DynMvNormalCholeskyTheta};
pub use fixed::{MvNormalCholesky, MvNormalCholeskyEta, MvNormalCholeskyTheta};

mod dynamic;
mod fixed;
mod kernel;

/// Multivariate normal with a lower-triangular Cholesky scale factor.
pub type MvNormalCholeskyDefault<const D: usize> = MvNormalCholesky<D, Identity, Log, Identity>;

/// Runtime-dimensional multivariate normal with default links.
pub type DynMvNormalCholeskyDefault = DynMvNormalCholesky<Identity, Log, Identity>;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::Family;

    use super::{
        DynMvNormalCholeskyDefault, DynMvNormalCholeskyEta, MvNormalCholeskyDefault,
        MvNormalCholeskyEta,
    };

    #[test]
    fn dynamic_case_matches_fixed_dimensional_case() {
        let fixed = MvNormalCholeskyDefault::<3>::new();
        let dynamic = DynMvNormalCholeskyDefault::new(3);
        let y = [1.7, -0.8, 0.2];
        let fixed_eta = MvNormalCholeskyEta {
            mu: [0.4, -0.3, 0.1],
            cholesky: [[-0.2, 0.0, 0.0], [0.25, 0.1, 0.0], [-0.1, 0.2, 0.3]],
        };
        let dynamic_eta = DynMvNormalCholeskyEta::new(
            fixed_eta.mu.to_vec(),
            fixed_eta.cholesky.as_flattened().to_vec(),
        );

        let (fixed_nll, fixed_gradient) = fixed.nll_and_gradient_eta(y, fixed_eta);
        let (dynamic_nll, dynamic_gradient) = dynamic.nll_and_gradient_eta(&y, dynamic_eta.clone());

        assert_relative_eq!(dynamic_nll, fixed_nll, epsilon = 1.0e-12);
        assert_eq!(dynamic.theta(dynamic_eta).mu.len(), dynamic.dimension());
        assert_eq!(dynamic_gradient.mu, fixed_gradient.mu);
        assert_eq!(
            dynamic_gradient.cholesky,
            fixed_gradient.cholesky.as_flattened()
        );
    }
}
