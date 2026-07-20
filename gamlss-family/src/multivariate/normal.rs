#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::suboptimal_flops
)]

use gamlss_core::{Identity, Log};

pub use dynamic::{DynMvNormalCholesky, DynMvNormalCholeskyEta, DynMvNormalCholeskyTheta};
pub use fixed::{MvNormalCholesky, MvNormalCholeskyEta, MvNormalCholeskyTheta};
pub use mean_std_partial_corr::{
    FixedPartialCorrelations, MvNormalMeanStdPartialCorr, MvNormalMeanStdPartialCorrDefault,
    MvNormalMeanStdPartialCorrEta, MvNormalMeanStdPartialCorrTheta,
};

mod dynamic;
mod fixed;
pub(in crate::multivariate) mod kernel;
pub(in crate::multivariate) mod mean_std_partial_corr;

/// Generic multivariate normal with a lower-triangular Cholesky scale factor.
///
/// This is the primary dimension-generic covariance parameterization. The
/// natural-scale covariance is `L L'`, where `L` is the lower-triangular
/// Cholesky scale factor. Diagonal entries are constrained positive by the
/// diagonal link, while off-diagonal entries remain unconstrained.
pub type MvNormalCholeskyDefault<const D: usize> = MvNormalCholesky<D, Identity, Log, Identity>;

/// Runtime-dimensional multivariate normal with default links.
pub type DynMvNormalCholeskyDefault = DynMvNormalCholesky<Identity, Log, Identity>;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::Family;

    use crate::multivariate::matrix::{FixedLowerTriangular, PackedLowerTriangular};

    use super::{
        DynMvNormalCholeskyDefault, DynMvNormalCholeskyEta, MvNormalCholeskyDefault,
        MvNormalCholeskyEta,
    };

    #[test]
    fn dynamic_case_matches_fixed_dimensional_case() {
        let fixed = MvNormalCholeskyDefault::<3>::new();
        let dynamic = DynMvNormalCholeskyDefault::new(3).unwrap();
        let y = [1.7, -0.8, 0.2];
        let fixed_eta = MvNormalCholeskyEta::new(
            [0.4, -0.3, 0.1],
            FixedLowerTriangular::from_lower_rows([
                [-0.2, 0.0, 0.0],
                [0.25, 0.1, 0.0],
                [-0.1, 0.2, 0.3],
            ]),
        );
        let dynamic_eta = DynMvNormalCholeskyEta::new(
            fixed_eta.mu().to_vec(),
            PackedLowerTriangular::try_new(3, vec![-0.2, 0.25, 0.1, -0.1, 0.2, 0.3]).unwrap(),
        )
        .unwrap();

        let (fixed_nll, fixed_gradient) =
            fixed.nll_and_gradient_eta(y, &fixed_eta, &mut fixed.workspace());
        let (dynamic_nll, dynamic_gradient) =
            dynamic.nll_and_gradient_eta(&y, &dynamic_eta, &mut dynamic.workspace());

        assert_relative_eq!(dynamic_nll, fixed_nll, epsilon = 1.0e-12);
        assert_eq!(
            dynamic
                .theta(&dynamic_eta, &mut dynamic.workspace())
                .mu()
                .len(),
            dynamic.dimension()
        );
        assert_eq!(dynamic_gradient.mu(), fixed_gradient.mu());
        for row in 0..3 {
            for col in 0..=row {
                assert_eq!(
                    dynamic_gradient.cholesky_entry(row, col),
                    fixed_gradient.cholesky().get(row, col)
                );
            }
        }
    }
}
