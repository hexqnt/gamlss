#![allow(clippy::float_cmp)]
#[test]
fn high_level_crate_reexports_diagnostics_module() {
    use gamlss::diagnostics as _;
}

#[cfg(feature = "multivariate")]
#[test]
fn high_level_crate_reexports_multivariate_family() {
    use gamlss::prelude::*;

    let _ = DynMvNormalCholeskyDefault::new(2);
    let _ = MvNormalCholeskyDefault::<2>::new();
    let _ = gamlss::family::DynMvNormalCholeskyDefault::new(2);
    let _ = gamlss::family::MvNormalCholeskyDefault::<2>::new();
}

#[test]
fn high_level_crate_reexports_prepared_cyclic_penalty() {
    use gamlss::spline::PreparedCyclicDifferencePenalty;

    let penalty = PreparedCyclicDifferencePenalty::try_new(0.5f64, 2).unwrap();
    assert_eq!(penalty.lambda(), 0.5);
    assert_eq!(penalty.order(), 2);
    assert_eq!(penalty.coefficients().len(), 3);
}
