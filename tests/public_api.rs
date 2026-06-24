#[test]
fn high_level_crate_reexports_diagnostics_module() {
    use gamlss::diagnostics as _;
}

#[test]
fn high_level_crate_reexports_prepared_cyclic_penalty() {
    use gamlss::spline::PreparedCyclicDifferencePenalty;

    let penalty = PreparedCyclicDifferencePenalty::try_new(0.5, 2).unwrap();
    assert_eq!(penalty.lambda(), 0.5);
    assert_eq!(penalty.order(), 2);
    assert_eq!(penalty.coefficients().len(), 3);
}
