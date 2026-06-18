#[test]
fn high_level_crate_reexports_diagnostics_module() {
    use gamlss::diagnostics as _;
}

#[test]
fn high_level_crate_reexports_prepared_cyclic_penalty() {
    use gamlss::spline::PreparedCyclicDifferencePenalty;

    let penalty = PreparedCyclicDifferencePenalty::new(0.5, 2);
    assert_eq!(penalty.coefficients().len(), 3);
}
