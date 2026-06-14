use gamlss_core::{FloorSoftplusScalar, NegativeSoftplusScalar, PredictorBlock, SoftplusScalar};

#[test]
fn convenience_predictor_helpers_remain_root_reexports() {
    let softplus = SoftplusScalar::new(1);
    let negative = NegativeSoftplusScalar::new(1);
    let floored = FloorSoftplusScalar::new(1, 0.5);

    assert_eq!(softplus.nparams(), 1);
    assert_eq!(negative.nparams(), 1);
    assert_eq!(floored.nparams(), 1);
}
