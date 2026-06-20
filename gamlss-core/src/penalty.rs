use std::ops::Range;

/// Zero penalty.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoPenalty;

impl Penalty for NoPenalty {
    #[inline(always)]
    fn value(&self, _: &[f64]) -> f64 {
        0.0
    }

    #[inline(always)]
    fn add_gradient(&self, _: &[f64], _: &mut [f64]) {}
}

impl GlobalPenalty for NoPenalty {
    #[inline(always)]
    fn value(&self, _: &[f64]) -> f64 {
        0.0
    }

    #[inline(always)]
    fn add_gradient(&self, _: &[f64], _: &mut [f64]) {}
}

impl MatrixPenalty for NoPenalty {
    #[inline(always)]
    fn add_penalty_matrix(&self, dim: usize, gram: &mut [f64]) {
        debug_assert_matrix_shape(dim, gram);
    }
}

/// Ridge penalty `lambda * sum(beta_i^2)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RidgePenalty {
    /// Regularization weight.
    pub lambda: f64,
}

impl RidgePenalty {
    /// Creates a ridge penalty with the given `lambda`.
    #[must_use]
    #[inline]
    pub const fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Penalty for RidgePenalty {
    #[inline]
    fn value(&self, beta: &[f64]) -> f64 {
        self.lambda * beta.iter().map(|value| value * value).sum::<f64>()
    }

    #[inline]
    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(beta.len(), grad.len());

        let scale = 2.0 * self.lambda;
        for (grad_value, beta_value) in grad.iter_mut().zip(beta) {
            *grad_value = scale.mul_add(*beta_value, *grad_value);
        }
    }
}

impl MatrixPenalty for RidgePenalty {
    #[inline]
    fn add_penalty_matrix(&self, dim: usize, gram: &mut [f64]) {
        debug_assert_matrix_shape(dim, gram);

        if dim == 0 {
            return;
        }

        for (row, row_values) in gram.chunks_exact_mut(dim).enumerate() {
            row_values[row] += self.lambda;
        }
    }
}

/// Applies a local penalty to a subrange of a larger coefficient block.
///
/// This is useful when one parameter predictor is composed from several terms
/// and only one term should receive a local regularizer. The range is expressed
/// in the local coefficient coordinates passed to [`Penalty`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentPenalty<P> {
    range: Range<usize>,
    penalty: P,
}

impl<P> SegmentPenalty<P> {
    /// Creates a segment penalty over `range`.
    #[must_use]
    #[inline]
    pub const fn new(range: Range<usize>, penalty: P) -> Self {
        Self { range, penalty }
    }

    /// Returns the local coefficient range affected by this penalty.
    #[must_use]
    #[inline]
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Returns the wrapped penalty.
    #[must_use]
    #[inline]
    pub const fn penalty(&self) -> &P {
        &self.penalty
    }
}

impl<P> Penalty for SegmentPenalty<P>
where
    P: Penalty,
{
    #[inline]
    fn value(&self, beta: &[f64]) -> f64 {
        self.penalty.value(&beta[self.range.clone()])
    }

    #[inline]
    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(beta.len(), grad.len());

        let start = self.range.start;
        let end = self.range.end;
        self.penalty
            .add_gradient(&beta[start..end], &mut grad[start..end]);
    }
}

/// Weighted coefficient in a full-vector linear form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearTerm {
    /// Index in the full flat coefficient vector.
    pub index: usize,
    /// Multiplicative coefficient for `beta[index]`.
    pub weight: f64,
}

impl LinearTerm {
    /// Creates a linear-form term.
    #[must_use]
    #[inline]
    pub const fn new(index: usize, weight: f64) -> Self {
        Self { index, weight }
    }
}

/// Linear form over a full flat coefficient vector.
///
/// The value is `constant + sum(term.weight * beta[term.index])`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearForm {
    /// Coefficients participating in the form.
    pub terms: Vec<LinearTerm>,
    /// Additive constant.
    pub constant: f64,
}

impl LinearForm {
    /// Creates a linear form from terms and a constant.
    #[must_use]
    #[inline]
    pub const fn new(terms: Vec<LinearTerm>, constant: f64) -> Self {
        Self { terms, constant }
    }

    /// Creates a builder for a linear form over the full beta vector.
    #[must_use]
    #[inline]
    pub fn builder() -> LinearFormBuilder {
        LinearFormBuilder::new()
    }

    /// Converts this form into `weight * max(form(beta), 0)^2`.
    ///
    /// This represents a soft quadratic penalty for constraints written as
    /// `form(beta) <= 0`.
    #[must_use]
    #[inline]
    pub const fn hinge_le(self, weight: f64) -> HingeQuadraticPenalty {
        HingeQuadraticPenalty::new(self, weight)
    }

    /// Converts this form into a relative quadratic absolute-limit penalty.
    #[must_use]
    #[inline]
    pub const fn absolute_limit(self, weight: f64, scale: f64, limit: f64) -> AbsoluteLimitPenalty {
        AbsoluteLimitPenalty::new(self, weight, scale, limit)
    }

    /// Evaluates the form at `beta`.
    ///
    /// # Panics
    ///
    /// Panics if a term index is out of bounds for `beta`.
    #[must_use]
    #[inline]
    pub fn value(&self, beta: &[f64]) -> f64 {
        self.terms.iter().fold(self.constant, |sum, term| {
            sum + term.weight * beta[term.index]
        })
    }

    #[inline]
    fn add_scaled_gradient(&self, scale: f64, grad: &mut [f64]) {
        for term in &self.terms {
            grad[term.index] += scale * term.weight;
        }
    }
}

/// Builder for [`LinearForm`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LinearFormBuilder {
    terms: Vec<LinearTerm>,
    constant: f64,
}

impl LinearFormBuilder {
    /// Creates an empty linear-form builder.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            terms: Vec::new(),
            constant: 0.0,
        }
    }

    /// Adds one weighted coefficient.
    #[must_use]
    #[inline]
    pub fn term(mut self, index: usize, weight: f64) -> Self {
        self.terms.push(LinearTerm::new(index, weight));
        self
    }

    /// Adds pre-built weighted terms.
    #[must_use]
    #[inline]
    pub fn terms(mut self, terms: impl IntoIterator<Item = LinearTerm>) -> Self {
        self.terms.extend(terms);
        self
    }

    /// Adds consecutive weighted coefficients starting at `start`.
    ///
    /// The first weight is applied to `beta[start]`, the second to
    /// `beta[start + 1]`, and so on.
    #[must_use]
    #[inline]
    pub fn weighted_terms(mut self, start: usize, weights: impl IntoIterator<Item = f64>) -> Self {
        self.terms.extend(
            weights
                .into_iter()
                .enumerate()
                .map(|(offset, weight)| LinearTerm::new(start + offset, weight)),
        );
        self
    }

    /// Adds weighted coefficients inside `range`.
    ///
    /// Extra weights are ignored after `range` is exhausted. If fewer weights
    /// are provided than `range.len()`, only the corresponding prefix of the
    /// range is added.
    #[must_use]
    #[inline]
    pub fn weighted_range(
        mut self,
        range: Range<usize>,
        weights: impl IntoIterator<Item = f64>,
    ) -> Self {
        self.terms.extend(
            range
                .zip(weights)
                .map(|(index, weight)| LinearTerm::new(index, weight)),
        );
        self
    }

    /// Sets the additive constant.
    #[must_use]
    #[inline]
    pub const fn constant(mut self, constant: f64) -> Self {
        self.constant = constant;
        self
    }

    /// Builds the linear form.
    #[must_use]
    #[inline]
    pub fn build(self) -> LinearForm {
        LinearForm::new(self.terms, self.constant)
    }
}

/// Quadratic hinge penalty `weight * max(form(beta), 0)^2`.
#[derive(Debug, Clone, PartialEq)]
pub struct HingeQuadraticPenalty {
    /// Linear form whose positive part is penalized.
    pub form: LinearForm,
    /// Penalty weight.
    pub weight: f64,
}

impl HingeQuadraticPenalty {
    /// Creates a quadratic hinge penalty.
    #[must_use]
    #[inline]
    pub const fn new(form: LinearForm, weight: f64) -> Self {
        Self { form, weight }
    }

    #[inline]
    fn contribution(&self, beta: &[f64]) -> PenaltyContribution {
        if !self.weight.is_finite() || self.weight <= 0.0 {
            return PenaltyContribution::ZERO;
        }

        let form_value = self.form.value(beta);
        if form_value.is_nan() {
            return PenaltyContribution::new(f64::NAN, f64::NAN);
        }

        let violation = form_value.max(0.0);
        if violation <= 0.0 {
            return PenaltyContribution::ZERO;
        }

        PenaltyContribution::new(
            self.weight * violation * violation,
            2.0 * self.weight * violation,
        )
    }
}

impl GlobalPenalty for HingeQuadraticPenalty {
    #[inline]
    fn value(&self, beta: &[f64]) -> f64 {
        self.contribution(beta).value
    }

    #[inline]
    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        let contribution = self.contribution(beta);
        if contribution.gradient_scale != 0.0 {
            self.form
                .add_scaled_gradient(contribution.gradient_scale, grad);
        }
    }
}

/// Relative quadratic penalty for exceeding an absolute linear-form limit.
#[derive(Debug, Clone, PartialEq)]
pub struct AbsoluteLimitPenalty {
    /// Linear form whose absolute value is constrained.
    pub form: LinearForm,
    /// Penalty weight.
    pub weight: f64,
    /// Converts `abs(form(beta))` into the constrained physical scale.
    pub scale: f64,
    /// Maximum allowed scaled absolute value.
    pub limit: f64,
}

impl AbsoluteLimitPenalty {
    /// Creates an absolute-limit penalty.
    #[must_use]
    #[inline]
    pub const fn new(form: LinearForm, weight: f64, scale: f64, limit: f64) -> Self {
        Self {
            form,
            weight,
            scale,
            limit,
        }
    }

    #[inline]
    fn contribution(&self, beta: &[f64]) -> PenaltyContribution {
        if !self.weight.is_finite()
            || self.weight <= 0.0
            || !self.scale.is_finite()
            || self.scale <= 0.0
            || !self.limit.is_finite()
            || self.limit < 0.0
        {
            return PenaltyContribution::ZERO;
        }

        let form_value = self.form.value(beta);
        if form_value.is_nan() {
            return PenaltyContribution::new(f64::NAN, f64::NAN);
        }

        let abs_scaled = self.scale * form_value.abs();
        let excess = abs_scaled - self.limit;
        if excess <= 0.0 {
            return PenaltyContribution::ZERO;
        }

        let denominator = self.limit.max(1.0e-12);
        let relative = (excess / denominator).min(1.0e6);

        let sign = if form_value >= 0.0 { 1.0 } else { -1.0 };
        PenaltyContribution::new(
            self.weight * relative * relative,
            2.0 * self.weight * relative * self.scale * sign / denominator,
        )
    }
}

impl GlobalPenalty for AbsoluteLimitPenalty {
    #[inline]
    fn value(&self, beta: &[f64]) -> f64 {
        self.contribution(beta).value
    }

    #[inline]
    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        let contribution = self.contribution(beta);
        if contribution.gradient_scale != 0.0 {
            self.form
                .add_scaled_gradient(contribution.gradient_scale, grad);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PenaltyContribution {
    value: f64,
    gradient_scale: f64,
}

impl PenaltyContribution {
    const ZERO: Self = Self {
        value: 0.0,
        gradient_scale: 0.0,
    };

    #[inline]
    const fn new(value: f64, gradient_scale: f64) -> Self {
        Self {
            value,
            gradient_scale,
        }
    }
}

/// Penalty for the coefficients of a single parameter block.
///
/// Implementations receive the local coefficient slice for one parameter
/// block. The model validates slice lengths before evaluation where possible;
/// hot-path implementations may use debug assertions for length checks.
pub trait Penalty {
    /// Penalty value for the current coefficients.
    fn value(&self, beta: &[f64]) -> f64;
    /// Adds the penalty gradient into the existing `grad`.
    ///
    /// Implementations must add into `grad` and must not clear it, because the
    /// likelihood gradient may already be present in the same buffer.
    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]);
}

/// Penalty evaluated on the full model parameter vector.
///
/// This is useful for constraints or regularization coupling several parameter
/// blocks, while [`Penalty`] remains the local per-block mechanism.
///
/// Implementations receive the full flat beta vector and add their gradient to
/// the full model gradient. They should not allocate or mutate global state.
pub trait GlobalPenalty {
    /// Penalty value for the full beta vector.
    fn value(&self, beta: &[f64]) -> f64;
    /// Adds the penalty gradient into an existing full gradient vector.
    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]);
}

/// Penalty that can be expressed as a quadratic form `β^T P β`.
///
/// This extension trait enables Fisher Scoring solvers to add the penalty
/// matrix to the weighted Gram matrix: `X^T W X + P`. Penalties that cannot
/// be expressed as a constant quadratic form (e.g., slope-limit or
/// monotonic constraints) should not implement this trait.
///
/// Currently [`NoPenalty`] and [`RidgePenalty`] implement this trait.
/// Spline penalties like [`crate::DifferencePenalty`] will implement it
/// in `gamlss-spline`.
pub trait MatrixPenalty: Penalty {
    /// Adds the penalty matrix `P` to `gram` in row-major order.
    ///
    /// `gram` is a `dim × dim` matrix, where `dim` equals the number
    /// of coefficients in the parameter block. Implementations should add
    /// their contribution — the caller is responsible for zeroing `gram`
    /// before the first call.
    fn add_penalty_matrix(&self, dim: usize, gram: &mut [f64]);
}

#[inline]
fn debug_assert_matrix_shape(dim: usize, gram: &[f64]) {
    debug_assert_eq!(dim.checked_mul(dim), Some(gram.len()));
}

macro_rules! impl_global_penalty_tuple {
    (types = ($($ty:ident),+); indices = ($($idx:tt),+)) => {
        impl<$($ty,)+> GlobalPenalty for ($($ty,)+)
        where
            $($ty: GlobalPenalty,)+
        {
            #[inline]
            fn value(&self, beta: &[f64]) -> f64 {
                0.0 $(+ self.$idx.value(beta))+
            }

            #[inline]
            fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
                $(self.$idx.add_gradient(beta, grad);)+
            }
        }
    };
}

macro_rules! impl_penalty_tuple {
    (types = ($($ty:ident),+); indices = ($($idx:tt),+)) => {
        impl<$($ty,)+> Penalty for ($($ty,)+)
        where
            $($ty: Penalty,)+
        {
            #[inline]
            fn value(&self, beta: &[f64]) -> f64 {
                0.0 $(+ self.$idx.value(beta))+
            }

            #[inline]
            fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
                $(self.$idx.add_gradient(beta, grad);)+
            }
        }
    };
}

impl_penalty_tuple!(types = (P1); indices = (0));
impl_penalty_tuple!(types = (P1, P2); indices = (0, 1));
impl_penalty_tuple!(types = (P1, P2, P3); indices = (0, 1, 2));
impl_penalty_tuple!(types = (P1, P2, P3, P4); indices = (0, 1, 2, 3));
impl_penalty_tuple!(types = (P1, P2, P3, P4, P5); indices = (0, 1, 2, 3, 4));
impl_penalty_tuple!(types = (P1, P2, P3, P4, P5, P6); indices = (0, 1, 2, 3, 4, 5));
impl_penalty_tuple!(types = (P1, P2, P3, P4, P5, P6, P7); indices = (0, 1, 2, 3, 4, 5, 6));
impl_penalty_tuple!(types = (P1, P2, P3, P4, P5, P6, P7, P8); indices = (0, 1, 2, 3, 4, 5, 6, 7));

impl_global_penalty_tuple!(types = (P1); indices = (0));
impl_global_penalty_tuple!(types = (P1, P2); indices = (0, 1));
impl_global_penalty_tuple!(types = (P1, P2, P3); indices = (0, 1, 2));
impl_global_penalty_tuple!(types = (P1, P2, P3, P4); indices = (0, 1, 2, 3));
impl_global_penalty_tuple!(types = (P1, P2, P3, P4, P5); indices = (0, 1, 2, 3, 4));
impl_global_penalty_tuple!(types = (P1, P2, P3, P4, P5, P6); indices = (0, 1, 2, 3, 4, 5));
impl_global_penalty_tuple!(types = (P1, P2, P3, P4, P5, P6, P7); indices = (0, 1, 2, 3, 4, 5, 6));
impl_global_penalty_tuple!(types = (P1, P2, P3, P4, P5, P6, P7, P8); indices = (0, 1, 2, 3, 4, 5, 6, 7));

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{
        AbsoluteLimitPenalty, GlobalPenalty, HingeQuadraticPenalty, LinearForm, LinearFormBuilder,
        LinearTerm, MatrixPenalty, NoPenalty, Penalty, RidgePenalty, SegmentPenalty,
    };

    #[derive(Debug, Clone, Copy)]
    struct LinearPenalty(f64);

    impl Penalty for LinearPenalty {
        fn value(&self, beta: &[f64]) -> f64 {
            self.0 * beta.iter().sum::<f64>()
        }

        fn add_gradient(&self, _: &[f64], grad: &mut [f64]) {
            for value in grad {
                *value += self.0;
            }
        }
    }

    impl GlobalPenalty for LinearPenalty {
        fn value(&self, beta: &[f64]) -> f64 {
            self.0 * beta.iter().sum::<f64>()
        }

        fn add_gradient(&self, _: &[f64], grad: &mut [f64]) {
            for value in grad {
                *value += self.0;
            }
        }
    }

    #[test]
    fn eight_penalty_tuple_adds_values_and_gradients() {
        let penalty = (
            LinearPenalty(1.0),
            LinearPenalty(2.0),
            LinearPenalty(3.0),
            LinearPenalty(4.0),
            LinearPenalty(5.0),
            LinearPenalty(6.0),
            LinearPenalty(7.0),
            LinearPenalty(8.0),
        );
        let beta = [2.0];
        let mut grad = [1.0];

        assert_relative_eq!(Penalty::value(&penalty, &beta), 72.0);
        Penalty::add_gradient(&penalty, &beta, &mut grad);
        assert_relative_eq!(grad[0], 37.0);
    }

    #[test]
    fn segment_penalty_applies_value_to_selected_range() {
        let penalty = SegmentPenalty::new(1..4, RidgePenalty::new(2.0));
        let beta = [10.0, 1.0, -2.0, 3.0, 20.0];

        assert_eq!(penalty.range(), 1..4);
        assert_eq!(penalty.penalty(), &RidgePenalty::new(2.0));
        assert_relative_eq!(
            penalty.value(&beta),
            RidgePenalty::new(2.0).value(&beta[1..4])
        );
    }

    #[test]
    fn segment_penalty_adds_gradient_only_inside_selected_range() {
        let penalty = SegmentPenalty::new(1..4, RidgePenalty::new(2.0));
        let beta = [10.0, 1.0, -2.0, 3.0, 20.0];
        let mut grad = [100.0, 0.0, 0.0, 0.0, 200.0];

        penalty.add_gradient(&beta, &mut grad);

        assert_relative_eq!(grad[0], 100.0);
        assert_relative_eq!(grad[1], 4.0);
        assert_relative_eq!(grad[2], -8.0);
        assert_relative_eq!(grad[3], 12.0);
        assert_relative_eq!(grad[4], 200.0);
    }

    #[test]
    fn segment_penalty_tuples_compose_disjoint_ranges() {
        let penalty = (
            SegmentPenalty::new(0..2, RidgePenalty::new(1.0)),
            SegmentPenalty::new(2..4, LinearPenalty(3.0)),
        );
        let beta = [1.0, 2.0, 3.0, 4.0];
        let mut grad = [0.0; 4];

        assert_relative_eq!(Penalty::value(&penalty, &beta), 5.0 + 21.0);
        Penalty::add_gradient(&penalty, &beta, &mut grad);

        assert_relative_eq!(grad[0], 2.0);
        assert_relative_eq!(grad[1], 4.0);
        assert_relative_eq!(grad[2], 3.0);
        assert_relative_eq!(grad[3], 3.0);
    }

    #[test]
    fn eight_global_penalty_tuple_adds_values_and_gradients() {
        let penalty = (
            LinearPenalty(1.0),
            LinearPenalty(2.0),
            LinearPenalty(3.0),
            LinearPenalty(4.0),
            LinearPenalty(5.0),
            LinearPenalty(6.0),
            LinearPenalty(7.0),
            LinearPenalty(8.0),
        );
        let beta = [2.0];
        let mut grad = [1.0];

        assert_relative_eq!(GlobalPenalty::value(&penalty, &beta), 72.0);
        GlobalPenalty::add_gradient(&penalty, &beta, &mut grad);
        assert_relative_eq!(grad[0], 37.0);
    }

    #[test]
    fn linear_form_evaluates_full_beta_vector_terms() {
        let form = LinearForm::new(vec![LinearTerm::new(2, 0.5), LinearTerm::new(0, -2.0)], 1.0);
        let beta = [3.0, 10.0, 8.0];

        assert_relative_eq!(form.value(&beta), -1.0);
    }

    #[test]
    fn linear_form_builder_adds_terms_weighted_terms_and_constant() {
        let form = LinearForm::builder()
            .term(2, 0.5)
            .weighted_terms(0, [-2.0, 0.25])
            .weighted_range(3..5, [1.25, -1.5, 100.0])
            .terms([LinearTerm::new(1, 0.75)])
            .constant(1.0)
            .build();
        let explicit = LinearForm::new(
            vec![
                LinearTerm::new(2, 0.5),
                LinearTerm::new(0, -2.0),
                LinearTerm::new(1, 0.25),
                LinearTerm::new(3, 1.25),
                LinearTerm::new(4, -1.5),
                LinearTerm::new(1, 0.75),
            ],
            1.0,
        );
        let beta = [3.0, 10.0, 8.0, -2.0, 0.5];

        assert_eq!(
            LinearFormBuilder::new().build(),
            LinearForm::new(Vec::new(), 0.0)
        );
        assert_eq!(form, explicit);
        assert_relative_eq!(form.value(&beta), explicit.value(&beta));
    }

    #[test]
    fn linear_form_helpers_create_global_penalties() {
        let hinge = LinearForm::builder()
            .term(0, 1.0)
            .constant(-0.5)
            .build()
            .hinge_le(2.0);
        let limit = LinearForm::builder()
            .term(1, -1.0)
            .build()
            .absolute_limit(3.0, 2.0, 0.5);
        let beta = [1.0, -1.0];

        assert_relative_eq!(hinge.value(&beta), 0.5);
        assert_relative_eq!(limit.value(&beta), 27.0);
    }

    #[test]
    fn hinge_quadratic_penalty_gradient_matches_finite_difference() {
        let penalty = HingeQuadraticPenalty::new(
            LinearForm::new(
                vec![LinearTerm::new(0, 1.0), LinearTerm::new(2, -0.5)],
                -0.1,
            ),
            3.0,
        );
        let beta = [1.0, -2.0, 0.4];

        assert_global_penalty_gradient_matches_finite_difference(&penalty, &beta);
    }

    #[test]
    fn hinge_quadratic_penalty_ignores_nonpositive_side_and_invalid_weight() {
        let inactive =
            HingeQuadraticPenalty::new(LinearForm::new(vec![LinearTerm::new(0, 1.0)], -2.0), 3.0);
        let invalid = HingeQuadraticPenalty::new(
            LinearForm::new(vec![LinearTerm::new(0, 1.0)], 0.0),
            f64::NAN,
        );
        let beta = [1.0];

        for penalty in [inactive, invalid] {
            let mut grad = [5.0];
            assert_eq!(penalty.value(&beta), 0.0);
            penalty.add_gradient(&beta, &mut grad);
            assert_eq!(grad, [5.0]);
        }
    }

    #[test]
    fn hinge_quadratic_penalty_propagates_nan_form_values() {
        let penalty =
            HingeQuadraticPenalty::new(LinearForm::new(vec![LinearTerm::new(0, 1.0)], 0.0), 3.0);
        let beta = [f64::NAN];
        let mut grad = [0.0];

        assert!(penalty.value(&beta).is_nan());
        penalty.add_gradient(&beta, &mut grad);
        assert!(grad[0].is_nan());
    }

    #[test]
    fn absolute_limit_penalty_gradient_matches_finite_difference() {
        let penalty = AbsoluteLimitPenalty::new(
            LinearForm::new(
                vec![LinearTerm::new(0, 1.0), LinearTerm::new(1, -2.0)],
                0.25,
            ),
            5.0,
            1.5,
            0.4,
        );
        let beta = [0.8, -0.2];

        assert_global_penalty_gradient_matches_finite_difference(&penalty, &beta);
    }

    #[test]
    fn absolute_limit_penalty_ignores_inactive_and_invalid_inputs() {
        let inactive = AbsoluteLimitPenalty::new(
            LinearForm::new(vec![LinearTerm::new(0, 1.0)], 0.0),
            3.0,
            1.0,
            2.0,
        );
        let invalid = AbsoluteLimitPenalty::new(
            LinearForm::new(vec![LinearTerm::new(0, 1.0)], 0.0),
            3.0,
            f64::NAN,
            2.0,
        );
        let beta = [1.0];

        for penalty in [inactive, invalid] {
            let mut grad = [5.0];
            assert_eq!(penalty.value(&beta), 0.0);
            penalty.add_gradient(&beta, &mut grad);
            assert_eq!(grad, [5.0]);
        }
    }

    #[test]
    fn absolute_limit_penalty_propagates_nan_form_values() {
        let penalty = AbsoluteLimitPenalty::new(
            LinearForm::new(vec![LinearTerm::new(0, 1.0)], 0.0),
            3.0,
            1.0,
            0.5,
        );
        let beta = [f64::NAN];
        let mut grad = [0.0];

        assert!(penalty.value(&beta).is_nan());
        penalty.add_gradient(&beta, &mut grad);
        assert!(grad[0].is_nan());
    }

    #[test]
    fn global_linear_penalty_tuple_composes_values_and_gradients() {
        let penalty = (
            HingeQuadraticPenalty::new(LinearForm::new(vec![LinearTerm::new(0, 1.0)], -0.5), 2.0),
            AbsoluteLimitPenalty::new(
                LinearForm::new(vec![LinearTerm::new(1, -1.0)], 0.0),
                3.0,
                2.0,
                0.5,
            ),
        );
        let beta = [1.0, -1.0];
        let mut grad = [0.0, 0.0];

        assert_relative_eq!(GlobalPenalty::value(&penalty, &beta), 0.5 + 27.0);
        GlobalPenalty::add_gradient(&penalty, &beta, &mut grad);

        assert_relative_eq!(grad[0], 2.0);
        assert_relative_eq!(grad[1], -72.0);
    }

    #[test]
    fn no_penalty_matrix_adds_nothing() {
        let mut gram = vec![1.0, 2.0, 3.0, 4.0];
        NoPenalty.add_penalty_matrix(2, &mut gram);
        assert_eq!(gram, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matrix_penalties_accept_empty_matrices() {
        let mut gram = Vec::new();

        NoPenalty.add_penalty_matrix(0, &mut gram);
        RidgePenalty::new(3.0).add_penalty_matrix(0, &mut gram);

        assert!(gram.is_empty());
    }

    #[test]
    fn ridge_penalty_matrix_adds_lambda_to_diagonal() {
        let penalty = RidgePenalty::new(3.0);
        // 3x3 Gram matrix: [[1,2,3], [4,5,6], [7,8,9]]
        let mut gram = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        penalty.add_penalty_matrix(3, &mut gram);
        // Diagonal += lambda: [4, 2, 3, 4, 8, 6, 7, 8, 12]
        assert_eq!(gram[0], 4.0);
        assert_eq!(gram[4], 8.0);
        assert_eq!(gram[8], 12.0);
        // Off-diagonal unchanged
        assert_eq!(gram[1], 2.0);
        assert_eq!(gram[2], 3.0);
        assert_eq!(gram[3], 4.0);
        assert_eq!(gram[5], 6.0);
        assert_eq!(gram[6], 7.0);
        assert_eq!(gram[7], 8.0);
    }

    fn assert_global_penalty_gradient_matches_finite_difference<P>(penalty: &P, beta: &[f64])
    where
        P: GlobalPenalty,
    {
        let epsilon = 1.0e-6;
        let mut grad = vec![0.0; beta.len()];
        penalty.add_gradient(beta, &mut grad);

        for index in 0..beta.len() {
            let mut plus = beta.to_vec();
            plus[index] += epsilon;
            let mut minus = beta.to_vec();
            minus[index] -= epsilon;
            let finite_difference =
                (penalty.value(&plus) - penalty.value(&minus)) / (2.0 * epsilon);

            assert_relative_eq!(grad[index], finite_difference, epsilon = 1.0e-6);
        }
    }
}
