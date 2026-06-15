/// Penalty для коэффициентов одного parameter block.
///
/// Implementations receive the local coefficient slice for one parameter
/// block. The model validates slice lengths before evaluation where possible;
/// hot-path implementations may use debug assertions for length checks.
pub trait Penalty {
    /// Значение penalty для текущих коэффициентов.
    fn value(&self, beta: &[f64]) -> f64;
    /// Добавляет градиент penalty в уже существующий `grad`.
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

/// Нулевая penalty.
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

/// Ridge penalty `lambda * sum(beta_i^2)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RidgePenalty {
    /// Вес регуляризации.
    pub lambda: f64,
}

impl RidgePenalty {
    /// Создаёт ridge penalty с заданным `lambda`.
    #[must_use]
    pub const fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Penalty for RidgePenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        self.lambda * beta.iter().map(|value| value * value).sum::<f64>()
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(beta.len(), grad.len());

        let scale = 2.0 * self.lambda;
        for (grad_value, beta_value) in grad.iter_mut().zip(beta) {
            *grad_value = scale.mul_add(*beta_value, *grad_value);
        }
    }
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
    fn add_penalty_matrix(&self, gram: &mut [f64]);
}

impl MatrixPenalty for NoPenalty {
    fn add_penalty_matrix(&self, _gram: &mut [f64]) {}
}

impl MatrixPenalty for RidgePenalty {
    fn add_penalty_matrix(&self, gram: &mut [f64]) {
        let dim = (gram.len() as f64).sqrt() as usize;
        for (row, row_values) in gram.chunks_exact_mut(dim).enumerate() {
            row_values[row] += self.lambda;
        }
    }
}

macro_rules! impl_global_penalty_tuple {
    (types = ($($ty:ident),+); indices = ($($idx:tt),+)) => {
        impl<$($ty,)+> GlobalPenalty for ($($ty,)+)
        where
            $($ty: GlobalPenalty,)+
        {
            fn value(&self, beta: &[f64]) -> f64 {
                0.0 $(+ self.$idx.value(beta))+
            }

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
            fn value(&self, beta: &[f64]) -> f64 {
                0.0 $(+ self.$idx.value(beta))+
            }

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

    use super::{GlobalPenalty, MatrixPenalty, NoPenalty, Penalty, RidgePenalty};

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
    fn no_penalty_matrix_adds_nothing() {
        let mut gram = vec![1.0, 2.0, 3.0, 4.0];
        NoPenalty.add_penalty_matrix(&mut gram);
        assert_eq!(gram, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ridge_penalty_matrix_adds_lambda_to_diagonal() {
        let penalty = RidgePenalty::new(3.0);
        // 3x3 Gram matrix: [[1,2,3], [4,5,6], [7,8,9]]
        let mut gram = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        penalty.add_penalty_matrix(&mut gram);
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
}
