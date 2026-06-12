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
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Penalty for RidgePenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        self.lambda * beta.iter().map(|value| value * value).sum::<f64>()
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(beta.len(), grad.len());

        for (grad_value, beta_value) in grad.iter_mut().zip(beta) {
            *grad_value += 2.0 * self.lambda * beta_value;
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

    use super::{GlobalPenalty, Penalty};

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
}
