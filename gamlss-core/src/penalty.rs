/// Penalty для коэффициентов одного parameter block.
pub trait Penalty {
    /// Значение penalty для текущих коэффициентов.
    fn value(&self, beta: &[f64]) -> f64;
    /// Добавляет градиент penalty в уже существующий `grad`.
    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]);
}

/// Penalty evaluated on the full model parameter vector.
///
/// This is useful for constraints or regularization coupling several parameter
/// blocks, while [`Penalty`] remains the local per-block mechanism.
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

impl_global_penalty_tuple!(types = (P1); indices = (0));
impl_global_penalty_tuple!(types = (P1, P2); indices = (0, 1));
impl_global_penalty_tuple!(types = (P1, P2, P3); indices = (0, 1, 2));
impl_global_penalty_tuple!(types = (P1, P2, P3, P4); indices = (0, 1, 2, 3));
impl_global_penalty_tuple!(types = (P1, P2, P3, P4, P5); indices = (0, 1, 2, 3, 4));
impl_global_penalty_tuple!(types = (P1, P2, P3, P4, P5, P6); indices = (0, 1, 2, 3, 4, 5));
