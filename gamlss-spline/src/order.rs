/// Поддерживаемые степени/порядки сплайнов для эффективного локального
/// вычисления базиса.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplineOrder {
    /// Linear B-spline.
    Linear = 1,
    /// Quadratic B-spline.
    Quadratic = 2,
    /// Cubic B-spline.
    Cubic = 3,
}

impl SplineOrder {
    /// Степень полинома.
    #[inline(always)]
    pub fn degree(self) -> usize {
        self as usize
    }

    /// Минимальное число коэффициентов для данного порядка.
    #[inline(always)]
    pub fn min_basis(self) -> usize {
        self.degree() + 1
    }
}
