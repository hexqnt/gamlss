/// Supported spline degrees/orders for efficient local basis computation.
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
    /// Polynomial degree.
    #[inline(always)]
    pub fn degree(self) -> usize {
        self as usize
    }

    /// Minimum number of coefficients for the given order.
    #[inline(always)]
    pub fn min_basis(self) -> usize {
        self.degree() + 1
    }
}
