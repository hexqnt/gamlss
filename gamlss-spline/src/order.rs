/// Supported polynomial degrees for efficient local basis computation.
///
/// The public type retains the name “order”, but [`SplineOrder::degree`] returns the numeric enum value: one for linear, two for quadratic, and three for cubic splines.
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
    #[inline]
    #[must_use]
    pub const fn degree(self) -> usize {
        self as usize
    }

    /// Minimum number of coefficients for the given order.
    #[inline]
    #[must_use]
    pub const fn min_basis(self) -> usize {
        self.degree() + 1
    }
}
