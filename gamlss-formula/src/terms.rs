use std::ops::{Add, Range};

use gamlss_spline::{
    CyclicSplineSpec, ISplineBasis, MonotoneDirection, OpenUniformSplineBasis, SplineOrder,
};

use crate::{Category, Col};

/// Pre-data term specification.
#[derive(Debug, Clone, PartialEq)]
pub enum TermSpec {
    /// Intercept column of ones.
    Intercept,
    /// Linear term backed by one numeric column.
    Linear {
        /// Source column.
        col: Col<f64>,
    },
    /// Numeric offset added to the predictor without adding coefficients.
    Offset {
        /// Source column.
        col: Col<f64>,
    },
    /// Boolean indicator term.
    Indicator {
        /// Source column.
        col: Col<bool>,
    },
    /// Treatment-coded categorical term.
    Factor {
        /// Source column.
        col: Col<Category>,
    },
    /// Product of two numeric columns.
    Interaction {
        /// Left source column.
        left: Col<f64>,
        /// Right source column.
        right: Col<f64>,
    },
    /// Open-uniform P-spline term.
    PSpline(PSplineTerm),
    /// Cyclic P-spline term.
    CyclicPSpline(CyclicPSplineTerm),
    /// Fourier seasonal term.
    Fourier(FourierTerm),
    /// Tensor-product open-uniform P-spline term.
    TensorPSpline(TensorPSplineTerm),
    /// Hard-monotone I-spline term.
    Monotone(MonotoneTerm),
}

/// Pre-data term expression for one parameter predictor.
#[derive(Debug, Clone, PartialEq)]
pub struct TermExpr {
    terms: Vec<TermSpec>,
    default_intercept_if_empty: bool,
}

impl TermExpr {
    /// Creates an expression from one term.
    #[must_use]
    pub fn new(term: TermSpec) -> Self {
        Self {
            terms: vec![term],
            default_intercept_if_empty: false,
        }
    }

    /// Creates an explicit empty expression with no implicit intercept.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            terms: Vec::new(),
            default_intercept_if_empty: false,
        }
    }

    pub(crate) fn into_terms(mut self) -> Vec<TermSpec> {
        if self.terms.is_empty() && self.default_intercept_if_empty {
            self.terms.push(TermSpec::Intercept);
        }
        self.terms
    }
}

impl Default for TermExpr {
    fn default() -> Self {
        Self {
            terms: Vec::new(),
            default_intercept_if_empty: true,
        }
    }
}

impl Add for TermExpr {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        self.terms.append(&mut rhs.terms);
        if !rhs.default_intercept_if_empty {
            self.default_intercept_if_empty = false;
        }
        self
    }
}

/// Creates an intercept term expression.
#[must_use]
pub fn intercept() -> TermExpr {
    TermExpr::new(TermSpec::Intercept)
}

/// Creates an explicit empty term expression without an intercept.
#[must_use]
pub fn no_intercept() -> TermExpr {
    TermExpr::empty()
}

/// Creates a linear term expression.
#[must_use]
pub fn linear(col: Col<f64>) -> TermExpr {
    TermExpr::new(TermSpec::Linear { col })
}

/// Creates a numeric offset term expression.
#[must_use]
pub fn offset(col: Col<f64>) -> TermExpr {
    TermExpr::new(TermSpec::Offset { col })
}

/// Creates a boolean indicator term expression.
#[must_use]
pub fn indicator(col: Col<bool>) -> TermExpr {
    TermExpr::new(TermSpec::Indicator { col })
}

/// Creates a categorical factor term expression.
#[must_use]
pub fn factor(col: Col<Category>) -> TermExpr {
    TermExpr::new(TermSpec::Factor { col })
}

/// Creates a numeric product interaction term expression.
#[must_use]
pub fn interaction(left: Col<f64>, right: Col<f64>) -> TermExpr {
    TermExpr::new(TermSpec::Interaction { left, right })
}

/// Creates an open-uniform P-spline term expression with default options.
#[must_use]
pub fn pspline(col: Col<f64>) -> PSplineTerm {
    PSplineTerm {
        col,
        k: 20,
        order: SplineOrder::Cubic,
        lambda: 1.0,
        penalty_order: 2,
    }
}

/// Creates a cyclic P-spline term expression with default options.
#[must_use]
pub fn cyclic_pspline(col: Col<f64>) -> CyclicPSplineTerm {
    CyclicPSplineTerm {
        col,
        k: 20,
        order: SplineOrder::Cubic,
        lambda: 1.0,
        penalty_order: 2,
    }
}

/// Creates a Fourier term expression with default options.
#[must_use]
pub fn fourier(col: Col<f64>) -> FourierTerm {
    FourierTerm {
        col,
        period: 1.0,
        order: 1,
        include_intercept: false,
    }
}

/// Creates a tensor-product P-spline term expression with default options.
#[must_use]
pub fn tensor_pspline(left: Col<f64>, right: Col<f64>) -> TensorPSplineTerm {
    TensorPSplineTerm {
        left,
        right,
        left_k: 10,
        right_k: 10,
        left_order: SplineOrder::Cubic,
        right_order: SplineOrder::Cubic,
    }
}

/// Creates a hard-monotone I-spline term expression with default options.
#[must_use]
pub fn monotone(col: Col<f64>) -> MonotoneTerm {
    MonotoneTerm {
        col,
        k: 20,
        degree: SplineOrder::Cubic.degree(),
        direction: MonotoneDirection::Increasing,
    }
}

macro_rules! impl_term_expr {
    ($term:ty, $variant:ident) => {
        impl From<$term> for TermExpr {
            fn from(value: $term) -> Self {
                Self::new(TermSpec::$variant(value))
            }
        }

        impl Add<$term> for TermExpr {
            type Output = TermExpr;

            fn add(self, rhs: $term) -> Self::Output {
                self + TermExpr::from(rhs)
            }
        }

        impl<Rhs> Add<Rhs> for $term
        where
            Rhs: Into<TermExpr>,
        {
            type Output = TermExpr;

            fn add(self, rhs: Rhs) -> Self::Output {
                TermExpr::from(self) + rhs.into()
            }
        }
    };
}

/// Open-uniform P-spline term options.
#[derive(Debug, Clone, PartialEq)]
pub struct PSplineTerm {
    pub(crate) col: Col<f64>,
    pub(crate) k: usize,
    pub(crate) order: SplineOrder,
    pub(crate) lambda: f64,
    pub(crate) penalty_order: usize,
}

impl PSplineTerm {
    /// Returns the source column.
    #[must_use]
    pub fn col(&self) -> &Col<f64> {
        &self.col
    }

    /// Returns the number of spline basis functions.
    #[must_use]
    pub fn n_basis(&self) -> usize {
        self.k
    }

    /// Returns the spline order.
    #[must_use]
    pub fn spline_order(&self) -> SplineOrder {
        self.order
    }

    /// Returns the difference penalty weight.
    #[must_use]
    pub fn penalty_lambda(&self) -> f64 {
        self.lambda
    }

    /// Returns the finite-difference penalty order.
    #[must_use]
    pub fn difference_order(&self) -> usize {
        self.penalty_order
    }

    /// Sets the number of spline basis functions.
    #[must_use]
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the spline order.
    #[must_use]
    pub fn order(mut self, order: SplineOrder) -> Self {
        self.order = order;
        self
    }

    /// Sets the difference penalty weight.
    #[must_use]
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Sets the finite-difference penalty order.
    #[must_use]
    pub fn penalty_order(mut self, order: usize) -> Self {
        self.penalty_order = order;
        self
    }
}

/// Cyclic P-spline term options.
#[derive(Debug, Clone, PartialEq)]
pub struct CyclicPSplineTerm {
    pub(crate) col: Col<f64>,
    pub(crate) k: usize,
    pub(crate) order: SplineOrder,
    pub(crate) lambda: f64,
    pub(crate) penalty_order: usize,
}

impl CyclicPSplineTerm {
    /// Sets the number of spline basis functions.
    #[must_use]
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the spline order.
    #[must_use]
    pub fn order(mut self, order: SplineOrder) -> Self {
        self.order = order;
        self
    }

    /// Sets the cyclic difference penalty weight.
    #[must_use]
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Sets the finite-difference penalty order.
    #[must_use]
    pub fn penalty_order(mut self, order: usize) -> Self {
        self.penalty_order = order;
        self
    }
}

/// Fourier term options.
#[derive(Debug, Clone, PartialEq)]
pub struct FourierTerm {
    pub(crate) col: Col<f64>,
    pub(crate) period: f64,
    pub(crate) order: usize,
    pub(crate) include_intercept: bool,
}

impl FourierTerm {
    /// Sets the Fourier period.
    #[must_use]
    pub fn period(mut self, period: f64) -> Self {
        self.period = period;
        self
    }

    /// Sets the Fourier order.
    #[must_use]
    pub fn order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Includes an intercept column in this term.
    #[must_use]
    pub fn include_intercept(mut self, include: bool) -> Self {
        self.include_intercept = include;
        self
    }
}

/// Tensor-product P-spline term options.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorPSplineTerm {
    pub(crate) left: Col<f64>,
    pub(crate) right: Col<f64>,
    pub(crate) left_k: usize,
    pub(crate) right_k: usize,
    pub(crate) left_order: SplineOrder,
    pub(crate) right_order: SplineOrder,
}

impl TensorPSplineTerm {
    /// Sets basis counts for the left and right axes.
    #[must_use]
    pub fn k(mut self, left: usize, right: usize) -> Self {
        self.left_k = left;
        self.right_k = right;
        self
    }

    /// Sets spline orders for the left and right axes.
    #[must_use]
    pub fn order(mut self, left: SplineOrder, right: SplineOrder) -> Self {
        self.left_order = left;
        self.right_order = right;
        self
    }
}

/// Hard-monotone I-spline term options.
#[derive(Debug, Clone, PartialEq)]
pub struct MonotoneTerm {
    pub(crate) col: Col<f64>,
    pub(crate) k: usize,
    pub(crate) degree: usize,
    pub(crate) direction: MonotoneDirection,
}

impl MonotoneTerm {
    /// Sets the number of I-spline basis functions.
    #[must_use]
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the I-spline degree.
    #[must_use]
    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Sets the monotonicity direction.
    #[must_use]
    pub fn direction(mut self, direction: MonotoneDirection) -> Self {
        self.direction = direction;
        self
    }
}

impl_term_expr!(PSplineTerm, PSpline);
impl_term_expr!(CyclicPSplineTerm, CyclicPSpline);
impl_term_expr!(FourierTerm, Fourier);
impl_term_expr!(TensorPSplineTerm, TensorPSpline);
impl_term_expr!(MonotoneTerm, Monotone);

/// Fitted term metadata reusable for prediction.
#[derive(Debug, Clone, PartialEq)]
pub enum FittedTerm {
    /// Intercept term.
    Intercept {
        /// Local coefficient range inside the parameter block.
        range: Range<usize>,
        /// Coefficient name.
        coefficient: String,
    },
    /// Linear term.
    Linear {
        /// Source column.
        col: Col<f64>,
        /// Local coefficient range inside the parameter block.
        range: Range<usize>,
        /// Coefficient name.
        coefficient: String,
    },
    /// Numeric offset term.
    Offset {
        /// Source column.
        col: Col<f64>,
    },
    /// Boolean indicator term.
    Indicator {
        /// Source column.
        col: Col<bool>,
        /// Local coefficient range inside the parameter block.
        range: Range<usize>,
        /// Coefficient name.
        coefficient: String,
    },
    /// Treatment-coded categorical factor.
    Factor {
        /// Source column.
        col: Col<Category>,
        /// Local coefficient range inside the parameter block.
        range: Range<usize>,
        /// Sorted training levels.
        levels: Vec<String>,
        /// Baseline level.
        baseline: String,
        /// Coefficient names.
        coefficients: Vec<String>,
    },
    /// Numeric product interaction.
    Interaction {
        /// Left source column.
        left: Col<f64>,
        /// Right source column.
        right: Col<f64>,
        /// Local coefficient range inside the parameter block.
        range: Range<usize>,
        /// Coefficient name.
        coefficient: String,
    },
    /// P-spline term with fitted basis metadata.
    PSpline {
        /// Source column.
        col: Col<f64>,
        /// Local coefficient range inside the parameter block.
        range: Range<usize>,
        /// Fitted spline basis metadata.
        basis: OpenUniformSplineBasis,
        /// Penalty weight.
        lambda: f64,
        /// Difference penalty order.
        penalty_order: usize,
        /// Coefficient names for this term.
        coefficients: Vec<String>,
    },
    /// Cyclic P-spline term with fitted metadata.
    CyclicPSpline {
        /// Source column.
        col: Col<f64>,
        /// Local coefficient range inside the parameter block.
        range: Range<usize>,
        /// Cyclic spline metadata.
        spec: CyclicSplineSpec,
        /// Penalty weight.
        lambda: f64,
        /// Difference penalty order.
        penalty_order: usize,
        /// Coefficient names for this term.
        coefficients: Vec<String>,
    },
    /// Fourier term with fitted metadata.
    Fourier {
        /// Source column.
        col: Col<f64>,
        /// Local coefficient range inside the parameter block.
        range: Range<usize>,
        /// Period.
        period: f64,
        /// Fourier order.
        order: usize,
        /// Whether this term includes an intercept coefficient.
        include_intercept: bool,
        /// Coefficient names for this term.
        coefficients: Vec<String>,
    },
    /// Tensor-product P-spline metadata.
    TensorPSpline {
        /// Left source column.
        left: Col<f64>,
        /// Right source column.
        right: Col<f64>,
        /// Local coefficient range inside the parameter block.
        range: Range<usize>,
        /// Left fitted basis metadata.
        left_basis: OpenUniformSplineBasis,
        /// Right fitted basis metadata.
        right_basis: OpenUniformSplineBasis,
        /// Coefficient names for this term.
        coefficients: Vec<String>,
    },
    /// Hard-monotone I-spline metadata.
    Monotone {
        /// Source column.
        col: Col<f64>,
        /// Local coefficient range inside the parameter block.
        range: Range<usize>,
        /// Fitted I-spline basis metadata.
        basis: ISplineBasis,
        /// Monotonicity direction.
        direction: MonotoneDirection,
        /// Coefficient names for this term.
        coefficients: Vec<String>,
    },
}

impl FittedTerm {
    /// Local coefficient range inside the parameter block.
    #[must_use]
    pub fn range(&self) -> Range<usize> {
        match self {
            Self::Intercept { range, .. }
            | Self::Linear { range, .. }
            | Self::Indicator { range, .. }
            | Self::Factor { range, .. }
            | Self::Interaction { range, .. }
            | Self::PSpline { range, .. }
            | Self::CyclicPSpline { range, .. }
            | Self::Fourier { range, .. }
            | Self::TensorPSpline { range, .. }
            | Self::Monotone { range, .. } => range.clone(),
            Self::Offset { .. } => 0..0,
        }
    }

    /// Appends coefficient names for this term.
    pub(crate) fn append_coefficient_names<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Self::Intercept { coefficient, .. }
            | Self::Linear { coefficient, .. }
            | Self::Indicator { coefficient, .. }
            | Self::Interaction { coefficient, .. } => out.push(coefficient.as_str()),
            Self::PSpline { coefficients, .. }
            | Self::CyclicPSpline { coefficients, .. }
            | Self::Fourier { coefficients, .. }
            | Self::Factor { coefficients, .. }
            | Self::TensorPSpline { coefficients, .. }
            | Self::Monotone { coefficients, .. } => {
                out.extend(coefficients.iter().map(String::as_str));
            }
            Self::Offset { .. } => {}
        }
    }
}
