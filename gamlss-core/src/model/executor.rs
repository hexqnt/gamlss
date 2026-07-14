use std::ops::Range;

use crate::{
    LowerTriangularParameterBlock, ModelError, ParameterBlock, ParameterName, Penalty,
    PredictorBlock, SimplexLogitParameterBlock, StrictLowerTriangularParameterBlock,
    VectorParameterBlock,
    shape::{
        Broadcast, Lower, ParameterShape, Product, Repeated, Scalar, ScalarTuple, Simplex,
        StrictLower, Vector,
    },
};

use super::{
    GradientWorkspace, ParameterAxis, ParameterDescriptor, ParameterPath, add_into,
    validate_block_rows,
};

/// Sealed execution contract between static shape topology and concrete blocks.
pub trait ShapeBlocks<S: ParameterShape> {
    fn nrows(&self) -> Option<usize>;
    fn try_len(&self) -> Result<usize, ModelError>;
    fn validate(&self, nobs: usize) -> Result<(), ModelError>;
    fn values_row(&self, beta: &[f64], row: usize) -> S::Values;
    fn penalty_value(&self, beta: &[f64]) -> f64;
    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]);
    fn set_initial(&self, values: &S::Values, beta: &mut [f64]);
    fn leaf_count(&self) -> usize;
    fn prepare_workspace(&self, nobs: usize, workspace: &mut GradientWorkspace, cursor: &mut usize);
    fn set_scores(
        &self,
        scores: &S::Values,
        row: usize,
        weight: f64,
        workspace: &mut GradientWorkspace,
        cursor: &mut usize,
    );
    fn backprop(
        &self,
        beta: &[f64],
        grad: &mut [f64],
        workspace: &mut GradientWorkspace,
        cursor: &mut usize,
    );
    fn visit_descriptors<V>(&self, prefix: &ParameterPath, cursor: &mut usize, visit: &mut V)
    where
        V: FnMut(usize, ParameterDescriptor);
    fn visit_slices<V>(&self, cursor: &mut usize, visit: &mut V)
    where
        V: FnMut(usize, &'static str, Range<usize>);
}

impl<P, X, Pen> ShapeBlocks<Scalar<P>> for ParameterBlock<P, X, Pen>
where
    P: ParameterName,
    X: PredictorBlock,
    Pen: Penalty,
{
    fn nrows(&self) -> Option<usize> {
        Some(self.x().nrows())
    }
    fn try_len(&self) -> Result<usize, ModelError> {
        Ok(self.try_range()?.end)
    }
    fn validate(&self, n: usize) -> Result<(), ModelError> {
        self.x().validate()?;
        validate_block_rows(P::NAME, self.x().nrows(), n)?;
        self.penalty().validate_dim(self.len())?;
        self.try_range()?;
        Ok(())
    }
    fn values_row(&self, beta: &[f64], row: usize) -> f64 {
        self.x().eta_row(row, &beta[self.range()])
    }
    fn penalty_value(&self, beta: &[f64]) -> f64 {
        self.penalty().value(&beta[self.range()])
    }
    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        let r = self.range();
        self.penalty().add_gradient(&beta[r.clone()], &mut grad[r]);
    }
    fn set_initial(&self, v: &f64, beta: &mut [f64]) {
        if v.is_finite() {
            self.x().set_constant_start(*v, &mut beta[self.range()]);
        }
    }
    fn leaf_count(&self) -> usize {
        1
    }
    fn prepare_workspace(&self, n: usize, w: &mut GradientWorkspace, c: &mut usize) {
        w.prepare_row_gradient(*c, n);
        let _ = w.local_gradient_mut(*c, self.len());
        *c += 1;
    }
    fn set_scores(
        &self,
        s: &f64,
        row: usize,
        weight: f64,
        w: &mut GradientWorkspace,
        c: &mut usize,
    ) {
        w.set_row_gradient(*c, row, weight * *s);
        *c += 1;
    }
    fn backprop(&self, beta: &[f64], grad: &mut [f64], w: &mut GradientWorkspace, c: &mut usize) {
        backprop_leaf(self.x(), self.range(), beta, grad, w, *c);
        *c += 1;
    }
    fn visit_descriptors<V>(&self, p: &ParameterPath, c: &mut usize, v: &mut V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        v(
            *c,
            ParameterDescriptor::new(P::NAME, p.clone(), self.range()),
        );
        *c += 1;
    }
    fn visit_slices<V>(&self, c: &mut usize, v: &mut V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        v(*c, P::NAME, self.range());
        *c += 1;
    }
}

fn backprop_leaf<X: PredictorBlock>(
    predictor: &X,
    range: Range<usize>,
    beta: &[f64],
    grad: &mut [f64],
    workspace: &mut GradientWorkspace,
    cursor: usize,
) {
    let beta_block = &beta[range.clone()];
    let (scores, local) = workspace.row_gradient_and_local_gradient_mut(cursor, range.len());
    predictor.add_gradient(scores, beta_block, local);
    add_into(&mut grad[range], local);
}

macro_rules! impl_scalar_tuple_blocks {
    ($k:literal; $(($index:tt, $param:ident, $x:ident, $penalty:ident)),+ $(,)?) => {
        impl<$($param, $x, $penalty,)+> ShapeBlocks<ScalarTuple<($($param,)+), $k>>
            for ($(ParameterBlock<$param, $x, $penalty>,)+)
        where
            $($param: ParameterName, $x: PredictorBlock, $penalty: Penalty,)+
        {
            fn nrows(&self) -> Option<usize> { Some(self.0.x().nrows()) }

            fn try_len(&self) -> Result<usize, ModelError> {
                let mut len = 0;
                $(len = len.max(self.$index.try_range()?.end);)+
                Ok(len)
            }

            fn validate(&self, nobs: usize) -> Result<(), ModelError> {
                $(
                    self.$index.x().validate()?;
                    validate_block_rows(<$param as ParameterName>::NAME, self.$index.x().nrows(), nobs)?;
                    self.$index.penalty().validate_dim(self.$index.len())?;
                    self.$index.try_range()?;
                )+
                Ok(())
            }

            fn values_row(&self, beta: &[f64], row: usize) -> [f64; $k] {
                [$(self.$index.x().eta_row(row, &beta[self.$index.range()])),+]
            }

            fn penalty_value(&self, beta: &[f64]) -> f64 {
                0.0 $(+ self.$index.penalty().value(&beta[self.$index.range()]))+
            }

            fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
                $(
                    let range = self.$index.range();
                    self.$index.penalty().add_gradient(&beta[range.clone()], &mut grad[range]);
                )+
            }

            fn set_initial(&self, values: &[f64; $k], beta: &mut [f64]) {
                $(if values[$index].is_finite() { self.$index.x().set_constant_start(values[$index], &mut beta[self.$index.range()]); })+
            }

            fn leaf_count(&self) -> usize { $k }

            fn prepare_workspace(&self, nobs: usize, workspace: &mut GradientWorkspace, cursor: &mut usize) {
                $(
                    workspace.prepare_row_gradient(*cursor, nobs);
                    let _ = workspace.local_gradient_mut(*cursor, self.$index.len());
                    *cursor += 1;
                )+
            }

            fn set_scores(&self, scores: &[f64; $k], row: usize, weight: f64, workspace: &mut GradientWorkspace, cursor: &mut usize) {
                $(workspace.set_row_gradient(*cursor, row, weight * scores[$index]); *cursor += 1;)+
            }

            fn backprop(&self, beta: &[f64], grad: &mut [f64], workspace: &mut GradientWorkspace, cursor: &mut usize) {
                $(backprop_leaf(self.$index.x(), self.$index.range(), beta, grad, workspace, *cursor); *cursor += 1;)+
            }

            fn visit_descriptors<V>(&self, prefix: &ParameterPath, cursor: &mut usize, visit: &mut V)
            where V: FnMut(usize, ParameterDescriptor) {
                $(
                    visit(*cursor, ParameterDescriptor::new(<$param as ParameterName>::NAME, prefix.clone(), self.$index.range()));
                    *cursor += 1;
                )+
            }

            fn visit_slices<V>(&self, cursor: &mut usize, visit: &mut V)
            where V: FnMut(usize, &'static str, Range<usize>) {
                $(visit(*cursor, <$param as ParameterName>::NAME, self.$index.range()); *cursor += 1;)+
            }
        }
    };
}

impl_scalar_tuple_blocks!(1; (0, P1, X1, Pen1));
impl_scalar_tuple_blocks!(2; (0, P1, X1, Pen1), (1, P2, X2, Pen2));
impl_scalar_tuple_blocks!(3; (0, P1, X1, Pen1), (1, P2, X2, Pen2), (2, P3, X3, Pen3));
impl_scalar_tuple_blocks!(4; (0, P1, X1, Pen1), (1, P2, X2, Pen2), (2, P3, X3, Pen3), (3, P4, X4, Pen4));
impl_scalar_tuple_blocks!(5; (0, P1, X1, Pen1), (1, P2, X2, Pen2), (2, P3, X3, Pen3), (3, P4, X4, Pen4), (4, P5, X5, Pen5));
impl_scalar_tuple_blocks!(6; (0, P1, X1, Pen1), (1, P2, X2, Pen2), (2, P3, X3, Pen3), (3, P4, X4, Pen4), (4, P5, X5, Pen5), (5, P6, X6, Pen6));
impl_scalar_tuple_blocks!(7; (0, P1, X1, Pen1), (1, P2, X2, Pen2), (2, P3, X3, Pen3), (3, P4, X4, Pen4), (4, P5, X5, Pen5), (5, P6, X6, Pen6), (6, P7, X7, Pen7));
impl_scalar_tuple_blocks!(8; (0, P1, X1, Pen1), (1, P2, X2, Pen2), (2, P3, X3, Pen3), (3, P4, X4, Pen4), (4, P5, X5, Pen5), (5, P6, X6, Pen6), (6, P7, X7, Pen7), (7, P8, X8, Pen8));

// Structured implementations are written explicitly below; keeping their index
// mappings visible is clearer than hiding triangular arithmetic in a macro.

impl<P, const D: usize, X, Pen> ShapeBlocks<Vector<P, D>> for VectorParameterBlock<P, D, X, Pen>
where
    P: ParameterName,
    X: PredictorBlock,
    Pen: Penalty,
{
    fn nrows(&self) -> Option<usize> {
        self.components().first().map(PredictorBlock::nrows)
    }
    fn try_len(&self) -> Result<usize, ModelError> {
        Ok(self.try_range()?.end)
    }
    fn validate(&self, nobs: usize) -> Result<(), ModelError> {
        self.try_range()?;
        for x in self.components() {
            x.validate()?;
            validate_block_rows(P::NAME, x.nrows(), nobs)?;
        }
        self.penalty().validate_dim(self.len())
    }
    fn values_row(&self, beta: &[f64], row: usize) -> [f64; D] {
        std::array::from_fn(|i| {
            self.components()[i].eta_row(row, &beta[self.component_range(i).unwrap()])
        })
    }
    fn penalty_value(&self, beta: &[f64]) -> f64 {
        self.penalty().value(&beta[self.range()])
    }
    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        let r = self.range();
        self.penalty().add_gradient(&beta[r.clone()], &mut grad[r]);
    }
    fn set_initial(&self, v: &[f64; D], beta: &mut [f64]) {
        for (i, x) in self.components().iter().enumerate() {
            if v[i].is_finite() {
                x.set_constant_start(v[i], &mut beta[self.component_range(i).unwrap()]);
            }
        }
    }
    fn leaf_count(&self) -> usize {
        D
    }
    fn prepare_workspace(&self, n: usize, w: &mut GradientWorkspace, c: &mut usize) {
        for i in 0..D {
            w.prepare_row_gradient(*c, n);
            let _ = w.local_gradient_mut(*c, self.component_range(i).unwrap().len());
            *c += 1;
        }
    }
    fn set_scores(
        &self,
        s: &[f64; D],
        r: usize,
        weight: f64,
        w: &mut GradientWorkspace,
        c: &mut usize,
    ) {
        for value in s {
            w.set_row_gradient(*c, r, weight * value);
            *c += 1;
        }
    }
    fn backprop(&self, beta: &[f64], grad: &mut [f64], w: &mut GradientWorkspace, c: &mut usize) {
        for (i, x) in self.components().iter().enumerate() {
            backprop_leaf(x, self.component_range(i).unwrap(), beta, grad, w, *c);
            *c += 1;
        }
    }
    fn visit_descriptors<V>(&self, prefix: &ParameterPath, c: &mut usize, v: &mut V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        for i in 0..D {
            v(
                *c,
                ParameterDescriptor::new(
                    P::NAME,
                    prefix
                        .clone()
                        .with_axis(ParameterAxis::Vector { component: i }),
                    self.component_range(i).unwrap(),
                ),
            );
            *c += 1;
        }
    }
    fn visit_slices<V>(&self, c: &mut usize, v: &mut V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        v(*c, P::NAME, self.range());
        *c += 1;
    }
}

macro_rules! impl_triangular_shape_blocks {
    ($shape:ident, $block:ident, $strict:expr, $axis:ident) => {
        impl<P, const D: usize, X, Pen> ShapeBlocks<$shape<P, D>> for $block<P, D, X, Pen>
        where
            P: ParameterName,
            X: PredictorBlock,
            Pen: Penalty,
        {
            fn nrows(&self) -> Option<usize> {
                self.entries().first().map(PredictorBlock::nrows)
            }
            fn try_len(&self) -> Result<usize, ModelError> {
                Ok(self.try_range()?.end)
            }
            fn validate(&self, nobs: usize) -> Result<(), ModelError> {
                self.try_range()?;
                for x in self.entries() {
                    x.validate()?;
                    validate_block_rows(P::NAME, x.nrows(), nobs)?;
                }
                self.penalty().validate_dim(self.len())
            }
            fn values_row(&self, beta: &[f64], row: usize) -> [[f64; D]; D] {
                let mut out = [[0.0; D]; D];
                let mut i = 0;
                for r in 0..D {
                    let end = if $strict { r } else { r + 1 };
                    for col in 0..end {
                        out[r][col] = self.entries()[i]
                            .eta_row(row, &beta[self.entry_range(r, col).unwrap()]);
                        i += 1;
                    }
                }
                out
            }
            fn penalty_value(&self, beta: &[f64]) -> f64 {
                self.penalty().value(&beta[self.range()])
            }
            fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
                let r = self.range();
                self.penalty().add_gradient(&beta[r.clone()], &mut grad[r]);
            }
            fn set_initial(&self, v: &[[f64; D]; D], beta: &mut [f64]) {
                let mut i = 0;
                for r in 0..D {
                    let end = if $strict { r } else { r + 1 };
                    for col in 0..end {
                        if v[r][col].is_finite() {
                            self.entries()[i].set_constant_start(
                                v[r][col],
                                &mut beta[self.entry_range(r, col).unwrap()],
                            );
                        }
                        i += 1;
                    }
                }
            }
            fn leaf_count(&self) -> usize {
                self.entries().len()
            }
            fn prepare_workspace(&self, n: usize, w: &mut GradientWorkspace, c: &mut usize) {
                let mut i = 0;
                for r in 0..D {
                    let end = if $strict { r } else { r + 1 };
                    for col in 0..end {
                        w.prepare_row_gradient(*c, n);
                        let _ = w.local_gradient_mut(*c, self.entry_range(r, col).unwrap().len());
                        *c += 1;
                        i += 1;
                    }
                }
                debug_assert_eq!(i, self.entries().len());
            }
            fn set_scores(
                &self,
                s: &[[f64; D]; D],
                row: usize,
                weight: f64,
                w: &mut GradientWorkspace,
                c: &mut usize,
            ) {
                for r in 0..D {
                    let end = if $strict { r } else { r + 1 };
                    for col in 0..end {
                        w.set_row_gradient(*c, row, weight * s[r][col]);
                        *c += 1;
                    }
                }
            }
            fn backprop(
                &self,
                beta: &[f64],
                grad: &mut [f64],
                w: &mut GradientWorkspace,
                c: &mut usize,
            ) {
                let mut i = 0;
                for r in 0..D {
                    let end = if $strict { r } else { r + 1 };
                    for col in 0..end {
                        backprop_leaf(
                            &self.entries()[i],
                            self.entry_range(r, col).unwrap(),
                            beta,
                            grad,
                            w,
                            *c,
                        );
                        *c += 1;
                        i += 1;
                    }
                }
            }
            fn visit_descriptors<V>(&self, prefix: &ParameterPath, c: &mut usize, v: &mut V)
            where
                V: FnMut(usize, ParameterDescriptor),
            {
                for r in 0..D {
                    let end = if $strict { r } else { r + 1 };
                    for col in 0..end {
                        v(
                            *c,
                            ParameterDescriptor::new(
                                P::NAME,
                                prefix
                                    .clone()
                                    .with_axis(ParameterAxis::$axis { row: r, col }),
                                self.entry_range(r, col).unwrap(),
                            ),
                        );
                        *c += 1;
                    }
                }
            }
            fn visit_slices<V>(&self, c: &mut usize, v: &mut V)
            where
                V: FnMut(usize, &'static str, Range<usize>),
            {
                v(*c, P::NAME, self.range());
                *c += 1;
            }
        }
    };
}

impl_triangular_shape_blocks!(Lower, LowerTriangularParameterBlock, false, Lower);
impl_triangular_shape_blocks!(
    StrictLower,
    StrictLowerTriangularParameterBlock,
    true,
    StrictLower
);

impl<P, const C: usize, X, Pen> ShapeBlocks<Simplex<P, C>>
    for SimplexLogitParameterBlock<P, C, X, Pen>
where
    P: ParameterName,
    X: PredictorBlock,
    Pen: Penalty,
{
    fn nrows(&self) -> Option<usize> {
        self.logits().first().map(PredictorBlock::nrows)
    }
    fn try_len(&self) -> Result<usize, ModelError> {
        Ok(self.try_range()?.end)
    }
    fn validate(&self, nobs: usize) -> Result<(), ModelError> {
        self.try_range()?;
        for x in self.logits() {
            x.validate()?;
            validate_block_rows(P::NAME, x.nrows(), nobs)?;
        }
        self.penalty().validate_dim(self.len())
    }
    fn values_row(&self, beta: &[f64], row: usize) -> [f64; C] {
        let mut out = [0.0; C];
        for (i, x) in self.logits().iter().enumerate() {
            out[i] = x.eta_row(row, &beta[self.logit_range(i).unwrap()]);
        }
        out
    }
    fn penalty_value(&self, beta: &[f64]) -> f64 {
        self.penalty().value(&beta[self.range()])
    }
    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        let r = self.range();
        self.penalty().add_gradient(&beta[r.clone()], &mut grad[r]);
    }
    fn set_initial(&self, v: &[f64; C], beta: &mut [f64]) {
        for (i, x) in self.logits().iter().enumerate() {
            if v[i].is_finite() {
                x.set_constant_start(v[i], &mut beta[self.logit_range(i).unwrap()]);
            }
        }
    }
    fn leaf_count(&self) -> usize {
        self.logits().len()
    }
    fn prepare_workspace(&self, n: usize, w: &mut GradientWorkspace, c: &mut usize) {
        for i in 0..self.logits().len() {
            w.prepare_row_gradient(*c, n);
            let _ = w.local_gradient_mut(*c, self.logit_range(i).unwrap().len());
            *c += 1;
        }
    }
    fn set_scores(
        &self,
        s: &[f64; C],
        row: usize,
        weight: f64,
        w: &mut GradientWorkspace,
        c: &mut usize,
    ) {
        for value in s.iter().take(self.logits().len()) {
            w.set_row_gradient(*c, row, weight * value);
            *c += 1;
        }
    }
    fn backprop(&self, beta: &[f64], grad: &mut [f64], w: &mut GradientWorkspace, c: &mut usize) {
        for (i, x) in self.logits().iter().enumerate() {
            backprop_leaf(x, self.logit_range(i).unwrap(), beta, grad, w, *c);
            *c += 1;
        }
    }
    fn visit_descriptors<V>(&self, prefix: &ParameterPath, c: &mut usize, v: &mut V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        for i in 0..self.logits().len() {
            v(
                *c,
                ParameterDescriptor::new(
                    P::NAME,
                    prefix
                        .clone()
                        .with_axis(ParameterAxis::SimplexLogit { class: i }),
                    self.logit_range(i).unwrap(),
                ),
            );
            *c += 1;
        }
    }
    fn visit_slices<V>(&self, c: &mut usize, v: &mut V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        v(*c, P::NAME, self.range());
        *c += 1;
    }
}

impl<A, B, BA, BB> ShapeBlocks<Product<A, B>> for (BA, BB)
where
    A: ParameterShape,
    B: ParameterShape,
    BA: ShapeBlocks<A>,
    BB: ShapeBlocks<B>,
{
    fn nrows(&self) -> Option<usize> {
        self.0.nrows().or_else(|| self.1.nrows())
    }
    fn try_len(&self) -> Result<usize, ModelError> {
        Ok(self.0.try_len()?.max(self.1.try_len()?))
    }
    fn validate(&self, n: usize) -> Result<(), ModelError> {
        self.0.validate(n)?;
        self.1.validate(n)
    }
    fn values_row(&self, beta: &[f64], row: usize) -> (A::Values, B::Values) {
        (self.0.values_row(beta, row), self.1.values_row(beta, row))
    }
    fn penalty_value(&self, beta: &[f64]) -> f64 {
        self.0.penalty_value(beta) + self.1.penalty_value(beta)
    }
    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        self.0.add_penalty_gradient(beta, grad);
        self.1.add_penalty_gradient(beta, grad);
    }
    fn set_initial(&self, v: &(A::Values, B::Values), beta: &mut [f64]) {
        self.0.set_initial(&v.0, beta);
        self.1.set_initial(&v.1, beta);
    }
    fn leaf_count(&self) -> usize {
        self.0.leaf_count() + self.1.leaf_count()
    }
    fn prepare_workspace(&self, n: usize, w: &mut GradientWorkspace, c: &mut usize) {
        self.0.prepare_workspace(n, w, c);
        self.1.prepare_workspace(n, w, c);
    }
    fn set_scores(
        &self,
        s: &(A::Values, B::Values),
        r: usize,
        weight: f64,
        w: &mut GradientWorkspace,
        c: &mut usize,
    ) {
        self.0.set_scores(&s.0, r, weight, w, c);
        self.1.set_scores(&s.1, r, weight, w, c);
    }
    fn backprop(&self, beta: &[f64], grad: &mut [f64], w: &mut GradientWorkspace, c: &mut usize) {
        self.0.backprop(beta, grad, w, c);
        self.1.backprop(beta, grad, w, c);
    }
    fn visit_descriptors<V>(&self, p: &ParameterPath, c: &mut usize, v: &mut V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        self.0.visit_descriptors(p, c, v);
        self.1.visit_descriptors(p, c, v);
    }
    fn visit_slices<V>(&self, c: &mut usize, v: &mut V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        self.0.visit_slices(c, v);
        self.1.visit_slices(c, v);
    }
}

impl<A, B, C, BA, BB, BC> ShapeBlocks<Product<Product<A, B>, C>> for (BA, BB, BC)
where
    A: ParameterShape,
    B: ParameterShape,
    C: ParameterShape,
    BA: ShapeBlocks<A>,
    BB: ShapeBlocks<B>,
    BC: ShapeBlocks<C>,
{
    fn nrows(&self) -> Option<usize> {
        self.0
            .nrows()
            .or_else(|| self.1.nrows())
            .or_else(|| self.2.nrows())
    }
    fn try_len(&self) -> Result<usize, ModelError> {
        Ok(self
            .0
            .try_len()?
            .max(self.1.try_len()?)
            .max(self.2.try_len()?))
    }
    fn validate(&self, n: usize) -> Result<(), ModelError> {
        self.0.validate(n)?;
        self.1.validate(n)?;
        self.2.validate(n)
    }
    fn values_row(&self, beta: &[f64], row: usize) -> ((A::Values, B::Values), C::Values) {
        (
            (self.0.values_row(beta, row), self.1.values_row(beta, row)),
            self.2.values_row(beta, row),
        )
    }
    fn penalty_value(&self, beta: &[f64]) -> f64 {
        self.0.penalty_value(beta) + self.1.penalty_value(beta) + self.2.penalty_value(beta)
    }
    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        self.0.add_penalty_gradient(beta, grad);
        self.1.add_penalty_gradient(beta, grad);
        self.2.add_penalty_gradient(beta, grad);
    }
    fn set_initial(&self, v: &((A::Values, B::Values), C::Values), beta: &mut [f64]) {
        self.0.set_initial(&v.0.0, beta);
        self.1.set_initial(&v.0.1, beta);
        self.2.set_initial(&v.1, beta);
    }
    fn leaf_count(&self) -> usize {
        self.0.leaf_count() + self.1.leaf_count() + self.2.leaf_count()
    }
    fn prepare_workspace(&self, n: usize, w: &mut GradientWorkspace, c: &mut usize) {
        self.0.prepare_workspace(n, w, c);
        self.1.prepare_workspace(n, w, c);
        self.2.prepare_workspace(n, w, c);
    }
    fn set_scores(
        &self,
        s: &((A::Values, B::Values), C::Values),
        r: usize,
        weight: f64,
        w: &mut GradientWorkspace,
        c: &mut usize,
    ) {
        self.0.set_scores(&s.0.0, r, weight, w, c);
        self.1.set_scores(&s.0.1, r, weight, w, c);
        self.2.set_scores(&s.1, r, weight, w, c);
    }
    fn backprop(&self, beta: &[f64], grad: &mut [f64], w: &mut GradientWorkspace, c: &mut usize) {
        self.0.backprop(beta, grad, w, c);
        self.1.backprop(beta, grad, w, c);
        self.2.backprop(beta, grad, w, c);
    }
    fn visit_descriptors<V>(&self, p: &ParameterPath, c: &mut usize, v: &mut V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        self.0.visit_descriptors(p, c, v);
        self.1.visit_descriptors(p, c, v);
        self.2.visit_descriptors(p, c, v);
    }
    fn visit_slices<V>(&self, c: &mut usize, v: &mut V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        self.0.visit_slices(c, v);
        self.1.visit_slices(c, v);
        self.2.visit_slices(c, v);
    }
}

impl<A, B, const C: usize> ShapeBlocks<Repeated<A, C>> for [B; C]
where
    A: ParameterShape,
    B: ShapeBlocks<A>,
{
    fn nrows(&self) -> Option<usize> {
        self.iter().find_map(ShapeBlocks::nrows)
    }
    fn try_len(&self) -> Result<usize, ModelError> {
        self.iter().try_fold(0, |len, b| Ok(len.max(b.try_len()?)))
    }
    fn validate(&self, n: usize) -> Result<(), ModelError> {
        for b in self {
            b.validate(n)?;
        }
        Ok(())
    }
    fn values_row(&self, beta: &[f64], row: usize) -> [A::Values; C] {
        std::array::from_fn(|i| self[i].values_row(beta, row))
    }
    fn penalty_value(&self, beta: &[f64]) -> f64 {
        self.iter().map(|b| b.penalty_value(beta)).sum()
    }
    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        for b in self {
            b.add_penalty_gradient(beta, grad);
        }
    }
    fn set_initial(&self, v: &[A::Values; C], beta: &mut [f64]) {
        for (i, b) in self.iter().enumerate() {
            b.set_initial(&v[i], beta);
        }
    }
    fn leaf_count(&self) -> usize {
        self.iter().map(ShapeBlocks::leaf_count).sum()
    }
    fn prepare_workspace(&self, n: usize, w: &mut GradientWorkspace, c: &mut usize) {
        for b in self {
            b.prepare_workspace(n, w, c);
        }
    }
    fn set_scores(
        &self,
        s: &[A::Values; C],
        r: usize,
        weight: f64,
        w: &mut GradientWorkspace,
        c: &mut usize,
    ) {
        for (i, b) in self.iter().enumerate() {
            b.set_scores(&s[i], r, weight, w, c);
        }
    }
    fn backprop(&self, beta: &[f64], grad: &mut [f64], w: &mut GradientWorkspace, c: &mut usize) {
        for b in self {
            b.backprop(beta, grad, w, c);
        }
    }
    fn visit_descriptors<V>(&self, p: &ParameterPath, c: &mut usize, v: &mut V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        for (i, b) in self.iter().enumerate() {
            b.visit_descriptors(
                &p.clone().with_axis(ParameterAxis::Component { index: i }),
                c,
                v,
            );
        }
    }
    fn visit_slices<V>(&self, c: &mut usize, v: &mut V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        for b in self {
            b.visit_slices(c, v);
        }
    }
}

impl<A, B, const C: usize> ShapeBlocks<Broadcast<A, C>> for B
where
    A: ParameterShape,
    B: ShapeBlocks<A>,
{
    fn nrows(&self) -> Option<usize> {
        <B as ShapeBlocks<A>>::nrows(self)
    }

    fn try_len(&self) -> Result<usize, ModelError> {
        <B as ShapeBlocks<A>>::try_len(self)
    }

    fn validate(&self, n: usize) -> Result<(), ModelError> {
        <B as ShapeBlocks<A>>::validate(self, n)
    }

    fn values_row(&self, beta: &[f64], row: usize) -> [A::Values; C] {
        let value = <B as ShapeBlocks<A>>::values_row(self, beta, row);
        std::array::from_fn(|_| value.clone())
    }

    fn penalty_value(&self, beta: &[f64]) -> f64 {
        <B as ShapeBlocks<A>>::penalty_value(self, beta)
    }

    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        <B as ShapeBlocks<A>>::add_penalty_gradient(self, beta, grad);
    }

    fn set_initial(&self, values: &[A::Values; C], beta: &mut [f64]) {
        if let Some(first) = values.first() {
            <B as ShapeBlocks<A>>::set_initial(self, first, beta);
        }
    }

    fn leaf_count(&self) -> usize {
        <B as ShapeBlocks<A>>::leaf_count(self)
    }

    fn prepare_workspace(&self, n: usize, w: &mut GradientWorkspace, c: &mut usize) {
        <B as ShapeBlocks<A>>::prepare_workspace(self, n, w, c);
    }

    fn set_scores(
        &self,
        scores: &[A::Values; C],
        row: usize,
        weight: f64,
        workspace: &mut GradientWorkspace,
        cursor: &mut usize,
    ) {
        let mut total = A::zeros();
        for score in scores {
            A::add_assign(&mut total, score);
        }
        <B as ShapeBlocks<A>>::set_scores(self, &total, row, weight, workspace, cursor);
    }

    fn backprop(
        &self,
        beta: &[f64],
        grad: &mut [f64],
        workspace: &mut GradientWorkspace,
        cursor: &mut usize,
    ) {
        <B as ShapeBlocks<A>>::backprop(self, beta, grad, workspace, cursor);
    }

    fn visit_descriptors<V>(&self, prefix: &ParameterPath, cursor: &mut usize, visit: &mut V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        <B as ShapeBlocks<A>>::visit_descriptors(self, prefix, cursor, visit);
    }

    fn visit_slices<V>(&self, cursor: &mut usize, visit: &mut V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        <B as ShapeBlocks<A>>::visit_slices(self, cursor, visit);
    }
}
