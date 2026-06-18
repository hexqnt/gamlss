use gamlss_core::{
    Gamlss, Identity, Log, Logit, Mu, ParameterBlock, ParameterBlocks, Precision, Rate, Scale,
    Shape, Sigma,
};
use gamlss_family::{
    DefaultBeta, DefaultGamma, DefaultInverseGaussian, DefaultLogNormal, DefaultNormal,
    DefaultWeibull,
};

use crate::{
    BuiltModel, Col, DataView, FormulaError, FormulaPenalty, ModelSchema, NumericResponse,
    PredictionDesign, ResponseSchema, TermExpr,
    compile::{ResponseDomain, fit_terms, predictor_from_fitted_terms, required_response},
    penalty::prediction_penalty,
    predictor::FormulaPredictorBlock,
    schema::terms_for,
};

/// Formula block type used by supported family specs.
pub type FormulaBlock<P, L> = ParameterBlock<P, L, FormulaPredictorBlock, FormulaPenalty>;

/// Blocks for a normal formula model.
pub type NormalBlocks = (FormulaBlock<Mu, Identity>, FormulaBlock<Sigma, Log>);
/// Blocks for a gamma formula model.
pub type GammaBlocks = (FormulaBlock<Shape, Log>, FormulaBlock<Rate, Log>);
/// Blocks for a log-normal formula model.
pub type LogNormalBlocks = (FormulaBlock<Mu, Identity>, FormulaBlock<Sigma, Log>);
/// Blocks for a Weibull formula model.
pub type WeibullBlocks = (FormulaBlock<Shape, Log>, FormulaBlock<Scale, Log>);
/// Blocks for an inverse Gaussian formula model.
pub type InverseGaussianBlocks = (FormulaBlock<Mu, Log>, FormulaBlock<Shape, Log>);
/// Blocks for a beta formula model.
pub type BetaBlocks = (FormulaBlock<Mu, Logit>, FormulaBlock<Precision, Log>);

/// Compiled normal formula model.
pub type CompiledNormal<'a> = Gamlss<DefaultNormal, NormalBlocks, NumericResponse<'a>>;
/// Compiled gamma formula model.
pub type CompiledGamma<'a> = Gamlss<DefaultGamma, GammaBlocks, NumericResponse<'a>>;
/// Compiled log-normal formula model.
pub type CompiledLogNormal<'a> = Gamlss<DefaultLogNormal, LogNormalBlocks, NumericResponse<'a>>;
/// Compiled Weibull formula model.
pub type CompiledWeibull<'a> = Gamlss<DefaultWeibull, WeibullBlocks, NumericResponse<'a>>;
/// Compiled inverse Gaussian formula model.
pub type CompiledInverseGaussian<'a> =
    Gamlss<DefaultInverseGaussian, InverseGaussianBlocks, NumericResponse<'a>>;
/// Compiled beta formula model.
pub type CompiledBeta<'a> = Gamlss<DefaultBeta, BetaBlocks, NumericResponse<'a>>;

/// Built normal formula model.
pub type BuiltNormal<'a> = BuiltModel<CompiledNormal<'a>>;
/// Built gamma formula model.
pub type BuiltGamma<'a> = BuiltModel<CompiledGamma<'a>>;
/// Built log-normal formula model.
pub type BuiltLogNormal<'a> = BuiltModel<CompiledLogNormal<'a>>;
/// Built Weibull formula model.
pub type BuiltWeibull<'a> = BuiltModel<CompiledWeibull<'a>>;
/// Built inverse Gaussian formula model.
pub type BuiltInverseGaussian<'a> = BuiltModel<CompiledInverseGaussian<'a>>;
/// Built beta formula model.
pub type BuiltBeta<'a> = BuiltModel<CompiledBeta<'a>>;

macro_rules! define_spec {
    (
        $(#[$meta:meta])*
        $spec:ident, $built:ident, $compiled:ident, $blocks:ident, $family:ty;
        family_name = $family_name:literal, domain = $domain:expr;
        first = $first_field:ident, $first_method:ident, $first_name:literal, $first_param:ty, $first_link:ty;
        second = $second_field:ident, $second_method:ident, $second_name:literal, $second_param:ty, $second_link:ty
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $spec {
            response: Option<Col<f64>>,
            weights: Option<Col<f64>>,
            $first_field: Option<TermExpr>,
            $second_field: Option<TermExpr>,
            duplicate_parameter: Option<&'static str>,
        }

        impl $spec {
            /// Creates an empty model specification.
            #[must_use]
            pub fn new() -> Self {
                Self {
                    response: None,
                    weights: None,
                    $first_field: None,
                    $second_field: None,
                    duplicate_parameter: None,
                }
            }

            /// Sets the response column.
            #[must_use]
            pub fn response(mut self, response: Col<f64>) -> Self {
                self.response = Some(response);
                self
            }

            /// Sets optional observation weights.
            #[must_use]
            pub fn weights(mut self, weights: Col<f64>) -> Self {
                self.weights = Some(weights);
                self
            }

            /// Sets terms for the first distribution parameter.
            #[must_use]
            pub fn $first_method(mut self, terms: impl Into<TermExpr>) -> Self {
                if self.$first_field.is_some() {
                    self.duplicate_parameter.get_or_insert($first_name);
                } else {
                    self.$first_field = Some(terms.into());
                }
                self
            }

            /// Sets terms for the second distribution parameter.
            #[must_use]
            pub fn $second_method(mut self, terms: impl Into<TermExpr>) -> Self {
                if self.$second_field.is_some() {
                    self.duplicate_parameter.get_or_insert($second_name);
                } else {
                    self.$second_field = Some(terms.into());
                }
                self
            }

            /// Builds a typed core model plus formula metadata.
            pub fn build<'a, D>(&self, data: &'a D) -> Result<$built<'a>, FormulaError>
            where
                D: DataView + ?Sized,
            {
                if data.nrows() == 0 {
                    return Err(FormulaError::EmptyData);
                }
                if let Some(parameter) = self.duplicate_parameter {
                    return Err(FormulaError::DuplicateParameter(parameter));
                }

                let (response_col, response) = required_response(
                    $family_name,
                    $domain,
                    data,
                    &self.response,
                    &self.weights,
                )?;
                let first_build = fit_terms($first_name, self.$first_field.clone(), data)?;
                let second_build = fit_terms($second_name, self.$second_field.clone(), data)?;

                let first = ParameterBlock::<$first_param, $first_link, _, _>::from_predictor(
                    first_build.predictor,
                    first_build.penalty,
                    0,
                );
                let second = ParameterBlock::<$second_param, $second_link, _, _>::from_predictor(
                    second_build.predictor,
                    second_build.penalty,
                    0,
                );
                let blocks: $blocks = ParameterBlocks::new((first, second));
                let model: $compiled<'a> =
                    Gamlss::try_new_with_observations(<$family>::new(), blocks, response)?;
                let layout = model.parameter_layout();
                let schema = ModelSchema {
                    response: ResponseSchema {
                        col: response_col,
                        weights: self.weights.clone(),
                        nrows: data.nrows(),
                    },
                    parameters: vec![first_build.terms, second_build.terms],
                };

                Ok(BuiltModel::new(model, schema, layout))
            }
        }

        impl Default for $spec {
            fn default() -> Self {
                Self::new()
            }
        }

        impl<'a> BuiltModel<$compiled<'a>> {
            /// Rebuilds compatible typed prediction blocks from fitted term metadata.
            pub fn prediction_blocks<D>(&self, data: &D) -> Result<$blocks, FormulaError>
            where
                D: DataView + ?Sized,
            {
                let first_terms = terms_for(self.schema(), $first_name);
                let second_terms = terms_for(self.schema(), $second_name);
                let first_x = predictor_from_fitted_terms($first_name, first_terms, data)?;
                let second_x = predictor_from_fitted_terms($second_name, second_terms, data)?;

                let first = ParameterBlock::<$first_param, $first_link, _, _>::from_predictor(
                    first_x,
                    prediction_penalty(first_terms),
                    0,
                );
                let second = ParameterBlock::<$second_param, $second_link, _, _>::from_predictor(
                    second_x,
                    prediction_penalty(second_terms),
                    0,
                );

                Ok(ParameterBlocks::new((first, second)))
            }

            /// Builds reusable prediction design from fitted term metadata.
            pub fn prediction_design<D>(
                &self,
                data: &D,
            ) -> Result<PredictionDesign<$blocks>, FormulaError>
            where
                D: DataView + ?Sized,
            {
                Ok(PredictionDesign::new(self.prediction_blocks(data)?))
            }

            /// Predicts natural-scale distribution parameters with reusable prediction design.
            pub fn predict_theta_with_design(
                &self,
                theta: &[f64],
                design: &PredictionDesign<$blocks>,
            ) -> Result<Vec<<$family as gamlss_core::Family>::Theta>, FormulaError>
            {
                Ok(self.model().predict_theta_with_blocks(theta, design.blocks())?)
            }

            /// Predicts natural-scale distribution parameters for new rows.
            pub fn predict_theta<D>(
                &self,
                theta: &[f64],
                data: &D,
            ) -> Result<Vec<<$family as gamlss_core::Family>::Theta>, FormulaError>
            where
                D: DataView + ?Sized,
            {
                let design = self.prediction_design(data)?;
                self.predict_theta_with_design(theta, &design)
            }
        }
    };
}

define_spec!(
    /// Typed normal model specification.
    NormalSpec, BuiltNormal, CompiledNormal, NormalBlocks, DefaultNormal;
    family_name = "normal", domain = ResponseDomain::Finite;
    first = mu_terms, mu, "mu", Mu, Identity;
    second = sigma_terms, sigma, "sigma", Sigma, Log
);

define_spec!(
    /// Typed gamma model specification.
    GammaSpec, BuiltGamma, CompiledGamma, GammaBlocks, DefaultGamma;
    family_name = "gamma", domain = ResponseDomain::Positive;
    first = shape_terms, shape, "shape", Shape, Log;
    second = rate_terms, rate, "rate", Rate, Log
);

define_spec!(
    /// Typed log-normal model specification.
    LogNormalSpec, BuiltLogNormal, CompiledLogNormal, LogNormalBlocks, DefaultLogNormal;
    family_name = "log-normal", domain = ResponseDomain::Positive;
    first = mu_terms, mu, "mu", Mu, Identity;
    second = sigma_terms, sigma, "sigma", Sigma, Log
);

define_spec!(
    /// Typed Weibull model specification.
    WeibullSpec, BuiltWeibull, CompiledWeibull, WeibullBlocks, DefaultWeibull;
    family_name = "weibull", domain = ResponseDomain::Positive;
    first = shape_terms, shape, "shape", Shape, Log;
    second = scale_terms, scale, "scale", Scale, Log
);

define_spec!(
    /// Typed inverse Gaussian model specification.
    InverseGaussianSpec, BuiltInverseGaussian, CompiledInverseGaussian, InverseGaussianBlocks, DefaultInverseGaussian;
    family_name = "inverse Gaussian", domain = ResponseDomain::Positive;
    first = mu_terms, mu, "mu", Mu, Log;
    second = shape_terms, shape, "shape", Shape, Log
);

define_spec!(
    /// Typed beta model specification.
    BetaSpec, BuiltBeta, CompiledBeta, BetaBlocks, DefaultBeta;
    family_name = "beta", domain = ResponseDomain::Unit;
    first = mu_terms, mu, "mu", Mu, Logit;
    second = precision_terms, precision, "precision", Precision, Log
);

/// Entry point namespace for typed model specifications.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelSpec;

impl ModelSpec {
    /// Creates a normal model spec.
    #[must_use]
    pub fn normal() -> NormalSpec {
        NormalSpec::new()
    }

    /// Creates a gamma model spec.
    #[must_use]
    pub fn gamma() -> GammaSpec {
        GammaSpec::new()
    }

    /// Creates a log-normal model spec.
    #[must_use]
    pub fn log_normal() -> LogNormalSpec {
        LogNormalSpec::new()
    }

    /// Creates a Weibull model spec.
    #[must_use]
    pub fn weibull() -> WeibullSpec {
        WeibullSpec::new()
    }

    /// Creates an inverse Gaussian model spec.
    #[must_use]
    pub fn inverse_gaussian() -> InverseGaussianSpec {
        InverseGaussianSpec::new()
    }

    /// Creates a beta model spec.
    #[must_use]
    pub fn beta() -> BetaSpec {
        BetaSpec::new()
    }
}

/// Creates a normal model spec.
#[must_use]
pub fn normal() -> NormalSpec {
    ModelSpec::normal()
}

/// Creates a gamma model spec.
#[must_use]
pub fn gamma() -> GammaSpec {
    ModelSpec::gamma()
}

/// Creates a log-normal model spec.
#[must_use]
pub fn log_normal() -> LogNormalSpec {
    ModelSpec::log_normal()
}

/// Creates a Weibull model spec.
#[must_use]
pub fn weibull() -> WeibullSpec {
    ModelSpec::weibull()
}

/// Creates an inverse Gaussian model spec.
#[must_use]
pub fn inverse_gaussian() -> InverseGaussianSpec {
    ModelSpec::inverse_gaussian()
}

/// Creates a beta model spec.
#[must_use]
pub fn beta() -> BetaSpec {
    ModelSpec::beta()
}
