/// Defines the repeated parameter plumbing for a family with two independently linked positive parameters.
///
/// Each invocation generates two distinct named carrier types. The eta carrier stores the predictors
///
/// $$
/// \boldsymbol{\eta} = (\eta_1, \eta_2),
/// $$
///
/// while the theta carrier stores the corresponding positive natural-scale parameters
///
/// $$
/// \boldsymbol{\theta} = (\theta_1, \theta_2).
/// $$
///
/// The generated [`gamlss_core::ParameterParts`] implementation maps the named eta fields to the fixed model order: `from_array([eta_1, eta_2])` constructs the carrier, while `part(0)` and `part(1)` return $\eta_1$ and $\eta_2$, respectively.
///
/// The generated `theta_from_links` method automates the predictor-to-parameter transformation for inverse links $g_1^{-1}$ and $g_2^{-1}$:
///
/// $$
/// \begin{aligned}
/// \theta_1 &= g_1^{-1}(\eta_1), \\\\
/// \theta_2 &= g_2^{-1}(\eta_2).
/// \end{aligned}
/// $$
///
/// Let $L$ denote the family-specific negative log-likelihood for one observation. After family code computes its natural-scale derivatives
///
/// $$
/// \nabla_{\boldsymbol{\theta}} L
/// = \left(
///     \frac{\partial L}{\partial \theta_1},
///     \frac{\partial L}{\partial \theta_2}
///   \right),
/// $$
///
/// the generated `chain_gradient` method converts them into the predictor-scale gradient expected by the optimizer:
///
/// $$
/// \nabla_{\boldsymbol{\eta}} L
/// = \left(
///     \frac{\partial L}{\partial \theta_1}
///     \frac{d g_1^{-1}(\eta_1)}{d\eta_1},
///     \frac{\partial L}{\partial \theta_2}
///     \frac{d g_2^{-1}(\eta_2)}{d\eta_2}
///   \right).
/// $$
///
/// For example, with log links, $\theta_i = \exp(\eta_i)$ and the generated gradient component is $\partial L / \partial \eta_i = (\partial L / \partial \theta_i)\theta_i$. Family-specific likelihoods, canonical reparameterizations, and the derivatives $\partial L / \partial \theta_i$ remain outside the macro.
#[allow(clippy::doc_markdown)]
macro_rules! define_two_positive_parameter_blocks {
    (
        eta:
        $(#[$eta_meta:meta])*
        $eta:ident {
            $(#[$eta_first_meta:meta])*
            $first:ident,
            $(#[$eta_second_meta:meta])*
            $second:ident $(,)?
        }
        theta:
        $(#[$theta_meta:meta])*
        $theta:ident {
            $(#[$theta_first_meta:meta])*
            $theta_first:ident,
            $(#[$theta_second_meta:meta])*
            $theta_second:ident $(,)?
        }
    ) => {
        $(#[$eta_meta])*
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $eta {
            $(#[$eta_first_meta])*
            pub $first: f64,
            $(#[$eta_second_meta])*
            pub $second: f64,
        }

        impl gamlss_core::ParameterParts<2> for $eta {
            #[inline]
            fn from_array(values: [f64; 2]) -> Self {
                Self {
                    $first: values[0],
                    $second: values[1],
                }
            }

            #[inline]
            fn part(&self, index: usize) -> f64 {
                match index {
                    0 => self.$first,
                    1 => self.$second,
                    _ => unreachable!(concat!(
                        stringify!($eta),
                        " only has indices 0 and 1"
                    )),
                }
            }
        }

        $(#[$theta_meta])*
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $theta {
            $(#[$theta_first_meta])*
            pub $theta_first: f64,
            $(#[$theta_second_meta])*
            pub $theta_second: f64,
        }

        impl $eta {
            /// Maps the two predictors independently to their positive natural-scale parameters.
            #[inline]
            fn theta_from_links<FirstLink, SecondLink>(self) -> $theta
            where
                FirstLink: gamlss_core::PositiveLink<f64>,
                SecondLink: gamlss_core::PositiveLink<f64>,
            {
                $theta {
                    $theta_first: FirstLink::inverse(self.$first),
                    $theta_second: SecondLink::inverse(self.$second),
                }
            }

            /// Pulls natural-scale NLL derivatives back through the two independent inverse links.
            #[inline]
            #[allow(dead_code)]
            fn chain_gradient<FirstLink, SecondLink>(
                self,
                d_first: f64,
                d_second: f64,
            ) -> Self
            where
                FirstLink: gamlss_core::PositiveLink<f64>,
                SecondLink: gamlss_core::PositiveLink<f64>,
            {
                Self {
                    $first: d_first * FirstLink::derivative_inverse(self.$first),
                    $second: d_second * SecondLink::derivative_inverse(self.$second),
                }
            }
        }
    };
}
