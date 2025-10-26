"""Hypothesis testing framework for portfolio performance claims validation.

Implements comprehensive hypothesis testing procedures for validating performance
improvement claims, including statistical power analysis, effect size calculations,
and Bayesian hypothesis testing for robustness validation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm


class EffectSizeInterpretation(Enum):
    """Cohen's conventions for effect size interpretation."""

    NEGLIGIBLE = "negligible"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class HypothesisTestResult:
    """Container for hypothesis test results."""

    test_statistic: float
    p_value: float
    effect_size: float
    power: float
    is_significant: bool
    confidence_interval: tuple[float, float]
    interpretation: str
    additional_info: dict


class PerformanceHypothesisTestingFramework:
    """Framework for hypothesis testing of portfolio performance claims."""

    def __init__(self, alpha: float = 0.05):
        """Initialize hypothesis testing framework.

        Args:
            alpha: Significance level (Type I error rate)
        """
        self.alpha = alpha

    def sharpe_ratio_improvement_test(
        self,
        returns_treatment: Union[pd.Series, np.ndarray],
        returns_control: Union[pd.Series, np.ndarray],
        min_improvement: float = 0.2,
        alternative: str = "greater",
    ) -> HypothesisTestResult:
        """Statistical power analysis for ≥0.2 Sharpe ratio improvement detection.

        Args:
            returns_treatment: Return series for treatment (ML approach)
            returns_control: Return series for control (baseline)
            min_improvement: Minimum improvement threshold to detect
            alternative: Alternative hypothesis ('greater', 'two-sided', 'less')

        Returns:
            HypothesisTestResult containing test results
        """
        ret_treatment = np.asarray(returns_treatment)
        ret_control = np.asarray(returns_control)

        if len(ret_treatment) != len(ret_control):
            raise ValueError("Return series must have equal length")

        n = len(ret_treatment)

        # Calculate Sharpe ratios
        sharpe_treatment = self._calculate_sharpe_ratio(ret_treatment)
        sharpe_control = self._calculate_sharpe_ratio(ret_control)
        observed_improvement = sharpe_treatment - sharpe_control

        # Jobson-Korkie test for Sharpe ratio difference
        test_stat, p_value = self._jobson_korkie_test(ret_treatment, ret_control)

        # Effect size calculation (Cohen's d adapted for Sharpe ratios)
        effect_size = self._calculate_sharpe_effect_size(ret_treatment, ret_control)

        # Statistical power analysis
        power = self._calculate_power_sharpe_test(
            ret_treatment, ret_control, min_improvement, self.alpha
        )

        # Confidence interval for Sharpe ratio difference
        ci_lower, ci_upper = self._sharpe_difference_confidence_interval(
            ret_treatment, ret_control, confidence_level=1 - self.alpha
        )

        # Determine significance based on alternative hypothesis
        if alternative == "greater":
            is_significant = (observed_improvement >= min_improvement) and (p_value < self.alpha)
            interpretation = f"Treatment Sharpe ratio is {'significantly' if is_significant else 'not significantly'} higher than control by at least {min_improvement}"
        elif alternative == "two-sided":
            is_significant = (abs(observed_improvement) >= min_improvement) and (
                p_value < self.alpha
            )
            interpretation = f"Treatment Sharpe ratio {'significantly differs' if is_significant else 'does not significantly differ'} from control by at least {min_improvement}"
        elif alternative == "less":
            is_significant = (observed_improvement <= -min_improvement) and (p_value < self.alpha)
            interpretation = f"Treatment Sharpe ratio is {'significantly' if is_significant else 'not significantly'} lower than control by at least {min_improvement}"

        return HypothesisTestResult(
            test_statistic=test_stat,
            p_value=p_value,
            effect_size=effect_size,
            power=power,
            is_significant=is_significant,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            additional_info={
                "sharpe_treatment": sharpe_treatment,
                "sharpe_control": sharpe_control,
                "observed_improvement": observed_improvement,
                "min_improvement_threshold": min_improvement,
                "sample_size": n,
                "alternative": alternative,
                "test_method": "Jobson-Korkie",
            },
        )

    def effect_size_calculation_cohen_financial(
        self,
        returns_treatment: Union[pd.Series, np.ndarray],
        returns_control: Union[pd.Series, np.ndarray],
        metric: str = "sharpe_ratio",
    ) -> dict[str, Union[float, str]]:
        """Effect size calculations using Cohen's conventions adapted for financial metrics.

        Args:
            returns_treatment: Treatment group returns
            returns_control: Control group returns
            metric: Performance metric to analyze

        Returns:
            Dictionary containing effect size analysis
        """
        ret_treatment = np.asarray(returns_treatment)
        ret_control = np.asarray(returns_control)

        if metric == "sharpe_ratio":
            effect_size = self._calculate_sharpe_effect_size(ret_treatment, ret_control)
            interpretation = self._interpret_sharpe_effect_size(effect_size)

        elif metric == "return":
            effect_size = self._calculate_return_effect_size(ret_treatment, ret_control)
            interpretation = self._interpret_return_effect_size(effect_size)

        elif metric == "volatility":
            effect_size = self._calculate_volatility_effect_size(ret_treatment, ret_control)
            interpretation = self._interpret_volatility_effect_size(effect_size)

        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Calculate confidence interval for effect size
        ci_lower, ci_upper = self._effect_size_confidence_interval(
            ret_treatment, ret_control, metric
        )

        return {
            "effect_size": effect_size,
            "interpretation": interpretation,
            "confidence_interval": (ci_lower, ci_upper),
            "metric": metric,
            "cohens_category": self._cohens_category(effect_size),
            "practical_significance": self._assess_practical_significance(effect_size, metric),
        }

    def sample_size_validation_adequate_power(
        self,
        returns_treatment: Union[pd.Series, np.ndarray],
        returns_control: Union[pd.Series, np.ndarray],
        min_effect_size: float = 0.2,
        desired_power: float = 0.8,
    ) -> dict[str, Union[int, float, bool]]:
        """Sample size validation ensuring adequate power for conclusions.

        Args:
            returns_treatment: Treatment group returns
            returns_control: Control group returns
            min_effect_size: Minimum effect size to detect
            desired_power: Desired statistical power

        Returns:
            Dictionary containing sample size analysis
        """
        ret_treatment = np.asarray(returns_treatment)
        ret_control = np.asarray(returns_control)

        current_n = len(ret_treatment)

        # Current power analysis
        current_power = self._calculate_power_sharpe_test(
            ret_treatment, ret_control, min_effect_size, self.alpha
        )

        # Required sample size calculation
        required_n = self._calculate_required_sample_size(
            ret_treatment, ret_control, min_effect_size, desired_power, self.alpha
        )

        # Post-hoc analysis
        detectable_effect_size = self._calculate_detectable_effect_size(
            ret_treatment, ret_control, desired_power, self.alpha
        )

        # Validation results
        is_adequate = current_power >= desired_power
        power_deficit = max(0, desired_power - current_power)
        sample_size_deficit = max(0, required_n - current_n)

        return {
            "current_sample_size": current_n,
            "current_power": current_power,
            "required_sample_size": required_n,
            "is_adequate_power": is_adequate,
            "power_deficit": power_deficit,
            "sample_size_deficit": sample_size_deficit,
            "detectable_effect_size": detectable_effect_size,
            "min_effect_size_target": min_effect_size,
            "desired_power": desired_power,
            "recommendations": self._generate_power_recommendations(
                is_adequate, power_deficit, sample_size_deficit
            ),
        }

    def bayesian_hypothesis_testing_robustness(
        self,
        returns_treatment: Union[pd.Series, np.ndarray],
        returns_control: Union[pd.Series, np.ndarray],
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        n_samples: int = 10000,
    ) -> dict[str, Union[float, np.ndarray, str]]:
        """Bayesian hypothesis testing framework for robustness validation.

        Args:
            returns_treatment: Treatment group returns
            returns_control: Control group returns
            prior_mean: Prior mean for Sharpe ratio difference
            prior_variance: Prior variance for Sharpe ratio difference
            n_samples: Number of MCMC samples

        Returns:
            Dictionary containing Bayesian analysis results
        """
        ret_treatment = np.asarray(returns_treatment)
        ret_control = np.asarray(returns_control)

        # Calculate observed statistics
        sharpe_treatment = self._calculate_sharpe_ratio(ret_treatment)
        sharpe_control = self._calculate_sharpe_ratio(ret_control)
        observed_diff = sharpe_treatment - sharpe_control

        # Bayesian updating using conjugate priors (normal-normal model)
        # Likelihood variance (using Jobson-Korkie variance estimate)
        likelihood_variance = self._calculate_jobson_korkie_variance(ret_treatment, ret_control)

        # Posterior parameters
        precision_prior = 1 / prior_variance
        precision_likelihood = 1 / likelihood_variance if likelihood_variance > 0 else 1e6

        posterior_precision = precision_prior + precision_likelihood
        posterior_variance = 1 / posterior_precision
        posterior_mean = (
            precision_prior * prior_mean + precision_likelihood * observed_diff
        ) / posterior_precision

        # Generate posterior samples
        posterior_samples = np.random.normal(posterior_mean, np.sqrt(posterior_variance), n_samples)

        # Bayesian hypothesis testing
        # H0: difference <= 0 vs H1: difference > 0
        prob_positive = np.mean(posterior_samples > 0)

        # H0: |difference| <= 0.2 vs H1: |difference| > 0.2
        prob_meaningful_improvement = np.mean(posterior_samples > 0.2)

        # Bayes Factor calculation (using Savage-Dickey density ratio)
        # BF01 = p(diff=0|data) / p(diff=0|prior)
        posterior_density_at_zero = stats.norm.pdf(0, posterior_mean, np.sqrt(posterior_variance))
        prior_density_at_zero = stats.norm.pdf(0, prior_mean, np.sqrt(prior_variance))

        bayes_factor_01 = posterior_density_at_zero / prior_density_at_zero
        bayes_factor_10 = 1 / bayes_factor_01

        # Credible intervals
        credible_intervals = {
            0.95: np.percentile(posterior_samples, [2.5, 97.5]),
            0.90: np.percentile(posterior_samples, [5, 95]),
            0.80: np.percentile(posterior_samples, [10, 90]),
        }

        # Interpretation
        interpretation = self._interpret_bayesian_results(
            prob_positive, prob_meaningful_improvement, bayes_factor_10
        )

        return {
            "posterior_mean": posterior_mean,
            "posterior_variance": posterior_variance,
            "posterior_samples": posterior_samples,
            "prob_positive_effect": prob_positive,
            "prob_meaningful_improvement": prob_meaningful_improvement,
            "bayes_factor_10": bayes_factor_10,
            "credible_intervals": credible_intervals,
            "interpretation": interpretation,
            "prior_specification": {"prior_mean": prior_mean, "prior_variance": prior_variance},
            "observed_difference": observed_diff,
        }

    # Helper methods
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns, ddof=1) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns, ddof=1)

    def _jobson_korkie_test(
        self, returns_a: np.ndarray, returns_b: np.ndarray
    ) -> tuple[float, float]:
        """Perform Jobson-Korkie test for Sharpe ratio difference."""
        n = len(returns_a)

        sharpe_a = self._calculate_sharpe_ratio(returns_a)
        sharpe_b = self._calculate_sharpe_ratio(returns_b)
        sharpe_diff = sharpe_a - sharpe_b

        # Calculate correlation and variance
        corr = np.corrcoef(returns_a, returns_b)[0, 1]

        var_sharpe_diff = (1 / n) * (
            2 - 2 * corr * sharpe_a * sharpe_b + 0.5 * sharpe_a**2 + 0.5 * sharpe_b**2
        )

        if var_sharpe_diff <= 0:
            return 0, 1.0

        test_stat = sharpe_diff / np.sqrt(var_sharpe_diff)
        p_value = 2 * (1 - norm.cdf(abs(test_stat)))  # Two-tailed

        return test_stat, p_value

    def _calculate_sharpe_effect_size(
        self, returns_treatment: np.ndarray, returns_control: np.ndarray
    ) -> float:
        """Calculate effect size for Sharpe ratio difference."""
        sharpe_treatment = self._calculate_sharpe_ratio(returns_treatment)
        sharpe_control = self._calculate_sharpe_ratio(returns_control)

        # Pooled standard deviation approach adapted for Sharpe ratios
        pooled_variance = (np.var(returns_treatment, ddof=1) + np.var(returns_control, ddof=1)) / 2
        pooled_std = np.sqrt(pooled_variance)

        if pooled_std == 0:
            return 0.0

        # Effect size as standardized difference
        effect_size = (
            (sharpe_treatment - sharpe_control) * np.sqrt(len(returns_treatment)) / pooled_std
        )

        return effect_size

    def _calculate_return_effect_size(
        self, returns_treatment: np.ndarray, returns_control: np.ndarray
    ) -> float:
        """Calculate Cohen's d for return differences."""
        mean_treatment = np.mean(returns_treatment)
        mean_control = np.mean(returns_control)

        pooled_std = np.sqrt(
            (
                (len(returns_treatment) - 1) * np.var(returns_treatment, ddof=1)
                + (len(returns_control) - 1) * np.var(returns_control, ddof=1)
            )
            / (len(returns_treatment) + len(returns_control) - 2)
        )

        if pooled_std == 0:
            return 0.0

        return (mean_treatment - mean_control) / pooled_std

    def _calculate_volatility_effect_size(
        self, returns_treatment: np.ndarray, returns_control: np.ndarray
    ) -> float:
        """Calculate effect size for volatility differences."""
        vol_treatment = np.std(returns_treatment, ddof=1)
        vol_control = np.std(returns_control, ddof=1)

        # Log ratio effect size for volatility
        if vol_control == 0:
            return np.inf if vol_treatment > 0 else 0.0

        return np.log(vol_treatment / vol_control)

    def _calculate_power_sharpe_test(
        self,
        returns_treatment: np.ndarray,
        returns_control: np.ndarray,
        effect_size: float,
        alpha: float,
    ) -> float:
        """Calculate statistical power for Sharpe ratio test."""
        len(returns_treatment)

        # Variance estimate for Sharpe difference under alternative
        var_estimate = self._calculate_jobson_korkie_variance(returns_treatment, returns_control)

        if var_estimate <= 0:
            return 0.0

        # Non-centrality parameter
        ncp = effect_size / np.sqrt(var_estimate)

        # Critical value
        z_critical = norm.ppf(1 - alpha / 2)

        # Power calculation
        power = 1 - norm.cdf(z_critical - ncp) + norm.cdf(-z_critical - ncp)

        return max(0, min(1, power))

    def _calculate_jobson_korkie_variance(
        self, returns_a: np.ndarray, returns_b: np.ndarray
    ) -> float:
        """Calculate Jobson-Korkie variance estimate."""
        n = len(returns_a)

        sharpe_a = self._calculate_sharpe_ratio(returns_a)
        sharpe_b = self._calculate_sharpe_ratio(returns_b)
        corr = np.corrcoef(returns_a, returns_b)[0, 1]

        variance = (1 / n) * (
            2 - 2 * corr * sharpe_a * sharpe_b + 0.5 * sharpe_a**2 + 0.5 * sharpe_b**2
        )

        return max(0, variance)

    def _sharpe_difference_confidence_interval(
        self,
        returns_treatment: np.ndarray,
        returns_control: np.ndarray,
        confidence_level: float = 0.95,
    ) -> tuple[float, float]:
        """Calculate confidence interval for Sharpe ratio difference."""
        sharpe_treatment = self._calculate_sharpe_ratio(returns_treatment)
        sharpe_control = self._calculate_sharpe_ratio(returns_control)
        sharpe_diff = sharpe_treatment - sharpe_control

        var_diff = self._calculate_jobson_korkie_variance(returns_treatment, returns_control)

        if var_diff <= 0:
            return sharpe_diff, sharpe_diff

        se_diff = np.sqrt(var_diff)
        alpha = 1 - confidence_level
        z_critical = norm.ppf(1 - alpha / 2)

        margin_error = z_critical * se_diff

        return sharpe_diff - margin_error, sharpe_diff + margin_error

    def _effect_size_confidence_interval(
        self, returns_treatment: np.ndarray, returns_control: np.ndarray, metric: str
    ) -> tuple[float, float]:
        """Calculate confidence interval for effect size."""
        n1, n2 = len(returns_treatment), len(returns_control)

        if metric == "sharpe_ratio":
            effect_size = self._calculate_sharpe_effect_size(returns_treatment, returns_control)
        elif metric == "return":
            effect_size = self._calculate_return_effect_size(returns_treatment, returns_control)
        else:
            effect_size = self._calculate_volatility_effect_size(returns_treatment, returns_control)

        # Approximate standard error for Cohen's d
        se_effect = np.sqrt((n1 + n2) / (n1 * n2) + effect_size**2 / (2 * (n1 + n2)))

        z_critical = norm.ppf(0.975)  # 95% CI
        margin_error = z_critical * se_effect

        return effect_size - margin_error, effect_size + margin_error

    def _calculate_required_sample_size(
        self,
        returns_treatment: np.ndarray,
        returns_control: np.ndarray,
        effect_size: float,
        power: float,
        alpha: float,
    ) -> int:
        """Calculate required sample size for desired power."""
        # Initial variance estimate
        var_estimate = self._calculate_jobson_korkie_variance(returns_treatment, returns_control)

        if var_estimate <= 0:
            return len(returns_treatment)

        # Power calculation function
        def power_func(n):
            ncp = effect_size / np.sqrt(var_estimate / n)
            z_critical = norm.ppf(1 - alpha / 2)
            return 1 - norm.cdf(z_critical - ncp) + norm.cdf(-z_critical - ncp)

        # Search for required sample size
        for n in range(10, 10000):
            if power_func(n) >= power:
                return n

        return 10000  # Maximum reasonable sample size

    def _calculate_detectable_effect_size(
        self, returns_treatment: np.ndarray, returns_control: np.ndarray, power: float, alpha: float
    ) -> float:
        """Calculate minimum detectable effect size with current sample."""
        len(returns_treatment)
        var_estimate = self._calculate_jobson_korkie_variance(returns_treatment, returns_control)

        if var_estimate <= 0:
            return 0.0

        # Solve for effect size given power
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        detectable_effect = (z_alpha + z_beta) * np.sqrt(var_estimate)

        return detectable_effect

    def _interpret_sharpe_effect_size(self, effect_size: float) -> str:
        """Interpret effect size for Sharpe ratio differences."""
        abs_effect = abs(effect_size)

        if abs_effect < 0.1:
            return EffectSizeInterpretation.NEGLIGIBLE.value
        elif abs_effect < 0.3:
            return EffectSizeInterpretation.SMALL.value
        elif abs_effect < 0.8:
            return EffectSizeInterpretation.MEDIUM.value
        else:
            return EffectSizeInterpretation.LARGE.value

    def _interpret_return_effect_size(self, effect_size: float) -> str:
        """Interpret effect size for return differences (Cohen's conventions)."""
        abs_effect = abs(effect_size)

        if abs_effect < 0.2:
            return EffectSizeInterpretation.NEGLIGIBLE.value
        elif abs_effect < 0.5:
            return EffectSizeInterpretation.SMALL.value
        elif abs_effect < 0.8:
            return EffectSizeInterpretation.MEDIUM.value
        else:
            return EffectSizeInterpretation.LARGE.value

    def _interpret_volatility_effect_size(self, effect_size: float) -> str:
        """Interpret effect size for volatility differences."""
        abs_effect = abs(effect_size)

        if abs_effect < 0.1:
            return EffectSizeInterpretation.NEGLIGIBLE.value
        elif abs_effect < 0.25:
            return EffectSizeInterpretation.SMALL.value
        elif abs_effect < 0.5:
            return EffectSizeInterpretation.MEDIUM.value
        else:
            return EffectSizeInterpretation.LARGE.value

    def _cohens_category(self, effect_size: float) -> str:
        """Classify effect size according to Cohen's categories."""
        abs_effect = abs(effect_size)

        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

    def _assess_practical_significance(self, effect_size: float, metric: str) -> str:
        """Assess practical significance of effect size."""
        abs_effect = abs(effect_size)

        if metric == "sharpe_ratio":
            if abs_effect >= 0.5:
                return "practically significant"
            elif abs_effect >= 0.2:
                return "potentially meaningful"
            else:
                return "practically negligible"
        else:
            if abs_effect >= 0.8:
                return "practically significant"
            elif abs_effect >= 0.5:
                return "potentially meaningful"
            else:
                return "practically negligible"

    def _generate_power_recommendations(
        self, is_adequate: bool, power_deficit: float, sample_size_deficit: int
    ) -> list[str]:
        """Generate recommendations based on power analysis."""
        recommendations = []

        if is_adequate:
            recommendations.append("Current sample size provides adequate statistical power.")
        else:
            recommendations.append(
                f"Statistical power is insufficient (deficit: {power_deficit:.3f})."
            )

            if sample_size_deficit > 0:
                recommendations.append(
                    f"Increase sample size by {sample_size_deficit} observations to achieve desired power."
                )

            if power_deficit > 0.2:
                recommendations.append(
                    "Consider adjusting effect size expectations or significance level."
                )
                recommendations.append(
                    "Results may have high risk of Type II error (false negatives)."
                )

        return recommendations

    def _interpret_bayesian_results(
        self, prob_positive: float, prob_meaningful: float, bayes_factor: float
    ) -> str:
        """Interpret Bayesian hypothesis testing results."""
        interpretation = []

        # Evidence strength from Bayes Factor
        if bayes_factor > 100:
            bf_strength = "extreme evidence for"
        elif bayes_factor > 30:
            bf_strength = "very strong evidence for"
        elif bayes_factor > 10:
            bf_strength = "strong evidence for"
        elif bayes_factor > 3:
            bf_strength = "moderate evidence for"
        elif bayes_factor > 1:
            bf_strength = "weak evidence for"
        else:
            bf_strength = "evidence against"

        interpretation.append(f"Bayes factor provides {bf_strength} performance improvement.")

        # Probability interpretations
        interpretation.append(f"Probability of positive effect: {prob_positive:.1%}")
        interpretation.append(
            f"Probability of meaningful improvement (>0.2): {prob_meaningful:.1%}"
        )

        # Overall conclusion
        if prob_meaningful > 0.95:
            interpretation.append("Strong Bayesian evidence for meaningful improvement.")
        elif prob_meaningful > 0.80:
            interpretation.append("Moderate Bayesian evidence for meaningful improvement.")
        elif prob_positive > 0.95:
            interpretation.append("Strong evidence for positive effect, but magnitude uncertain.")
        else:
            interpretation.append("Weak or insufficient evidence for performance improvement.")

        return " ".join(interpretation)
