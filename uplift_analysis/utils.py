# -*- coding: utf-8 -*-
"""
This module implements some basic utility functions required for the analysis and evaluation procedure.
"""

from typing import Union, Optional
import numpy as np
import pandas as pd
import scipy.stats as st

# Confidence Interval Setting
CONFIDENCE_MARGIN = 1.96  # the default one-sided margin for generating standard error ranges. corresponds to a 95% CI.


def is_multi_action(actions: pd.Series, neutral_indicator: Union[int, str]) -> bool:
    """
    This method checks whether the input dataframe is associated with a single action (except for the neutral
    action) or with a multitude of possible actions (multiple treatments).

    Parameters
    ----------
    actions: pd.Series
        A Pandas series representing a set of observed actions.
    neutral_indicator: Union[int,str]
        The action value associated with the neutral action.

    Returns
    -------
    bool
        A boolean indicating if the set is associated with multiple actions (True).

    """
    unique_actions = actions.unique().tolist()
    assert neutral_indicator in unique_actions
    return not ((len(unique_actions) == 2) and (neutral_indicator in unique_actions))


def is_binary_response(responses: pd.Series) -> bool:
    """
    This method checks whether the input dataframe is associated with a response of binary type.

    Parameters
    ----------
    responses: pd.Series
        A Pandas series representing a set of observed responses.

    Returns
    -------
    bool
        A boolean indicating if the set is associated with a binary response (True).

    """
    return (responses.dtype == bool) or ((responses.nunique() == 2) and (all(responses.isin([0, 1]))))


def get_standard_error_mean(sample_size, std, z: Optional[float] = CONFIDENCE_MARGIN):
    """
    A function for computing the one-sided margin, corresponding to a desired confidence interval coverage of the
    standard error of the sample mean estimator.
    Refer to `this page
    <https://www.statisticshowto.com/standard-error-of-measurement/>`__ for more details.

    Parameters
    ----------
    sample_size:
        The size of the sample (number of observations) for which the standard error estimation is required. It could be
        a scalar value, for a single computation, or an array-like input for multiple computations at once.
    std:
        The standard deviations, associated with each of the elements in ``sample_size``.
    z: Optional[float]
        The one-sided cofidence interval covrage, corresponding to the standrad normal distribution. The default value
        corresponds to a 95% confidence interval.

    Returns
    -------
    The standard error of mean estimation corresponding to the provided sample sizes and standard deviations.
    """

    return z * std / np.sqrt(sample_size)


def get_standard_error_proportion(sample_size, proportion_estimate, z: float = CONFIDENCE_MARGIN):
    """
    A function for computing the one-sided margin, corresponding to a desired confidence interval coverage of the
    standard error of the proportion estimator (expectation of a binary random variable).
    Refer to `this page
    <https://www.statology.org/standard-error-of-proportion/>`__ for more details.

    Parameters
    ----------
    sample_size:
        The size of the sample (number of observations) for which the standard error estimation is required. It could be
        a scalar value, for a single computation, or an array-like input for multiple computations at once.
    proportion_estimate:
        The estimated proportions, associated with each of the elements in ``sample_size``.
    z: Optional[float]
        The one-sided cofidence interval covrage, corresponding to the standrad normal distribution. The default value
        corresponds to a 95% confidence interval.

    Returns
    -------
    The standard error of proportion estimation corresponding to the provided sample sizes and proportion_estimates.
    """

    return z * np.sqrt(proportion_estimate * (1 - proportion_estimate) / sample_size)


def proportions_test(proportion_est_1, proportion_est_2, sample_siz_1, sample_siz_2):
    """
    This function implements an hypothesis testing for the difference between proportions of two groups.

    Given the proportion estimates of two groups, and the sample size associated with each of these groups,
    it tests the null hypothesis, that states that the proportions of the populations from which the two groups were
    sampled is identical. The alternative hypothesis in this case, is two-tailed, and it simply states, that the
    proportions of the populations from which the two groups are sampled, is different. The two-tailed hypothesis,
    implies that the order of the two groups in this case is arbitrary.

    For more detailes, see `Stat Trek <https://stattrek.com/>`_ page on `Hypothesis Test: Difference Between Proportions
    <https://stattrek.com/hypothesis-test/difference-in-proportions.aspx>`_.

    All the inputs can be array-like for performing multiple computations at once, or scalar values, for performing
    a single test.

    Parameters
    ----------
    proportion_est_1
        The estimated proportion for the first group.
    proportion_est_2
        The estimated proportion for the second group.
    sample_siz_1
        The sample size of the first group.
    sample_siz_2
        The sample size of the second group.

    Returns
    -------
    p_vals:
        The result(s) of the test(s) performed, in the form of p-value(s). Each p-value represents the probability that
        similar findings will occur in the situation where the null hypothesis is true.
    """
    # pooled sample proportion
    p = (proportion_est_1 * sample_siz_1 + proportion_est_2 * sample_siz_2) / (sample_siz_1 + sample_siz_2)
    # standard error
    se = np.sqrt(p * (1 - p) * ((1 / sample_siz_1) + (1 / sample_siz_2)))
    # compute z-score
    z = (proportion_est_1 - proportion_est_2) / se
    # p-values
    p_vals = (z > 0) * (1 - st.norm.cdf(z) + st.norm.cdf(-z)) + (z <= 0) * (st.norm.cdf(z) + 1 - st.norm.cdf(-z))

    return p_vals


def t_test(mu_1, mu_2, sample_siz_1, sample_siz_2, std_1, std_2):
    """
    This function implements an hypothesis testing for the difference between means of two groups.

    Given the mean estimates of two groups, the sample size associated with each of these groups, and their standarad
    deviations, the function uses **t-test** to examine the null hypothesis, that states that the means of the
    populations from which the two groups were sampled is identical.
    The alternative hypothesis in this case, is two-tailed, and it simply states, that the means of the populations
    from which the two groups are sampled, is different. The two-tailed hypothesis, implies that the order of the two
    groups in this case is arbitrary.

    For more detailes, see `Stat Trek <https://stattrek.com/>`__ page on `Hypothesis Test: Difference Between Means
    <https://stattrek.com/hypothesis-test/difference-in-means.aspx?tutorial=AP>`__.

    All the inputs can be array-like for performing multiple computations at once, or scalar values, for performing
    a single test.

    Parameters
    ----------
    mu_1
        The estimated mean for the first group.
    mu_2
        The estimated mean for the second group.
    sample_siz_1
        The sample size of the first group.
    sample_siz_2
        The sample size of the second group.
    std_1
        The standard deviation of the first group.
    std_2
        The standard deviation of the second group.

    Returns
    -------
    p_vals:
        The result(s) of the test(s) performed, in the form of p-value(s). Each p-value represents the probability that
        similar findings will occur in the situation where the null hypothesis is true.
    """
    # compute standard error or the sampling distribution
    first_set_div = (std_1 ** 2) / sample_siz_1
    second_set_div = (std_2 ** 2) / sample_siz_2
    se = np.sqrt(first_set_div + second_set_div)

    # degrees of freedom
    numerator = ((first_set_div + second_set_div) ** 2)
    denominator = (first_set_div ** 2) / (sample_siz_1 - 1) + (second_set_div ** 2) / (sample_siz_2 - 1)
    df = numerator / denominator

    # compute t-score
    t = (mu_1 - mu_2) / se
    # p values according to t-distribution
    p_vals = (t > 0) * (1 - st.t.cdf(t, df) + st.t.cdf(-t, df)) + (t <= 0) * (st.t.cdf(t, df) + 1 - st.t.cdf(-t, df))

    return p_vals
