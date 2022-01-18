from typing import Union
import numpy as np
import pandas as pd
import scipy.stats as st

CONFIDENCE_MARGIN = 1.96


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
    return not ((actions.nunique() == 2) and (neutral_indicator in actions.unique().tolist()))


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


def get_standard_error_mean(sample_size, std, z: float = CONFIDENCE_MARGIN):
    return z * std / np.sqrt(sample_size)


def get_standard_error_proportion(sample_size, proportion_estimate, z: float = CONFIDENCE_MARGIN):
    return z * np.sqrt(proportion_estimate * (1 - proportion_estimate) / sample_size)


def proportions_test(proportion_est_1, proportion_est_2, sample_siz_1, sample_siz_2):
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
    first_set_div = (std_1 ** 2) / sample_siz_1
    second_set_div = (std_2 ** 2) / sample_siz_2

    # standard error
    se = np.sqrt(first_set_div + second_set_div)
    # degrees of freedom
    numerator = ((first_set_div + second_set_div) ** 2)
    denominator = (first_set_div ** 2) / (sample_siz_1 - 1) + (second_set_div ** 2) / (sample_siz_2 - 1)
    df = numerator / denominator
    # compute t-score
    t = (mu_1 - mu_2) / se
    # p values
    p_vals = (t > 0) * (1 - st.t.cdf(t, df) + st.t.cdf(-t, df)) + (t <= 0) * (st.t.cdf(t, df) + 1 - st.t.cdf(-t, df))

    return p_vals
