import pytest
import numpy as np
import pandas as pd
from uplift_analysis import utils


def test_is_multi_action():
    """Test the is_multi_action function"""

    siz = 1000  # sample size
    multi = 5  # number of actions

    # create series for each case:
    neutral_action = 0
    binary_action = pd.Series(np.random.randint(2, size=siz))  # binary action scenario
    multi_action = pd.Series(np.random.randint(multi + 1, size=siz))  # multiple actions

    # make sure the function returns the expected answer
    assert not utils.is_multi_action(actions=binary_action, neutral_indicator=neutral_action)
    assert utils.is_multi_action(actions=multi_action, neutral_indicator=neutral_action)

    # Do the same for the case where the action is not necessarily numeric
    neutral_action = 'No'
    binary_action = pd.Series(np.random.choice(['Yes'] + [neutral_action], size=(siz,), replace=True))
    multi_action = pd.Series(np.random.choice(['Yes', 'Maybe', 'IDK'] + [neutral_action], size=(siz,), replace=True))

    assert not utils.is_multi_action(actions=binary_action, neutral_indicator=neutral_action)
    assert utils.is_multi_action(actions=multi_action, neutral_indicator=neutral_action)


def test_is_binary_response():
    """ Test the is_binary_response function"""
    siz = 1000
    binary = pd.Series(np.random.randint(2, size=siz))
    real = pd.Series(np.random.standard_normal(size=siz))

    assert utils.is_binary_response(binary)
    assert not utils.is_binary_response(real)
