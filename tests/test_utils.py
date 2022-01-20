import pytest
import numpy as np
import pandas as pd
from uplift_analysis import utils


def test_is_multi_action():
    """Test the is_multi_action function"""

    siz = 1000
    multi = 5

    neutral_action = 0
    binary_action = pd.Series(np.random.randint(2, size=siz))
    multi_action = pd.Series(np.random.randint(multi + 1, size=siz))

    assert not utils.is_multi_action(actions=binary_action, neutral_indicator=neutral_action)
    assert utils.is_multi_action(actions=multi_action, neutral_indicator=neutral_action)

    neutral_action = 'No'
    binary_action = pd.Series(np.random.choice(['Yes'] + [neutral_action], size=(siz,), replace=True))
    multi_action = pd.Series(np.random.choice(['Yes', 'Maybe', 'IDK'] + [neutral_action], size=(siz,), replace=True))

    assert not utils.is_multi_action(actions=binary_action, neutral_indicator=neutral_action)
    assert utils.is_multi_action(actions=multi_action, neutral_indicator=neutral_action)


def test_is_binary_response():
    siz = 1000
    binary = pd.Series(np.random.randint(2, size=siz))
    real = pd.Series(np.random.standard_normal(size=siz))

    assert utils.is_binary_response(binary)
    assert not utils.is_binary_response(real)
