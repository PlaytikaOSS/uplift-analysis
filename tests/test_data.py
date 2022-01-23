import pytest
import pandas as pd
import numpy as np
from uplift_analysis import data


def test_eval_set():
    """Test the EvalSet dataclass"""

    # create some scored dataframe
    action_set = ['A', 'B', 'C', 'D']
    siz = 1000

    scored_df = pd.DataFrame({
        'obs': np.random.choice(action_set, size=siz, replace=True),
        'resp': np.random.randint(2, size=siz),
        'sc': np.random.standard_normal(size=siz),
        'rec': np.random.choice(action_set, size=siz, replace=True)
    })

    # make sure it can be used for creating a corresponding EvalSet object
    eval_set = data.EvalSet(df=scored_df)

    # should fail as the EvalSet object was not configured with the corresponding field names
    with pytest.raises(Exception) as _:
        eval_set.set_problem_type()

    # should fail as the provided configuration does not comply with the permitted types that are expected
    with pytest.raises(Exception) as _:
        eval_set.set_props(observed_action_field=42)

    # configure the EvalSet object
    eval_set.set_props(observed_action_field='obs',
                       response_field='resp',
                       score_field='sc',
                       proposed_action_field='rec')

    # still should fail as the "control_indicator" was not configured
    with pytest.raises(Exception) as _:
        eval_set.set_problem_type()

    # these should succeed
    eval_set.set_props(control_indicator='A')
    eval_set.set_problem_type()
