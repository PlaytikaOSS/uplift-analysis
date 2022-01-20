import pytest
import pandas as pd
import numpy as np
from uplift_analysis import data


def test_eval_set():
    """Test the EvalSet dataclass"""

    action_set = ['A', 'B', 'C', 'D']
    siz = 1000

    scored_df = pd.DataFrame({
        'obs': np.random.choice(action_set, size=siz, replace=True),
        'resp': np.random.randint(2, size=siz),
        'sc': np.random.standard_normal(size=siz),
        'rec': np.random.choice(action_set, size=siz, replace=True)
    })

    eval_set = data.EvalSet(df=scored_df)

    with pytest.raises(Exception) as _:
        eval_set.set_problem_type()

    with pytest.raises(Exception) as _:
        eval_set.set_props(observed_action_field=42)

    eval_set.set_props(observed_action_field='obs',
                       response_field='resp',
                       score_field='sc',
                       proposed_action_field='rec')

    with pytest.raises(Exception) as _:
        eval_set.set_problem_type()

    # this should succeed
    eval_set.set_props(control_indicator='A')
    eval_set.set_problem_type()
