import pytest
import pandas as pd
import numpy as np
from uplift_analysis import visualization, data


def test_visualize_selection_distribution():
    """Test visualize_selection_distribution function"""

    # should fail as the function expect either a dataframe or an EvalSet object
    with pytest.raises(Exception) as _:
        visualization.visualize_selection_distribution(eval_res=None)

    # create some dummy scored dataframe
    num_actions = 4
    siz = 1000
    scored_df = pd.DataFrame({
        'observed_action_field': np.random.randint(num_actions + 1, size=siz),
        'response': np.random.randint(2, size=siz),
        'score': np.random.standard_normal(size=siz),
        'proposed_action': np.random.randint(num_actions + 1, size=siz)
    })

    # should fail as the input dataframe does not have a `normalized_index` column
    with pytest.raises(Exception) as _:
        visualization.visualize_selection_distribution(eval_res=scored_df, column_name='proposed_action')
    scored_df.loc[:, 'normalized_index'] = 1 / (np.arange(len(scored_df)) + 1)[::-1]

    # still should fail as no specific column was provided
    with pytest.raises(Exception) as _:
        visualization.visualize_selection_distribution(eval_res=scored_df)

    # this should succeed
    visualization.visualize_selection_distribution(eval_res=scored_df, column_name='proposed_action')

    # create a corresponding EvalSet object
    eval_res = data.EvalSet(df=scored_df)

    # should fail as the EvalSet object was not evaluated
    with pytest.raises(Exception) as _:
        visualization.visualize_selection_distribution(eval_res=eval_res)

    eval_res.is_evaluated = True
    # this should succeed
    visualization.visualize_selection_distribution(eval_res=eval_res)


def test_should_average():
    """Test the should_average function"""

    eval_res = dict(a=1)
    avg, num_sets = visualization.should_average(eval_res, average=True)
    # in the case of an eval_res with a single element, avg should be False

    assert not avg
