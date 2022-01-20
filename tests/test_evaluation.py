import pytest
import pandas as pd
import numpy as np
from uplift_analysis import evaluation, data


def test_initialization():
    """Test the initialization of the Evaluator"""
    with pytest.raises(Exception) as _:
        ev = evaluation.Evaluator(sets_config={'response_field': 5})

    # this should succeed
    ev = evaluation.Evaluator(sets_config={'response_field': 'response'})


def test_single_eval_set():
    """Test the usage of the Evaluator on a single EvalSet object"""
    siz = 1000

    for num_actions in [2, 5]:
        scored_df = pd.DataFrame(
            {
                'recommended_action': np.random.randint(num_actions, size=siz),
                'score': np.random.standard_normal(size=siz),
                'observed_action': np.random.randint(num_actions, size=siz),
                'response': np.random.standard_normal(size=siz)
            }
        )

        eval_set = data.EvalSet(df=scored_df,
                                observed_action_field='observed_action',
                                response_field='response',
                                score_field='score',
                                proposed_action_field='recommended_action',
                                control_indicator=0)
        evaluator = evaluation.Evaluator()
        with pytest.raises(Exception) as _:
            evaluator.evaluate_set(input_set=None)

        eval_res, summary = evaluator.eval_and_show(data=eval_set, title_suffix='single', show_random=True,
                                                    num_random_rep=4)

        assert isinstance(eval_res, data.EvalSet)
        assert isinstance(summary, dict)

        with pytest.raises(Exception) as _:
            evaluator.visualize(eval_res=np.array([1, 2]))

        with pytest.raises(Exception) as _:
            evaluator.visualize(eval_res=eval_res, specify=['Untitled'])

        # this should succeed
        evaluator.visualize(eval_res=eval_res, specify=['agreements'])

        # this should fail
        eval_res.is_evaluated = False
        with pytest.raises(Exception) as _:
            evaluator.visualize(eval_res=eval_res, specify=['agreements'])


def test_multiple_eval_set():
    """Test the usage of the Evaluator on multiple EvalSet objects"""

    eval_sets = dict()
    for num_actions, siz, name in zip([2, 5], [1000, 2000], ['Eval_A', 'Eval_B']):
        scored_df = pd.DataFrame(
            {
                'recommended_action': np.random.randint(num_actions, size=siz),
                'score': np.random.standard_normal(size=siz),
                'observed_action': np.random.randint(num_actions, size=siz),
                'response': np.random.standard_normal(size=siz)
            }
        )

        eval_sets[name] = data.EvalSet(df=scored_df,
                                       observed_action_field='observed_action',
                                       response_field='response',
                                       score_field='score',
                                       proposed_action_field='recommended_action',
                                       control_indicator=0)

    evaluator = evaluation.Evaluator()
    with pytest.raises(Exception) as _:
        evaluator.evaluate_multiple(input_set=None)

    eval_res, summary = evaluator.eval_and_show(data=eval_sets, title_suffix='multiple', average=True)

    assert isinstance(eval_res, dict)
    assert isinstance(summary, pd.DataFrame)

    # this should succeed
    evaluator.visualize(eval_res=eval_res, specify=['agreements'])

    # this should fail
    eval_res['Eval_A'].is_evaluated = False
    with pytest.raises(Exception) as _:
        evaluator.visualize(eval_res=eval_res, specify=['agreements'])
