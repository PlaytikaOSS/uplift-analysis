import pytest
import pandas as pd
import numpy as np
from uplift_analysis import scoring


def test_scorer():
    """Test the Scorer class"""

    num_actions = 5  # number of actions
    siz = 1000  # sample size

    # create a dummy dataset containing two scores arrays
    # each array contain score for each action
    dataset = {'some_model_outputs': np.random.standard_normal(size=(siz, num_actions)),
               'another_model_outputs': np.random.standard_normal(size=(siz, num_actions))}

    # initialize scoring object
    scorer = scoring.Scorer()

    # should fail as the scorer was not configured
    with pytest.raises(Exception) as _:
        scorer.calculate_scores(dataset=dataset)

    # configure the scorer
    scorer.set_scoring_config({'name': 'identity',
                               'scoring_field': 'some_model_outputs',
                               'scoring_func': 'identity_score_calc'})
    # scoring should succeed
    ranking, recommended_action, score, action_dim = scorer.calculate_scores(dataset=dataset)

    # scoring_func should be either a callable or a listed method
    with pytest.raises(Exception) as _:
        scorer.calculate_scores(dataset=dataset,
                                scoring_configuration={'name': 'identity',
                                                       'scoring_field': 'some_model_outputs',
                                                       'scoring_func': 5})
    # should fail for the same reason
    with pytest.raises(Exception) as _:
        scorer.calculate_scores(dataset=dataset,
                                scoring_configuration={'name': 'identity',
                                                       'scoring_field': 'some_model_outputs',
                                                       'scoring_func': 'unknown_func'})

    # test the case of multiple scoring methods
    scoring_config = [
        {
            'name': 'cont',
            'scoring_field': 'another_model_outputs',
            'reference_field': 'another_model_outputs',
            'reference_idx': 0,
            'scoring_func': 'cont_score_calc'
        },
        {
            'name': 'binary',
            'scoring_field': 'some_model_outputs',
            'reference_field': 'some_model_outputs',
            'reference_idx': 0,
            'scoring_func': 'binary_score_calc'
        },
        {
            'name': 'sigmoid',
            'scoring_field': 'some_model_outputs',
            'reference_field': 'some_model_outputs',
            'reference_idx': 0,
            'scoring_func': 'sigmoid_frac_score_calc'
        },
    ]

    scorer.set_scoring_config(scoring_configuration=scoring_config)
    # should succeed
    scorer.calculate_scores(dataset=dataset)
