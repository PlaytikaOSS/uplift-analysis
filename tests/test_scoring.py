import pytest
import pandas as pd
import numpy as np
from uplift_analysis import scoring


def test_scorer():
    """Test the Scorer class"""

    num_actions = 5
    siz = 1000

    dataset = {'some_model_outputs': np.random.standard_normal(size=(siz, num_actions)),
               'another_model_outputs': np.random.standard_normal(size=(siz, num_actions))}

    scorer = scoring.Scorer()

    with pytest.raises(Exception) as _:
        scorer.calculate_scores(dataset=dataset)

    scorer.set_scoring_config({'name': 'identity',
                               'scoring_field': 'some_model_outputs',
                               'scoring_func': 'identity_score_calc'})
    # should succeed
    ranking, recommended_action, score, action_dim = scorer.calculate_scores(dataset=dataset)

    with pytest.raises(Exception) as _:
        scorer.calculate_scores(dataset=dataset,
                                scoring_configuration={'name': 'identity',
                                                       'scoring_field': 'some_model_outputs',
                                                       'scoring_func': 5})

    with pytest.raises(Exception) as _:
        scorer.calculate_scores(dataset=dataset,
                                scoring_configuration={'name': 'identity',
                                                       'scoring_field': 'some_model_outputs',
                                                       'scoring_func': 'unknown_func'})

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
