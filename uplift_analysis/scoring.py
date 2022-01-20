# -*- coding: utf-8 -*-
"""
This module implements a scoring utility wrapped as a class named ``Scorer``.
Given a set of (or a single) scoring configurations, each of which specifies the relevant fields, and the specific
function to apply to these fields, each observation within the input dataset is scored.
In case of multiple scoring configurations, the scores of the methods are combined into a new score, weighting the
magnitude of the score, associated with each action (relevant for a multiple actions scenario).

Notes:
    - ``Scorer`` also supports use-cases with multiple treatments.

"""
from typing import Dict, List, Union, Tuple, Optional
import scipy
import numpy as np
import pandas as pd


class Scorer:
    """
    The Scorer class is used for scoring observations on a given dataset, according to a provided configuration,
    or a set of configurations.

    Parameters
    ----------
    scoring_configuration: Optional[Union[Dict, List[Dict]]]
        A list of configurations or a single configuration (each of which represented as dict) specifying scoring
        methods.

    """
    scoring_configuration: Union[Dict, List[Dict]]

    def __init__(self, scoring_configuration: Optional[Union[Dict, List[Dict]]] = None):
        """
        See class docstring for further details regarding class functionality.
        """
        self.set_scoring_config(scoring_configuration)

    def set_scoring_config(self, scoring_configuration: Union[Dict, List[Dict]]) -> None:
        """
        A method for setting the scoring configuration associated with the object.

        Parameters
        ----------
        scoring_configuration: Union[Dict, List[Dict]]
            A list of configurations or a single configuration (each of which represented as dict) specifying scoring
            methods.

        """
        if scoring_configuration is not None:
            self.scoring_configuration = scoring_configuration

    def calculate_scores(self, dataset: Union[Dict[str, np.ndarray], pd.DataFrame],
                         scoring_configuration: Union[Dict, List[Dict]] = None) -> Tuple:
        """
        This function serves as the primary interface of the class. Given a dataset, and scoring configuration,
        this function returns the corresponding scores for each observation in the set, accompanied with the
        recommended action.

        Parameters
        ----------
        dataset: Union[Dict[str, np.ndarray], pd.DataFrame]
            the dataset to be scored.
        scoring_configuration: Union[Dict, List[Dict]]
            the configuration according to which the observations will be scored.

        Returns
        -------
        rankings: np.ndarray
            The relative rank (0,1] - highest means highest uplift score - of each observation in the dataset.
        scored_actions: np.ndarray
            The serial index of the action corresponding to the highest score, per observation.
        scores: np.ndarray
            The score for each observation, according to the provided configuration.
        action_dim: int
            The quantity of actions taken into account.
        """

        # if no configuration was provided, make sure the object was already configured beforehand.
        assert scoring_configuration or self.scoring_configuration
        if not scoring_configuration:
            # inherit the pre-defined configuration
            scoring_configuration = self.scoring_configuration

        if isinstance(scoring_configuration, dict):  # in case a single observation was provided
            return self.single_scoring_method_calc(dataset=dataset, scoring_method=scoring_configuration)
        else:  # in case the configuration lists a set of configuration methods
            return self.multiple_scoring_methods_calc(dataset=dataset, scoring_methods=scoring_configuration)

    def multiple_scoring_methods_calc(self, dataset: Union[Dict[str, np.ndarray], pd.DataFrame],
                                      scoring_methods: List[Dict]):
        """
        This function applies a set of scoring method configurations to the provided dataset,
        and returns the resulting scores, and the recommended actions after combining the set of computed scores.

        Parameters
        ----------
        dataset: Union[Dict[str, np.ndarray], pd.DataFrame]
            The set of observations to score.
        scoring_methods: List[Dict]
            A list of dictionaries representing the scoring method configurations.

        Returns
        -------
        rankings: np.ndarray
            The relative rank (0,1] - highest means highest uplift score - of each observation in the dataset.
        scored_actions: np.ndarray
            The serial index of the action corresponding to the highest score, per observation.
        scores: np.ndarray
            The score for each observation, according to the provided configuration.
        action_dim: int
            The quantity of actions taken into account.
        """

        rankings: Union[List, np.ndarray] = []  # populate with rankings of each single scoring method.
        scored_actions: Union[List, np.ndarray] = []  # populate with scored actions according to each single method.
        methods_scores: Union[List, np.ndarray] = []  # populate with scores according to each single method.

        # this variable will be assigned while iterating
        action_dim = None

        # iterate over the set of scoring methods
        for scoring_config in scoring_methods:
            # compute resulting scores according to the single method
            method_rankings, method_scored_actions, highest_score, score_dim = self.single_scoring_method_calc(
                dataset=dataset,
                scoring_method=scoring_config)

            # assign with the highest score_dim observed
            if action_dim is None or score_dim > action_dim:
                action_dim = score_dim

            # populate lists
            rankings.append(method_rankings)
            scored_actions.append(method_scored_actions)
            methods_scores.append(highest_score)

        # stack each list into a single array
        rankings = np.stack(rankings, axis=-1)
        scored_actions = np.stack(scored_actions, axis=-1)
        methods_scores = np.stack(methods_scores, axis=-1)

        # combine the recommendation of the set of configured methods
        return self.combine_scores(rankings, scored_actions, action_dim)

    def combine_scores(self, rankings: np.ndarray, scored_actions: np.ndarray, action_dim: int):
        """
        A function for combining the recommendations and scores resulting of multiple scoring methods, according
        to the relative rankings.

        Parameters
        ----------
        rankings: np.ndarray
            An array containing the relative ranking for each observation (row), according to each scoring
            method (column).
        scored_actions: np.ndarray
            An array containing the recommended action for each observation (row), according to each scoring
            method (column).
        action_dim: int
            The cardinality of the action space.

        Returns
        -------
        combined_rankings: np.ndarray
            The relative rank (0,1] - highest means highest uplift score - of each observation in the dataset.
        combined_score_action: np.ndarray
            The serial index of the action corresponding to the highest score, per observation.
        combined_score: np.ndarray
            The score for each observation, according to the provided configuration.
        action_dim: int
            The quantity of actions taken into account.

        """
        num_obs, num_methods = rankings.shape

        # aggregate weights for every action according to its corresponding rank
        # ======================================================================

        # initialize an array containing zeroed weights for each action, for each observation
        weights_matrix = np.zeros((num_obs, action_dim))
        # a dummy auxiliary array for dynamic indexing
        slicing_aux = np.arange(num_obs)

        # for each scoring method results set
        for method_idx in range(num_methods):
            # what were the recommended actions (per observation)?
            method_scored_actions = scored_actions[:, method_idx]
            # what was the relative rank (in terms of score) for each observation?
            corresponding_rank = rankings[:, method_idx]
            # accumulate the rank associated with the recommended actions with the weights matrix
            weights_matrix[slicing_aux, method_scored_actions] = weights_matrix[
                                                                     slicing_aux, method_scored_actions] + \
                                                                 corresponding_rank

        # count votes for each action across all methods
        df = pd.DataFrame(scored_actions)
        obligatory_columns = np.arange(0, action_dim).tolist()
        votes = pd.get_dummies(df.stack()).reindex(columns=obligatory_columns, fill_value=0).groupby(
            level=0).sum().values

        # multiply votes quantity by weights
        weighted_scores = np.multiply(weights_matrix, votes)

        # eventually the score is determined by the sum of the relative rankings across all the scoring methods
        combined_score = rankings.sum(axis=1)
        # the recommended action is determined according to the weighted scores
        combined_score_action = weighted_scores.argmax(axis=1)

        return self.rank_scores(combined_score), combined_score_action, combined_score, action_dim

    def single_scoring_method_calc(self, dataset: Union[Dict[str, np.ndarray], pd.DataFrame], scoring_method: Dict):
        """
        This function applies a single scoring method configuration to the provided dataset,
        and returns the resulting scores, and the recommended actions according to these scores.

        Parameters
        ----------
        dataset: Union[Dict[str, np.ndarray], pd.DataFrame]
            The set of observations to score.
        scoring_method: Dict
            A dictionary representing the scoring method configuration.

        Returns
        -------
        rankings: np.ndarray
            The relative rank (0,1] - highest means highest uplift score - of each observation in the dataset.
        scored_action: np.ndarray
            The serial index of the action corresponding to the highest score, per observation.
        observation_score: np.ndarray
            The score for each observation, according to the provided configuration.
        action_dim: int
            The quantity of actions taken into account.
        """
        # apply the scoring method configuration
        scores = self.score_computation(dataset=dataset, scoring_method=scoring_method)

        # if scoring resulted in a single dimension array
        if len(scores.shape) == 1 or scores.shape[1] == 1:
            observation_score = scores.squeeze()
            scored_action = np.ones_like(scores).squeeze()
        else:
            # by default, assuming the left column indicates scores corresponding to the neutral action
            force_active_decision = scoring_method.get('force_active_decision', True)
            if force_active_decision:
                # specify the column index associated with the action with the maximal score per observation,
                # without taking into account the left column
                observation_score = scores[:, 1:].max(axis=1)
                scored_action = scores[:, 1:].argmax(axis=1) + 1
            else:  # in case "no_action" is a legitimate decision to be scored as any other action
                # specify the column index associated with the action with the maximal score per observation
                observation_score = scores.max(axis=1)
                scored_action = scores.argmax(axis=1)

        return self.rank_scores(observation_score), scored_action, observation_score, scores.shape[1]

    def score_computation(self, dataset: Union[Dict[str, np.ndarray], pd.DataFrame],
                          scoring_method: Dict) -> np.ndarray:
        """
        This function uses a single scoring method configuration and applies it to the provided dataset,
        for score computation.

        Parameters
        ----------
        dataset: Union[Dict[str, np.ndarray], pd.DataFrame]
            The dataset containing the observations that require scoring.
        scoring_method: Dict
            A dictionary specifying the scoring method configuration.

        Returns
        -------
        np.ndarray
            The resulting scores, corresponding the provided scoring method configuration.
        """

        # which field represents the signal according to which the score will be computed
        scoring_field = scoring_method['scoring_field']
        # the function according to which the score will be computed
        scoring_func = scoring_method.get('scoring_func', self.identity_score_calc)
        if isinstance(scoring_func, str):
            # if the scoring function was provide as string, try to get the class attribute
            # corresponding to the function name
            scoring_func = getattr(self, scoring_func)
        # make sure that the resulting variable is eventually a function
        assert callable(scoring_func)

        # in scenarios when it is configured, state what is the reference signal
        reference_field: str = scoring_method.get('reference_field', None)
        reference_idx: int = scoring_method.get('reference_idx', None)
        if reference_field:
            if reference_idx is not None:
                # in cases when the reference signal is configured as a specific column of an array
                reference_signal = dataset[reference_field][:, [reference_idx]]
            else:  # in cases when no such index is specified, the entire field is used as a reference signal
                reference_signal = dataset[reference_field]
        else:  # when reference signal is not configured
            reference_signal = None

        # apply the configured function, providing the specific scoring field, accompanied by the reference signal
        return scoring_func(dataset[scoring_field], reference_signal)

    @staticmethod
    def binary_score_calc(action_est, no_action_est):
        return scipy.special.expit(action_est) - scipy.special.expit(no_action_est)

    @staticmethod
    def sigmoid_frac_score_calc(action_est, no_action_est):
        return scipy.special.expit(action_est) / scipy.special.expit(no_action_est)

    @staticmethod
    def cont_score_calc(action_est, no_action_est):
        return action_est - no_action_est

    @staticmethod
    def identity_score_calc(action_est, no_action_est):
        return action_est

    @staticmethod
    def rank_scores(observation_score: np.ndarray) -> np.ndarray:
        """
        A method for computing relative rank (among the provided dataset) for each observation, according
        to the computed score.

        Parameters
        ----------
        observation_score: np.ndarray
            An array representing the score for each observation.

        Returns
        -------
        np.ndarray
            relative value in the range (0,1] indicating score rank (within the given dataset),
            for each of the observations.

        """

        num_obs = observation_score.shape[0]
        # infer scores order within the dataset
        sorted_order = np.argsort(observation_score)

        ranks = np.empty_like(sorted_order)
        # ranks will be 1-based
        ranks[sorted_order] = np.arange(num_obs) + 1

        # normalize by the size of the dataset for scaling the rank
        return ranks / num_obs
