# -*- coding: utf-8 -*-
"""
This module implements an uplift evaluation utility wrapped as a class named Evaluator.
After specifying the relevant field names on the corresponding dataset (represented as a pandas dataframe),
the provided dataframe will be analyzed, in terms of uplift.

For such evaluation, user needs to state what will indicate the neutral action (whether it will be the numeric value
of 0, or a string such as 'Control').

The evaluation takes into account cases of multiple treatments; In case the actions in the evaluated set are binary
(single treatment) the evaluation metrics will end up being identical for the multiple-actions scenario and the binary
one.

Notes:
    - Evaluator also supports use-cases with multiple treatments.
    - Currently the uplift evaluation supports only responses of binary type.

"""

from typing import Dict, Union, List, Tuple, Optional
import pandas as pd
from scipy.integrate import simps
from .data import EvalSet


class Evaluator:
    """
    Evaluator class is used for uplift evaluation of a given dataset, represented as a pandas Dataframe, or as
    an EvalSet object.
    Its primary interface is the method ``evaluate_set()``.

    Parameters
    ----------
    sets_config: Optional[Union[Dict, None]]
        A set of configuration parameters, to be used for creating EvalSet objects of pandas Dataframes, if required.

    """

    _sets_config: Union[Dict, None] = None

    def __init__(self, sets_config: Optional[Union[Dict, None]] = None):
        self.sets_config = sets_config

    @property
    def sets_config(self) -> Union[Dict, None]:
        return self._sets_config

    @sets_config.setter
    def sets_config(self, v: Union[Dict, None]) -> None:
        if isinstance(v, dict):
            expected_fields = EvalSet.conf_fields()
            for key, value in v.items():
                assert key in expected_fields
                # the field name argument must be of the expected type
                expected_types = expected_fields[key]
                assert any([isinstance(value, expected_type) for expected_type in expected_types]), \
                    f"The expected types for the key: {key} are the following: [{','.join(map(str, expected_types))}]"

            self._sets_config = v

    def evaluate_set(self, input_set: Union[pd.DataFrame, EvalSet]) -> Tuple[EvalSet, Dict]:
        """
        This method serves as the primary interface of the class. Given a scored dataset, represented as a pandas
        DataFrame or as an EvalSet object, this function performs uplift analysis.

        Parameters
        ----------
        input_set: Union[pd.DataFrame, EvalSet]
            The dataset to be evaluated.

        Returns
        -------
        EvalSet
            The provided dataset after applying uplift analysis on it.
        Dict
            A summary of the analysis.
        """

        # in case the provided set is a dataframe, transform into an EvalSet object (according to some preset config)
        if isinstance(input_set, pd.DataFrame):
            input_set = EvalSet(df=input_set, **(self.sets_config or dict()))
        else:
            assert isinstance(input_set, EvalSet)

        input_set.set_problem_type()
        input_set.sort_and_rank()
        input_set.infer_subgroup_assignment()
        input_set.get_cumulative_counts()
        input_set.response_averaging()
        input_set.compute_uplift()
        input_set.compute_gain()
        input_set.compute_expected_response()
        input_set.compute_relative_lift()

        # mark the input_set as evaluated
        input_set.is_evaluated = True

        # the analyzed dataframe is returned as output, together with its summary
        return input_set, self.summarize_evaluation(input_set)

    def evaluate_multiple(self, scored_sets: Dict[str, Union[pd.DataFrame, EvalSet]]) -> Tuple[
        Dict[str, EvalSet], pd.DataFrame]:
        """
        This method utilizes the primary method ``evaluate_set()`` for evaluating multiple scored sets.

        Parameters
        ----------
        scored_sets: Dict[str, Union[pd.DataFrame, EvalSet]]
            The collection of scored datasets to be evaluated, represented by a dictionary indexed by the name of
            each method/experiment.

        Returns
        -------
        eval_res: Dict[str,EvalSet]
            A dictionary containing the evaluation result of each input dataset.
        comparison_df: pd.DataFrame
            A dataframe representing the comparison between the evaluated dataframes.
        """

        # outcome collectors for each element in the input dict
        eval_res, summaries = dict(), dict()

        # go over each scored set
        for name, set_to_evaluate in scored_sets.items():
            evaluation, summary = self.evaluate_set(scored_df=set_to_evaluate)
            eval_res[name] = evaluation
            summaries[name] = summary

        comparison_df = pd.DataFrame.from_dict(summaries, orient='index')

        return eval_res, comparison_df

    @staticmethod
    def summarize_evaluation(eval_set: EvalSet) -> Dict:
        """
        This function narrows down the evaluation of a dataset into a summary of the evaluated metrics.

        Parameters
        ----------
        eval_set: EvalSet
            The evaluated dataset.

        Returns
        -------
        Dict
            A summary of the evaluation results.
        """

        # what is the "sampling interval" between each sample/observation in the dataset.?
        # this variable will be used for computing the integral below the uplift curve
        dx = eval_set.get_quantile_interval()

        # start with computing summary metrics for the case where the action is binary
        summary = {
            'treated_AUUC': simps(eval_set.df['uplift_treated'].dropna().values, dx=dx),
            'treated_max_avg_resp': eval_set.df['expected_response_treated'].max(),
            'max_relative_lift_treated': eval_set.df['relative_lift_treated'].max(),
        }
        if eval_set.is_binary_response:  # gain is relevant for binary responses
            summary.update({'treated_max_gain': eval_set.df['gain_treated'].max()})

        # intersection metrics will be informative, and different from the general metrics, only when there is a
        # multitude of possible actions
        if eval_set.is_multiple_actions:

            summary.update({
                'intersect_AUUC': simps(eval_set.df['uplift_intersection'].dropna().values, dx=dx),
                'intersect_max_avg_resp': eval_set.df['expected_response_intersect'].max(),
                'max_relative_lift_intersect': eval_set.df['relative_lift_intersect'].max(),
            })
            if eval_set.is_binary_response:  # gain is relevant for binary responses
                summary.update({'intersect_max_gain': eval_set.df['gain_intersection'].max()})

        return summary

    def visualize(self, eval_res: Union[Dict[str, EvalSet], EvalSet], average: Optional[bool] = False):
        """
        This method provides a set of .

        Parameters
        ----------
        eval_res: Union[Dict[str, EvalSet], EvalSet]
            The collection of evaluated sets, or a single one, for which a set of descriptive charts will be displayed.
        average: Optional[bool]
            A boolean indicating whether to display the average performance as well. Relevant only if ``eval_res`` is
            a collection (``dict``)  of multiple ``EvalSet`` objects.
        """

        # averaging will be considered only in case of multiple sets
        num_sets = len(eval_res) if isinstance(eval_res, Dict) else 1
        average = average if num_sets > 1 else False
