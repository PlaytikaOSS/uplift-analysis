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

from typing import Dict, Union, List, Tuple
import pandas as pd
from scipy.integrate import simps


def is_multi_action(actions: pd.Series, neutral_indicator: Union[int, str]) -> bool:
    """
    This method checks whether the input dataframe is associated with a single action (except for the neutral
    action) or with a multitude of possible actions (multiple treatments).

    Parameters
    ----------
    actions: pd.Series
        A Pandas series representing a set of observed actions.
    neutral_indicator: Union[int,str]
        The action value associated with the neutral action.

    Returns
    -------
    bool
        A boolean indicating if the set is associated with multiple actions (True).

    """
    return not ((actions.nunique() == 2) and (neutral_indicator in actions.unique().tolist()))


def is_binary_response(responses: pd.Series) -> bool:
    """
    This method checks whether the input dataframe is associated with a response of binary type.

    Parameters
    ----------
    responses: pd.Series
        A Pandas series representing a set of observed responses.

    Returns
    -------
    bool
        A boolean indicating if the set is associated with a binary response (True).

    """
    return (responses.dtype == bool) or ((responses.nunique() == 2) and (all(responses.isin([0, 1]))))


class Evaluator:
    """
    Evaluator class is used for uplift evaluation of a given dataset, represented as a pandas Dataframe.
    The fields according to which the evaluation will be performed can be configured upon initialization,
    otherwise, these are set to the default value.
    Its primary interface is the method ``evaluate_set()``.

    Parameters
    ----------
    observed_action_field: str
        The name associated with the field / column containing the actual action assigned
        for each observation in the evaluated set.
    response_field: str
        The name associated with the field / column containing the observed response for each observation in the evaluated set.
    score_field: str
        The name associated with the field / column containing the output score for each observation in the
        evaluated set.
    proposed_action_field: str
        The name associated with the field / column containing the recommended action by the model,
        for each observation in the evaluated set.
    control_indicator: Union[int, str]
        The action value associated with the neutral action.
    """

    # the following dict specifies the fields required for performing the evaluation and their expected types
    conf_fields: Dict[str, List] = {
        'observed_action_field': [str],
        'response_field': [str],
        'score_field': [str],
        'proposed_action_field': [str],
        'control_indicator': [str, int]
    }

    def __init__(self,
                 observed_action_field: str = 'observed_action',
                 response_field: str = 'response',
                 score_field: str = 'score',
                 proposed_action_field: str = 'proposed_action',
                 control_indicator: Union[str, int] = 0
                 ):
        """
        See class docstring for further details regarding class functionality.
        """

        self.observed_action_field = observed_action_field
        self.response_field = response_field
        self.score_field = score_field
        self.proposed_action_field = proposed_action_field
        self.control_indicator = control_indicator

    def set_props(self, **kwargs) -> None:
        """
        A function for performing object configuration in terms of specifying the field names required for
        performing the uplift analysis.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments.

        """

        # if any of the provided arguments corresponds to any of the required fields,
        # configure the object correspondingly.
        for key, value in kwargs.items():
            if key in self.conf_fields:
                # the field name argument must be of the expected type
                expected_types = self.conf_fields[key]
                assert any([isinstance(value, expected_type) for expected_type in expected_types]), \
                    f"The expected types for the key: {key} are the following: [{','.join(map(str, expected_types))}]"
                # update the corresponding attribute
                setattr(self, key, value)

    def evaluate_set(self, scored_df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """
        This method serves as the primary interface of the class. Given a scored dataset, represented as a pandas
        DataFrame, this function performs uplift analysis, based on the configured field names.

        Parameters
        ----------
        scored_df: pd.DataFrame
            The dataframe to be evaluated.
        **kwargs:
            Arbitrary keyword arguments, which can be used for object configuration.

        Returns
        -------
        pd.DataFrame
            The provided dataframe after applying uplift analysis on it.
        Dict
            A summary of the analysis.
        """

        # use the input arguments for configure the evaluator, in terms of naming.
        self.set_props(**kwargs)

        scored_df = self._sort_and_rank(scored_df)
        scored_df = self._infer_subgroup_assignment(scored_df)
        scored_df = self._get_cumulative_counts(scored_df)
        scored_df = self._response_averaging(scored_df)
        scored_df = self._compute_uplift(scored_df)
        scored_df = self._compute_gain(scored_df)
        scored_df = self._compute_expected_response(scored_df)
        scored_df = self._compute_relative_lift(scored_df)

        # the analyzed dataframe is returned as output, together with its summary
        return scored_df, self.summarize_evaluation(scored_df)

    def evaluate_multiple(self, scored_dfs: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        This method utilizes the primary method ``evaluate_set()`` for evaluating multiple scored sets.

        Parameters
        ----------
        scored_dfs: Dict[str,pd.DataFrame]
            The collection of scored dataframes to be evaluated, represented by a dictionary indexed by the name of
            each method/experiment.

        Returns
        -------
        eval_res: Dict[str,pd.DataFrame]
            A dictionary containing the evaluation result of each input dataframe.
        comparison_df: pd.DataFrame
            A dataframe representing the comparison between the evaluated dataframes.
        """
        # outcome collectors for each element in the input dict
        eval_res, summaries = dict(), dict()

        # go over each scored set
        for name, set_df in scored_dfs.items():
            evaluation, summary = self.evaluate_set(scored_df=set_df)
            eval_res['name'] = evaluation
            summaries['name'] = summary

        comparison_df = pd.DataFrame.from_dict(summaries, orient='index')
        return eval_res, comparison_df

    def visualize(self, eval_res: Union[Dict[str, pd.DataFrame], pd.DataFrame], average: bool = False):
        """
        This method provides a set of .

        Parameters
        ----------
        scored_dfs: Dict[str,pd.DataFrame]
            The collection of scored dataframes to be evaluated, represented by a dictionary indexed by the name of
            each method/experiment.

        Returns
        -------
        eval_res: Dict[str,pd.DataFrame]
            A dictionary containing the evaluation result of each input dataframe.
        comparison_df: pd.DataFrame
            A dataframe representing the comparison between the evaluated dataframes.
        """
        pass

    def _sort_and_rank(self, in_df: pd.DataFrame) -> pd.DataFrame:
        """
        This method orders the dataset according to the provided score, in a descending manner, and assigns a
        new index, corresponding to the relative percentile of each observation in the dataset, in terms of score.

        Parameters
        ----------
        in_df: pd.DataFrame
            The dataframe to be sorted and ranked.

        Returns
        -------
        pd.DataFrame
            The sorted and ranked dataframe.

        """
        # order the dataset so that the observations with the highest score are put first
        in_df = in_df.sort_values(self.score_field, ascending=False)
        # index the dataframe according to the new order
        in_df = in_df.reset_index(drop=True)

        # normalize the index so that it ranges from (0,1]
        # this new "index" represents the corresponding percentile of each observation in terms of score
        in_df['normalized_index'] = (in_df.index + 1) / len(in_df)

        return in_df

    def _infer_subgroup_assignment(self, in_df: pd.DataFrame) -> pd.DataFrame:
        """
        This method specifies, for each observation in the dataset, what group was the observation assigned to -
        whether each observation was assigned an actual action (different from the neutral action), or the neutral one.

        In addition, this function marks observations in which the specific action observed/assigned matches/intersects
        with the recommended action according to the model - this is needed for cases in which there are multiple
        treatments.

        Parameters
        ----------
        in_df: pd.DataFrame
            The dataframe on which the grouping will be made.

        Returns
        -------
        pd.DataFrame
            The dataframe after assigning to subgroups.
        """
        in_df.loc[:, 'is_control'] = (in_df[self.observed_action_field] == self.control_indicator).astype(int)
        in_df.loc[:, 'is_treated'] = (in_df[self.observed_action_field] != self.control_indicator).astype(int)
        in_df.loc[:, 'is_intersect'] = (in_df[self.observed_action_field] == in_df[self.proposed_action_field]).astype(
            int)

        return in_df

    @staticmethod
    def _get_cumulative_counts(in_df: pd.DataFrame) -> pd.DataFrame:
        """
        This method performs summation in a cumulative manner of the observations in each group.
        This stage is performed after ranking and ordering the dataframe according to the provided score.

        Parameters
        ----------
        in_df: pd.DataFrame
            The dataframe on which the counting will be made.

        Returns
        -------
        pd.DataFrame
            The dataframe after computing the cumulative count.
        """
        in_df.loc[:, 'control_count'] = in_df['is_control'].cumsum()
        in_df.loc[:, 'treated_count'] = in_df['is_treated'].cumsum()
        in_df.loc[:, 'intersect_count'] = in_df['is_intersect'].cumsum()

        # compute a cumulative fraction of the treated observations along the ranked dataset
        in_df.loc[:, 'frac_of_overall_treated'] = in_df['treated_count'] / in_df['treated_count'].iloc[-1]

        return in_df

    def _response_averaging(self, in_df: pd.DataFrame) -> pd.DataFrame:
        """
        This method performs the averaging of the responses for each extent of exposure, and for each subgroup.

        Parameters
        ----------
        in_df: pd.DataFrame
            The dataframe on which the averaging will be made.

        Returns
        -------
        pd.DataFrame
            The dataframe after computing the average responses.
        """

        # Using the response/outcome series, and the columns indicating group assignment,
        # compute the cumulative sum of responses for each group
        sum_responses_control = (in_df[self.response_field].multiply(in_df['is_control'])).cumsum()
        sum_responses_intersect = (in_df[self.response_field].multiply(in_df['is_intersect'])).cumsum()
        sum_responses_treated = (in_df[self.response_field].multiply(in_df['is_treated'])).cumsum()

        # using the cumulative sums computed above, and the corresponding cumulative counts computed earlier,
        # average the response in a cumulative manner, for each group
        in_df.loc[:, 'above_response_control'] = sum_responses_control / in_df['control_count']
        in_df.loc[:, 'above_response_intersect'] = sum_responses_intersect / in_df['intersect_count']
        in_df.loc[:, 'above_response_treated'] = sum_responses_treated / in_df['treated_count']

        # specifically, for the observations within the control group (neutral action, or no action at all),
        # we also compute the complementary average response.
        # for that we need to know what is the overall sum of responses for that group:
        total_responses_control = sum_responses_control.iloc[-1]
        # and average the response - dividing the residual sum of responses (numerator)
        # by the residual count (denominator)
        in_df.loc[:, 'below_response_control'] = (total_responses_control - sum_responses_control) / (
                in_df['control_count'].iloc[-1] - in_df['control_count'])

        return in_df

    @staticmethod
    def _compute_uplift(in_df: pd.DataFrame) -> pd.DataFrame:
        """
        This function computes the uplift, as the difference between the average response of the treated group, and the
        not-treated (control) group, per percentile.
        The same is done for the intersection group, in which all the recommended actions go hand in hand with the
        actual actions assigned/observed - relevant for multiple treatments/actions scenario.

        Parameters
        ----------
        in_df: pd.DataFrame
            The dataframe containing the running average responses per group.

        Returns
        -------
        pd.DataFrame
            The dataframe along with the corresponding uplift calculation.
        """
        in_df.loc[:, 'uplift_intersection'] = in_df['above_response_intersect'] - in_df['above_response_control']
        in_df.loc[:, 'uplift_treated'] = in_df['above_response_treated'] - in_df['above_response_control']

        return in_df

    @staticmethod
    def _compute_gain(in_df: pd.DataFrame) -> pd.DataFrame:
        """
        This method computes the gain as the multiplication of the uplift curve with cumulative count of
        control/non-treated cases. The gain signal enables the perception of the trade-off between the quantity
        of exposed cases and the reduction in uplift. Interpreting the gain signal is harder when the response is not
        binary (real value response).

        Parameters
        ----------
        in_df: pd.DataFrame
            The dataframe containing the computed uplift measures.

        Returns
        -------
        pd.DataFrame
            The dataframe along with the corresponding gain calculation.

        """
        in_df['gain_intersection'] = in_df['uplift_intersection'].multiply(in_df['control_count'])
        in_df['gain_treated'] = in_df['uplift_treated'].multiply(in_df['control_count'])

        return in_df

    @staticmethod
    def _compute_expected_response(in_df: pd.DataFrame) -> pd.DataFrame:
        """
        This method computes the expected response per group, by weighting the average response of the exposed group,
        and the observed average response of the control group on the complementary part of the dataset,
        taking into account the relevant percentiles.

        Parameters
        ----------
        in_df: pd.DataFrame
            The dataframe containing the computed average responses.

        Returns
        -------
        pd.DataFrame
            The dataframe along with the corresponding expected response calculation.

        """
        in_df.loc[:, 'expected_response_intersect'] = in_df['normalized_index'] * in_df['above_response_intersect'] + (
                1 - in_df['normalized_index']) * in_df['below_response_control']
        in_df.loc[:, 'expected_response_treated'] = in_df['normalized_index'] * in_df['above_response_treated'] + (
                1 - in_df['normalized_index']) * in_df['below_response_control']

        return in_df

    def _compute_relative_lift(self, in_df: pd.DataFrame) -> pd.DataFrame:
        """
        This method computes the difference between the expected response computed on an earlier stage,
         and the average response of the control group overall.

         Parameters
        ----------
        in_df: pd.DataFrame
            The dataframe containing the computed expected responses.

        Returns
        -------
        pd.DataFrame
            The dataframe along with the corresponding relative lifts.

        """
        # use the average outcome of the control group as the baseline
        base_response_rate = in_df.loc[in_df['is_control'] == 1, self.response_field].mean()

        # subtract it from the expected response
        in_df.loc[:, 'relative_lift_intersect'] = in_df['expected_response_intersect'] - base_response_rate
        in_df.loc[:, 'relative_lift_treated'] = in_df['expected_response_treated'] - base_response_rate

        return in_df

    def summarize_evaluation(self, eval_df: pd.DataFrame) -> Dict:
        """
        This function narrows down the evaluation of a dataset into a summary of the evaluated metrics.

        Parameters
        ----------
        eval_df: pd.DataFrame
            The dataframe post the evaluation procedure.

        Returns
        -------
        Dict
            A summary of the evaluation results.
        """

        # what is the "sampling interval" between each sample/observation in the dataset.?
        # this variable will be used for computing the integral below the uplift curve
        dx = eval_df['normalized_index'].diff().iloc[-1]

        # detect whether the dataframe is associated with a binary response and/or multiple actions
        multi_action_bool = self.is_multi_action(eval_df)
        binary_response_bool = self.is_binary_response(eval_df)

        # start with computing summary metrics for the case where the action is binary
        summary = {
            'treated_AUUC': simps(eval_df['uplift_treated'].dropna().values, dx=dx),
            'treated_max_avg_resp': eval_df['expected_response_treated'].max(),
            'max_relative_lift_treated': eval_df['relative_lift_treated'].max(),
        }
        if binary_response_bool:  # gain is relevant for binary responses
            summary.update({'treated_max_gain': eval_df['gain_treated'].max()})

        # intersection metrics will be informative, and different from the general metrics, only when there is a
        # multitude of possible actions
        if multi_action_bool:

            summary.update({
                'intersect_AUUC': simps(eval_df['uplift_intersection'].dropna().values, dx=dx),
                'intersect_max_avg_resp': eval_df['expected_response_intersect'].max(),
                'max_relative_lift_intersect': eval_df['relative_lift_intersect'].max(),
            })
            if binary_response_bool:  # gain is relevant for binary responses
                summary.update({'intersect_max_gain': eval_df['gain_intersection'].max()})

        return summary

    def is_multi_action(self, df: pd.DataFrame) -> bool:
        """
        This method checks whether the input dataframe is associated with a single action (except for the neutral
        action) or with a multitude of possible actions (multiple treatments).

        Parameters
        ----------
        df: pd.DataFrame
            A dataframe going through the evaluation procedure.

        Returns
        -------
        bool
            A boolean indicating if the dataframe is associated with multiple actions (True).

        """

        return is_multi_action(actions=df[self.observed_action_field], neutral_indicator=self.control_indicator)

    def is_binary_response(self, df: pd.DataFrame) -> bool:
        """
        This method checks whether the input dataframe is associated with a response of binary type.

        Parameters
        ----------
        df: pd.DataFrame
            A dataframe going through the evaluation procedure.

        Returns
        -------
        bool
            A boolean indicating if the dataframe is associated with a binary response (True).

        """

        return is_binary_response(df[self.response_field])
