# -*- coding: utf-8 -*-
"""
This module implements the primary ``dataclass`` required for performing uplift analysis - ``EvalSet``.
"""

from typing import Union, Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd
from . import utils


@dataclass
class EvalSet:
    df: pd.DataFrame
    name: Optional[Union[str, None]] = None
    observed_action_field: Optional[str] = 'observed_action'
    response_field: Optional[str] = 'response'
    score_field: Optional[str] = 'score'
    proposed_action_field: Optional[str] = 'proposed_action'
    control_indicator: Optional[Union[str, int]] = 0
    _is_evaluated: bool = False
    _is_binary_response: bool = False
    _is_multiple_actions: bool = False

    """
    A dataclass to represent a dataset object, going through the uplift evaluation procedure, together with the
    corresponding meta-data - e.g. specification of the relevant field names.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe representing the dataset.
    name: Optional[Union[str, None]]
        Identification of the specific dataset contained in this object.
    observed_action_field: Optional[str]
        The name associated with the field / column containing the actual action assigned
        for each observation in the evaluated set.
    response_field: Optional[str]
        The name associated with the field / column containing the observed response for each observation in the 
        evaluated set.
    score_field: Optional[str]
        The name associated with the field / column containing the output score for each observation in the
        evaluated set.
    proposed_action_field: Optional[str]
        The name associated with the field / column containing the recommended action by the model,
        for each observation in the evaluated set.
    control_indicator: Optional[Union[str, int]]
        The action value associated with the neutral action.
    """

    @classmethod
    def conf_fields(cls) -> Dict[str, List]:
        """
        the returned dict specifies the fields required for performing the evaluation and their expected types
        """
        return {
            'observed_action_field': [str],
            'response_field': [str],
            'score_field': [str],
            'proposed_action_field': [str],
            'control_indicator': [str, int]
        }

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
        fields_conf = self.conf_fields()
        for key, value in kwargs.items():
            if key in fields_conf:
                # the field name argument must be of the expected type
                expected_types = fields_conf[key]
                assert any([isinstance(value, expected_type) for expected_type in expected_types]), \
                    f"The expected types for the key: {key} are the following: [{','.join(map(str, expected_types))}]"
                # update the corresponding attribute
                setattr(self, key, value)

    @property
    def is_evaluated(self) -> bool:
        """
        This property will be assigned by ``evaluation.Evaluator`` when the evaluation procedure is completed.
        """
        return self._is_evaluated

    @is_evaluated.setter
    def is_evaluated(self, v: bool) -> None:
        self._is_evaluated = v

    @property
    def is_binary_response(self) -> bool:
        """
        This property will indicate whether the dataset is associated with a binary response.
        """
        return self._is_binary_response

    @is_binary_response.setter
    def is_binary_response(self, v: bool) -> None:
        self._is_binary_response = v

    @property
    def is_multiple_actions(self) -> bool:
        """
        This property will indicate whether the dataset is associated with multiple actions.
        """
        return self._is_multiple_actions

    @is_multiple_actions.setter
    def is_multiple_actions(self, v: bool) -> None:
        self._is_multiple_actions = v

    def set_problem_type(self):
        """
        This method performs two checks:

        -   is the dataset associated with a single action (except for the neutral action), or with a multitude of
            possible actions (multiple treatments).
        -   is the dataset associated with a response of binary type.

        """

        self.is_multiple_actions = utils.is_multi_action(actions=self.df[self.observed_action_field],
                                                         neutral_indicator=self.control_indicator)

        self.is_binary_response = utils.is_binary_response(responses=self.df[self.response_field])

    def sort_and_rank(self) -> None:
        """
        This method orders the dataset according to the provided score, in a descending manner, and assigns a
        new index, corresponding to the relative percentile of each observation in the dataset, in terms of score.
        """
        # order the dataset so that the observations with the highest score are put first
        self.df = self.df.sort_values(self.score_field, ascending=False)
        # index the dataframe according to the new order
        self.df = self.df.reset_index(drop=True)

        # normalize the index so that it ranges from (0,1]
        # this new "index" represents the corresponding percentile of each observation in terms of score
        self.df['normalized_index'] = (self.df.index + 1) / len(self.df)

    def infer_subgroup_assignment(self) -> None:
        """
        This method specifies, for each observation in the dataset, what group was the observation assigned to -
        whether each observation was assigned an actual action (different from the neutral action), or the neutral one.

        In addition, this function marks observations in which the specific action observed/assigned matches/intersects
        with the recommended action according to the model - this is needed for cases in which there are multiple
        treatments.
        """
        self.df.loc[:, 'is_control'] = (self.df[self.observed_action_field] == self.control_indicator).astype(int)
        self.df.loc[:, 'is_treated'] = (self.df[self.observed_action_field] != self.control_indicator).astype(int)
        self.df.loc[:, 'is_intersect'] = (
                self.df[self.observed_action_field] == self.df[self.proposed_action_field]).astype(int)
        self.df.loc[:, 'is_unrealized'] = (
                self.df[self.observed_action_field] != self.df[self.proposed_action_field]).astype(int)

    def get_cumulative_counts(self) -> None:
        """
        This method performs summation in a cumulative manner of the observations in each group.
        This stage is performed after ranking and ordering the dataframe according to the provided score.
        """
        self.df.loc[:, 'control_count'] = self.df['is_control'].cumsum()
        self.df.loc[:, 'treated_count'] = self.df['is_treated'].cumsum()
        self.df.loc[:, 'intersect_count'] = self.df['is_intersect'].cumsum()
        self.df.loc[:, 'unrealized_count'] = self.df['is_unrealized'].cumsum()

        # compute a cumulative fraction of the treated observations along the ranked dataset
        self.df.loc[:, 'frac_of_overall_treated'] = self.df['treated_count'] / self.df['treated_count'].iloc[-1]

    def response_averaging(self) -> None:
        """
        This method performs the averaging of the responses for each extent of exposure, and for each subgroup.
        """

        # Using the response/outcome series, and the columns indicating group assignment,
        # compute the cumulative sum of responses for each group
        sum_responses_control = (self.df[self.response_field].multiply(self.df['is_control'])).cumsum()
        sum_responses_intersect = (self.df[self.response_field].multiply(self.df['is_intersect'])).cumsum()
        sum_responses_treated = (self.df[self.response_field].multiply(self.df['is_treated'])).cumsum()
        sum_responses_unrealized = (self.df[self.response_field].multiply(self.df['is_unrealized'])).cumsum()

        # using the cumulative sums computed above, and the corresponding cumulative counts computed earlier,
        # average the response in a cumulative manner, for each group
        self.df.loc[:, 'above_response_control'] = sum_responses_control / self.df['control_count']
        self.df.loc[:, 'above_response_intersect'] = sum_responses_intersect / self.df['intersect_count']
        self.df.loc[:, 'above_response_treated'] = sum_responses_treated / self.df['treated_count']
        self.df.loc[:, 'above_response_unrealized'] = sum_responses_unrealized / self.df['unrealized_count']

        # we also compute the complementary average response.
        # for that we need to know what is the overall sum of responses for that group:
        total_responses_control = sum_responses_control.iloc[-1]
        total_responses_treated = sum_responses_treated.iloc[-1]
        # and average the response - dividing the residual sum of responses (numerator)
        # by the residual count (denominator)
        self.df.loc[:, 'below_response_control'] = (total_responses_control - sum_responses_control) / (
                self.df['control_count'].iloc[-1] - self.df['control_count'])
        self.df.loc[:, 'below_response_treated'] = (total_responses_treated - sum_responses_treated) / (
                self.df['treated_count'].iloc[-1] - self.df['treated_count'])

    def compute_uplift(self) -> None:
        """
        This function computes the uplift, as the difference between the average response of the treated group, and the
        not-treated (control) group, per percentile.
        The same is done for the intersection group, in which all the recommended actions go hand in hand with the
        actual actions assigned/observed - relevant for multiple treatments/actions scenario.
        """
        self.df.loc[:, 'uplift_intersection'] = self.df['above_response_intersect'] - self.df['above_response_control']
        self.df.loc[:, 'uplift_treated'] = self.df['above_response_treated'] - self.df['above_response_control']
        self.df.loc[:, 'uplift_against_unrealized'] = self.df['above_response_intersect'] - self.df[
            'above_response_unrealized']

    def compute_gain(self) -> None:
        """
        This method computes the gain as the multiplication of the uplift curve with cumulative count of
        control/non-treated cases. The gain signal enables the perception of the trade-off between the quantity
        of exposed cases and the reduction in uplift. Interpreting the gain signal is harder when the response is not
        binary (real value response).
        """
        self.df['gain_intersection'] = self.df['uplift_intersection'].multiply(self.df['control_count'])
        self.df['gain_treated'] = self.df['uplift_treated'].multiply(self.df['control_count'])

    def compute_expected_response(self) -> None:
        """
        This method computes the expected response per group, by weighting the average response of the exposed group,
        and the observed average response of the control group on the complementary part of the dataset,
        taking into account the relevant percentiles.
        """
        self.df.loc[:, 'expected_response_intersect'] = self.df['normalized_index'] * self.df[
            'above_response_intersect'] + (1 - self.df['normalized_index']) * self.df['below_response_control']
        self.df.loc[:, 'expected_response_treated'] = self.df['normalized_index'] * self.df[
            'above_response_treated'] + (1 - self.df['normalized_index']) * self.df['below_response_control']

    def compute_relative_lift(self) -> None:
        """
        This method computes the difference between the expected response computed on an earlier stage,
        and the average response of the control group overall.
        """
        # use the average outcome of the control group as the baseline
        base_response_rate = self.df.loc[self.df['is_control'] == 1, self.response_field].mean()

        # subtract it from the expected response
        self.df.loc[:, 'relative_lift_intersect'] = self.df['expected_response_intersect'] - base_response_rate
        self.df.loc[:, 'relative_lift_treated'] = self.df['expected_response_treated'] - base_response_rate

    def get_quantile_interval(self) -> float:
        """
        This method returns the "sampling interval" between each sample/observation in the dataset, in terms of
        quantiles. This variable can be used, for example, for computing the integral below the uplift curve.

        Returns
        -------
        float
            The quantile interval.

        """
        return self.df['normalized_index'].diff().iloc[-1]
