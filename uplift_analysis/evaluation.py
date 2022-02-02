# -*- coding: utf-8 -*-
"""
This module implements an uplift evaluation utility wrapped as a class named Evaluator.
After specifying the relevant field names on the corresponding dataset (represented as a pandas dataframe or as a
``data.EvalSet`` object), the provided dataset will be analyzed, in terms of uplift.

The evaluation takes into account cases of multiple treatments; In case the actions in the evaluated set are binary
(single treatment) the evaluation metrics will end up being identical for the multiple-actions scenario and the binary
one.

Notes:
    - ``Evaluator`` also supports use-cases with multiple treatments.
"""

from typing import Dict, Union, List, Tuple, Optional, Callable
import copy
import itertools
import numpy as np
import pandas as pd
from scipy.integrate import simps
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
import seaborn as sns
from .data import EvalSet
from . import visualization
from . import utils


class Evaluator:
    """
    Evaluator class is used for uplift evaluation of a given dataset, represented as a pandas Dataframe, or as
    an EvalSet object.

    Parameters
    ----------
    sets_config: Optional[Union[Dict, None]]
        A set of configuration parameters, to be used for creating EvalSet objects of pandas Dataframes, if required.

    """

    _sets_config: Union[Dict, None] = None

    def __init__(self, sets_config: Optional[Union[Dict, None]] = None):
        self.sets_config = sets_config
        self.min_quantile = visualization.BASE_MIN_QUANTILE

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
        This method serves as a primary interface of the class. Given a scored dataset, represented as a pandas
        DataFrame or as an ``EvalSet`` object, this function performs uplift analysis.

        Parameters
        ----------
        input_set: Union[pd.DataFrame, EvalSet]
            The dataset to be evaluated. If provided as dataframe, it will be transformed into a corresponding
            ``EvalSet`` object, according to the ``sets_config`` property.

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

        # apply the required computations as part of the evaluation
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

        # the analyzed set is returned as output, together with its summary
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
            evaluation, summary = self.evaluate_set(input_set=set_to_evaluate)
            # store the resulting ``EvalSet`` object and its summary
            eval_res[name] = evaluation
            summaries[name] = summary

        # aggregate summaries
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
        # this variable will be used for computing the integral under curves
        dx = eval_set.get_quantile_interval()

        # compute summary aggregated metrics for the case
        summary = {
            'intersect_AUUC': simps(eval_set.df['uplift_intersection'].dropna().values, dx=dx),
            'intersect_max_avg_resp': eval_set.df['expected_response_intersect'].max(),
            'max_relative_lift_intersect': eval_set.df['relative_lift_intersect'].max(),
            'intersect_max_gain': eval_set.df['gain_intersection'].max()
        }

        # When there is a multitude of possible actions, the method will compute aggregate performance
        # metrics for the case where the action space is considered as binary (binary treatment).
        # In other cases, these metrics will be identical to the ones computed earlier.
        if eval_set.is_multiple_actions:
            summary.update({

                'treated_AUUC': simps(eval_set.df['uplift_treated'].dropna().values, dx=dx),
                'treated_max_avg_resp': eval_set.df['expected_response_treated'].max(),
                'max_relative_lift_treated': eval_set.df['relative_lift_treated'].max(),
                'treated_max_gain': eval_set.df['gain_treated'].max()
            })

        return summary

    @staticmethod
    def integrate(x: Union[np.ndarray, pd.Series], dx: float) -> float:
        """
        A static method for performing integration of a given signal based on Simpson's rule.

        Parameters
        ----------
        x: Union[np.ndarray, pd.Series]
            The signal to integrate.
        dx: float
            The sampling interval according to which the signal is sampled.

        Returns
        -------
        float
            The integration result.

        """
        if isinstance(x, np.ndarray):
            # get a corresponding pandas series
            x = pd.Series(x)

        if x.isna().mean() < 1.0:  # if not all the values are nan
            # mark all non-finite values as nan
            x = x.replace([-np.inf, np.inf], np.nan)
            # smooth them out using the last available value
            x = x.fillna(method='ffill')
            # get the assoicated numpy array
            x = x.values
            # integrate (ignoring the beginning in case, NaNs are still present)
            return simps(x[np.logical_and(~np.isnan(x), np.isfinite(x))], dx=dx)
        else:  # all the values are nan
            return np.nan

    @staticmethod
    def get_max(eval_set: EvalSet, metric: str):
        """
        A static method for retriveing the coordinates of the maximal value of a specific input signal, and the
        score associated with the maximal value.

        Parameters
        ----------
        eval_set: EvalSet
            The object containing the signal in which the maximal value will be located.
        metric:
            The name of the signal. Generally, corresponds to a column on the dataframe hosted in the
            ``EvalSet`` object.

        Returns
        -------
        Dict
            a dictionary containing a key 'point', in which the coordinates of the maximal value are stored,
            and `value`, which specifies the score value corresponding to the maximal point.
        """

        # locate max
        idxmax = eval_set.df[metric].idxmax()
        return {
            'point': [  # coordinates of the maximal point along the curve / signal
                eval_set.df['normalized_index'][idxmax],
                eval_set.df[metric].max()
            ],
            # the score associated with the maximal point
            'value': eval_set.df[eval_set.score_field][idxmax]
        }

    def visualization_methods(self) -> Dict[str, Callable]:
        """
        This method will list and index all the visualization methods this class has to offer.
        Only the methods listed as part of this function, will be taken into account, when calling
        ``Evaluator.visualize()``.

        Returns
        -------
        Dict[str,Callable]
            A dictionary specifying the keyword associated with each support visualization method.
        """
        return {
            'uplift': self.display_uplift_curve,
            'fractional_lift': self.display_fractional_lift_curve,
            'gain': self.display_gain,
            'avg_response': self.display_avg_response,
            'acceptance_region': self.display_acceptance_region_stats,
            'rejection_region': self.display_rejection_region_stats,
            'agreements': self.display_agreement_stats,
            'score_distribution': self.display_score_distribution,
        }

    def visualize(self,
                  eval_res: Union[Dict[str, EvalSet], EvalSet],
                  average: Optional[bool] = False,
                  title_suffix: Optional[str] = '',
                  show_random: Optional[bool] = False,
                  num_random_rep: Optional[int] = 1,
                  specify: Optional[Union[List[str], None]] = None):
        """
        This method provides and generates a set of charts for the description of the provided evaluation results.

        As opposed to calls in which ``eval_res`` holds multiple ``EvalSet`` objects, in cases where the provided
        ``eval_res`` will contain only a single ``data.EvalSet`` object, the generated chart will contain more
        interesting visualization that provide information enrichment (distinct additions for each chart).

        On most of the created charts, the x-axis corresponds to upper quantiles of the score distribution - i.e. after
        the scores on a given ``EvalSet`` object are sorted in a descending manner, as part of the evaluation procedure,
        we refer to a ``normalized_index`` which is simply a mapping between the scores and the range of (0,1], so that
        the highest score is mapped to nearest to zero as possible, and the lowest score is mapped to one. With this,
        the ``Exposed Fraction``, as noted in the charts labels, refers to calculations that include the observations
        up to (or from) a specific value of the ``normalized_index``. For example, on the **Uplift Curve** chart, the
        vertical ``x=0.4`` corresponds to calculated estimations taking into account model recommendations only of the
        top 40% precent scores.

        Parameters
        ----------
        eval_res: Union[Dict[str, EvalSet], EvalSet]
            The collection of evaluated sets, or a single one, for which a set of descriptive charts will be displayed.
        average: Optional[bool]
            A boolean indicating whether to display the average performance as well. Relevant only if ``eval_res`` is
            a collection (``dict``)  of multiple ``EvalSet`` objects.
        title_suffix: Optional[str]
            Optional string to add to the title of the charts.
        show_random: Optional[bool]
            If true, will create a randomly-scored corresponding set and display it as a benchmark. Relevant only when
            eval_res contains a single EvalSet.
        num_random_rep: Optional[int]
            The number of randomly-scores sets to average. Used only when ``show_random`` is set to ``True``.
        specify: Optional[Union[List[str]], None]
            A list for specifying only a partial list of charts to display. Every element in the list must be included
            in ``self.visualization_methods()``.
        """

        # verify that the EvalSet objects were evaluated
        if isinstance(eval_res, EvalSet):
            assert eval_res.is_evaluated, "The provided EvalSet object must be evaluated first."
        elif isinstance(eval_res, dict):
            assert all([(isinstance(eval_set, EvalSet) and eval_set.is_evaluated) \
                        for _, eval_set in eval_res.items()]), \
                "Every element in the provided dict must be of type EvalSet, and be evaluated as well."
        else:
            raise TypeError("The provided input 'eval_res' must be either an EvalSet or a dict of such.")

        # averaging will be considered only in case of multiple sets
        average, num_sets = visualization.should_average(eval_res, average)
        assert num_sets == 1 or isinstance(eval_res, dict), \
            "When the number of provided EvalSet objects is larger than 1, they must be provided as dict"

        # make sure all the specified charts, if provided, are listed
        if specify:
            assert all([elem in self.visualization_methods() for elem in specify]), \
                "Every element provided as part of `specify` argument must be listed under `visualization_methods()`"

        # generate random sets for benchmarking (applies only for a single input set)
        if show_random and num_sets == 1:
            random_res = self._create_random_sets(eval_res, num_random_rep)
        else:
            random_res = None

        # call each visualization method
        viz_methods = self.visualization_methods()
        for chart_name, func in viz_methods.items():
            # if ``specify`` was provided, call only corresponding functions
            if (specify is None) or (chart_name in specify):
                func(eval_res=eval_res,
                     num_sets=num_sets,
                     average=average,
                     title_suffix=title_suffix,
                     random_sets=random_res)

    def eval_and_show(self, data=Union[pd.DataFrame, EvalSet, Dict[str, Union[pd.DataFrame, EvalSet]]], **kwargs) -> \
            Tuple[Union[EvalSet, Dict[str, EvalSet]], Union[Dict, pd.DataFrame]]:
        """
        This method allows to use a single call for performing both the evaluation and the visualization of single /
        multiple scored datasets.

        Parameters
        ----------
        data: Union[pd.DataFrame, EvalSet, Dict[str, Union[pd.DataFrame, EvalSet]]]
            The scored input data, whether it is a single or multiple datasets.
        kwargs:
            Arbitrary keyword arguments, which will be passed to ``Evaluator.visualize()``.

        Returns
        -------
        eval_res: Union[EvalSet, Dict[str,EvalSet]]
            The evaluation result, whether it is a single ``EvalSet``, or a dictionary of such (in case the input `
            `data`` contained several datasets).
        summary: Union[Dict, pd.DataFrame]
            A dictionary containing the summary of the analysis, if only a single input set was provided, or a
            dataframe representing the comparison between the evaluated dataframes, in the case of multiple input sets.
        """

        if isinstance(data, dict):
            eval_res, summary = self.evaluate_multiple(data)
        else:
            eval_res, summary = self.evaluate_set(data)

        self.visualize(eval_res=eval_res, **kwargs)

        return eval_res, summary

    def _create_random_sets(self, eval_res: Union[EvalSet, Dict[str, EvalSet]], num_random_rep: int) -> List[EvalSet]:
        """
        This method uses a given ``EvalSet`` object for creating a list of new ``EvalSet`` objects, in which the scores
        are randomized, as well as the recommended treatments. These sets can be used for benchmarking the performance
        of the model evaluated using the input ``eval_res``.

        Parameters
        ----------
        eval_res: Union[EvalSet, Dict[str, EvalSet]]
            The input dataset to use for generating randomly scored datasets.
        num_random_rep: int
            The number of randomly scored datasets to create.

        Returns
        -------
        random_sets: List[EvalSet]
            A list of ``EvalSet`` objects with random scores, and random action recommendations.

        """
        # get the ``EvalSet`` object whether it was provided as input, or whether it was wrapped as part of a dict
        if isinstance(eval_res, dict):
            _, eval_set = next(iter(eval_res))
        else:
            eval_set = eval_res

        # how many observations does the input set contain
        num_samples = len(eval_set.df)
        # list possible actions - our inventory. Discarding the neutral action
        action_set = list(
            set(eval_set.df[eval_set.observed_action_field].unique().tolist()) - set([eval_set.control_indicator]))

        random_sets: List[EvalSet] = []  # will contain the generated sets
        for _ in range(num_random_rep):
            rnd_set = copy.deepcopy(eval_set)

            # randomize score
            rnd_set.df[rnd_set.score_field] = np.random.standard_normal(size=num_samples)
            # randomize action
            rnd_set.df[rnd_set.proposed_action_field] = np.random.choice(action_set, size=(num_samples,), replace=True)

            evaluation, _ = self.evaluate_set(input_set=rnd_set)
            random_sets.append(rnd_set)

        return random_sets

    def display_score_distribution(self, eval_res: Union[Dict[str, EvalSet], EvalSet],
                                   num_sets: Optional[Union[None, int]] = None,
                                   title_suffix: Optional[str] = '',
                                   **kwargs) -> Axes:
        """
        This method provides a visualization of the score density distribution in the input evaluated dataset(s).
        According to ``Evaluator.visualization_methods()``, this method fits the keyword ``score_distribution``.

        If the input ``eval_res`` contains a single ``EvalSet``, the overall score density, will be plotted, alongisde
        with the densities of the scores for the treated group, and the untreated group separately.
        If the single ``EvalSet`` object is associated with ``multiple_actions``, then the score density for each
        observed treatment will also be separately plotted.

        Parameters
        ----------
        eval_res: Union[Dict[str, EvalSet], EvalSet]
            The input evaluated dataset(s).
        num_sets: Optional[Union[None, int]]
            The number of ``EvalSet`` objects in ``eval_res``. If not provided, inferred independently.
        title_suffix: Optional[str]
            Optional string to add to the title of the chart.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        ax: Axes
            The axes on which the visualization was created, for further manipulation if required.
        """
        if not num_sets:
            _, num_sets = visualization.should_average(eval_res)

        fig, ax = plt.subplots()
        palette = itertools.cycle(sns.color_palette())

        if num_sets > 1:  # for multiple sets
            for (name, eval_set) in eval_res.items():
                s = eval_set.df[eval_set.score_field]
                sns.histplot(s, ax=ax, kde=True, bins=visualization.get_bin_quantity(s), stat='density', alpha=0.3,
                             label=name, color=next(palette))

            ax.set_xlabel('Uplift Score')

        else:  # for a single set
            if isinstance(eval_res, dict):  # must be a single entry dict
                name, eval_set = next(iter(eval_res))
            else:  # has to be an EvalSet
                name, eval_set = '', eval_res

            # overall scores
            s = eval_set.df[eval_set.score_field]
            sns.histplot(s, ax=ax, kde=True, bins=visualization.get_bin_quantity(s), stat='density', alpha=0.3,
                         label='Overall', color=next(palette))
            # scores of the treated group
            s = eval_set.df.loc[
                eval_set.df[eval_set.observed_action_field] != eval_set.control_indicator, eval_set.score_field]
            sns.histplot(s, ax=ax, kde=True, bins=visualization.get_bin_quantity(s), stat='density', alpha=0.3,
                         label='Treated', color=next(palette))
            # scores of the untreated group
            s = eval_set.df.loc[
                eval_set.df[eval_set.observed_action_field] == eval_set.control_indicator, eval_set.score_field]
            sns.histplot(s, ax=ax, kde=True, bins=visualization.get_bin_quantity(s), stat='density', alpha=0.3,
                         label='Untreated', color=next(palette))

            # in case there are multiple actions
            # display the density of the scores for each action
            if eval_set.is_multiple_actions:
                for grp, sub_df in eval_set.df.groupby(eval_set.observed_action_field):
                    s = sub_df[eval_set.score_field]
                    sns.histplot(s, ax=ax, kde=True, bins=visualization.get_bin_quantity(s), stat='density', alpha=0.3,
                                 label=f'Treatment: {grp}', color=next(palette))

            ax.set_xlabel(eval_set.score_field)

        ax.grid(True)
        ax.legend(fancybox=True, shadow=True)
        ax.set_title(f"Score Distribution\n{title_suffix}")

        return ax

    def display_uplift_curve(self,
                             eval_res: Union[Dict[str, EvalSet], EvalSet],
                             num_sets: Optional[Union[None, int]] = None,
                             average: Optional[bool] = False,
                             title_suffix: Optional[str] = '',
                             random_sets: Optional[Union[List[EvalSet], None]] = None,
                             **kwargs
                             ) -> Axes:
        """
        This method provides a visualization of the uplift curve(s) implied by the input evaluated dataset(s).
        According to ``Evaluator.visualization_methods()``, this method fits the keyword ``uplift``.

        The uplift curve describes, for each upper quantile, the difference in average response between the group of
        observations which were treated in accordance with the model recommendations, and a reference group of
        observations (e.g. observations which were not treated at all). The difference is computed for each upper
        quantile ``q``, i.e. we take the observations from the upper *q*-th quantile of the dataset, split into the
        relevant groups, and compute the difference in average response between these groups. Combining the
        calculations for all the quantiles, yields the uplift curve.
        Hence, the x-axis corresponds to upper quantiles of the score distribution - i.e. after
        the scores on a given ``EvalSet`` object are sorted in a descending manner, as part of the evaluation procedure,
        we refer to a ``normalized_index`` which is simply a mapping between the scores and the range of (0,1], so that
        the highest score is mapped to nearest to zero as possible, and the lowest score is mapped to one. With this,
        the ``Exposed Fraction``, as noted in the charts labels, refers to calculations that include the observations
        up to (or from) a specific value of the ``normalized_index``.

        Uplift curves will be labeled together with their corresponding Area Under Uplift Curve (AUUC), which is the
        result of the integral under the curve.

        In case the provided ``eval_res`` contains multiple ``EvalSet``\s, the visualization will contain, for each of
        the ``EvalSet`` objects, the curve of the uplift between the group in which the recommendations of the model
        intersect with the observed actions, and the group of observations associated with the neutral action
        (untreated). Each curve will be labeled according to its corresponding key on the provided dictionary.
        If ``average=True``, the chart will also contain the average performance, computed across the multiple
        ``EvalSet``\s provided (labeled as **Avg**).

        In case the provided ``eval_res`` contains a single ``EvalSet`` the visualization will contain the following
        curves:

        -   **Intersection Uplift** - uplift between the group in which the recommendations of the model
            intersect with the observed actions, and the group of observations associated with the neutral action
            (untreated).
        -   **Random** - the average uplift curve (just like the **Intersection Uplift** curve) calculated across
            the ``EvalSet`` objects contained in ``random_sets``, if provided.
        -   **Treated Uplift** - (relevant in case the ``EvalSet`` is associated with multiple actions) uplift between
            the treated group, disregarding the identity of the exact treatment, and the group of observations
            associated with the neutral action (untreated).
        -   **Realized Vs Unrealized Uplift** - (relevant in case the ``EvalSet`` is associated with multiple actions)
            uplift between the group in which the recommendations of the model intersect with the observed actions
            (*realized*), and the group of observations in which the recommendations do not intersect with the observed
            actions (*unrealized*). Here, observations in the reference group can be associated with some non-neutral
            action, but just not the one recommended by the model.
        -   **UnexposedResponseDiff** - the difference in average response in the complement region of the dataset, i.e.
            for each upper quantile ``q``, this curve takes into account the average responses of the treated group and
            the untreated group, in the lower ``(1-q)`` quantiles of the score, and subtracts between them. A
            positive-valued curve implies that the average response of the untreated group, is higher than that of the
            treated group, for some lower quantile score of the dataset.
        -   In addition, the black dashed line, corresponding to the right y-axis, provides information about the
            fraction of the treated subgroup in general, that is located in the upper quantile ``q``. If this line is
            linear, it implies that the treated observations and the untreated observations are distributed similarly,
            in terms of score.

        Parameters
        ----------
        eval_res: Union[Dict[str, EvalSet], EvalSet]
            The input evaluated dataset(s).
        num_sets: Optional[Union[None, int]]
            The number of ``EvalSet`` objects in ``eval_res``. If not provided, inferred independently.
        average: Optional[bool]
            A boolean indicating whether to display the average performance as well. Relevant only if ``eval_res`` is
            a collection (``dict``)  of multiple ``EvalSet`` objects.
        title_suffix: Optional[str]
            Optional string to add to the title of the chart.
        random_sets: Optional[Union[List[EvalSet], None]]
            A list of randomly scored ``EvalSet`` objects, for benchmarking the performance associated with the
            evaluated dataset, if desired. Relevant only if ``eval_res`` contains a single ``EvalSet`` object.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        ax: Axes
            The axes on which the visualization was created, for further manipulation if required.

        """
        if not num_sets:
            average, num_sets = visualization.should_average(eval_res, average)

        ax, lines = visualization.chart_display_template(eval_res=eval_res,
                                                         metric='uplift_intersection',
                                                         num_sets=num_sets,
                                                         average=average,
                                                         min_quantile=self.min_quantile)

        if num_sets > 1:
            # compute the area under curve for each EvalSet, and label accordingly
            for (name, eval_set), line in zip(eval_res.items(), lines):
                auuc = self.integrate(x=eval_set.df['uplift_intersection'], dx=eval_set.get_quantile_interval())
                line.set_label(f"{name} (AUUC={auuc:.3f})")
            ax.legend(fancybox=True, shadow=True)

        else:  # add more information
            if isinstance(eval_res, dict):  # a single entry dict
                name, eval_set = next(iter(eval_res))
            else:  # has to be an EvalSet
                name, eval_set = '', eval_res

            # compute area under the curve and label accordingly
            dx = eval_set.get_quantile_interval()
            auuc = self.integrate(x=eval_set.df['uplift_intersection'], dx=dx)
            lines[0].set_label(f"Intersection Uplift (AUUC={auuc:.3f})")
            # color fill for the area under the curve
            u = visualization.chop_lower_quantiles(eval_set.df.set_index('normalized_index')['uplift_intersection'],
                                                   q=self.min_quantile)
            visualization.area_fill(ax=ax, s=u, base=0., alpha=0.2)

            # in case randomization is required
            if random_sets is not None:
                # stack and average across the randomly scored datasets
                s = np.stack([rand_res.df['uplift_intersection'].values for rand_res in random_sets], axis=-1).mean(
                    axis=1)
                auuc = self.integrate(x=s, dx=dx)
                visualization.single_curve_plot(signal=pd.Series(s, index=eval_set.df['normalized_index']),
                                                ax=ax, lw=1.5, min_quantile=self.min_quantile,
                                                label=f'Random (AUUC={auuc:.3f})', color='grey', linestyle='-.')

            # lines that are relevant only for multiple treatments
            if eval_set.is_multiple_actions:
                # Treated Uplift (see docstring)
                auuc = self.integrate(x=eval_set.df['uplift_treated'], dx=dx)
                visualization.single_curve_plot(signal=eval_set.df,
                                                metric='uplift_treated',
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                label=f'Treated Uplift (AUUC={auuc:.3f})')

                # Realized Vs Unrealized (see doctsring)
                auuc = self.integrate(x=eval_set.df['uplift_against_unrealized'], dx=dx)
                visualization.single_curve_plot(signal=eval_set.df,
                                                metric='uplift_against_unrealized',
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                label=f'Realized Vs Unrealized Uplift (AUUC={auuc:.3f})')

            # Compute *UnexposedResponseRatio* curve (see docstring)
            unexposed_diff = eval_set.df['below_response_control'] - eval_set.df['below_response_treated']
            unexposed_diff.index = eval_set.df['normalized_index']
            visualization.single_curve_plot(signal=unexposed_diff,
                                            ax=ax, lw=3, min_quantile=self.min_quantile,
                                            max_quantile=(1 - self.min_quantile),
                                            label=f'UnexposedResponseDiff')

            ax.set_ylabel('Uplift Estimate')
            ax.grid(True)

            # display the fraction of treatments ,out of the entire subgroup of observations associated with
            # non-neutral action, which is find within the acceptance region, for each upper quantile *q*.
            ax2 = ax.twinx()
            eval_set.df.set_index('normalized_index')['frac_of_overall_treated'].plot(ax=ax2, lw=3, color='black',
                                                                                      linestyle='--', label=None)
            ax2.set_ylabel('Frac.Of Observed Actions AbvThresh')

            leg = ax.legend()
            leg.remove()
            l2 = ax2.legend(fancybox=True, shadow=True)
            ax2.add_artist(leg)
            l2.remove()

        visualization.emphasize_axes(ax=ax, base=(0, 0))
        ax.set_xlabel('Sample Score Quantiles (Descending) / Exposed Fraction')
        ax.set_title(f"Uplift Curve\n{title_suffix}")
        return ax

    def display_fractional_lift_curve(self,
                                      eval_res: Union[Dict[str, EvalSet], EvalSet],
                                      num_sets: Optional[Union[None, int]] = None,
                                      average: Optional[bool] = False,
                                      title_suffix: Optional[str] = '',
                                      random_sets: Optional[Union[List[EvalSet], None]] = None,
                                      **kwargs
                                      ) -> Axes:
        """
        This method provides a visualization of the fractional lift curve(s) implied by the input evaluated dataset(s).
        According to ``Evaluator.visualization_methods()``, this method fits the keyword ``fractional_lift``.

        The uplift curve describes, for each upper quantile, the ratio in average response between the group of
        observations which were treated in accordance with the model recommendations, and a reference group of
        observations (e.g. observations which were not treated at all). The ratio is computed for each upper
        quantile ``q``, i.e. we take the observations from the upper *q*-th quantile of the dataset, split into the
        relevant groups, and compute the ratio of average response between these groups. Combining the
        calculations for all the quantiles, yields the fractional lift curve.
        Hence, the x-axis corresponds to upper quantiles of the score distribution - i.e. after
        the scores on a given ``EvalSet`` object are sorted in a descending manner, as part of the evaluation procedure,
        we refer to a ``normalized_index`` which is simply a mapping between the scores and the range of (0,1], so that
        the highest score is mapped to nearest to zero as possible, and the lowest score is mapped to one. With this,
        the ``Exposed Fraction``, as noted in the charts labels, refers to calculations that include the observations
        up to (or from) a specific value of the ``normalized_index``.

        Fractional lift curves will be labeled together with their corresponding Area Under Uplift Curve (AUUC),
        which is the result of the integral under the curve.

        In case the provided ``eval_res`` contains multiple ``EvalSet``\s, the visualization will contain, for each of
        the ``EvalSet`` objects, the curve of the fractional lift between the group in which the recommendations
        of the model intersect with the observed actions, and the group of observations associated with the neutral
        action (untreated). Each curve will be labeled according to its corresponding key on the provided dictionary.
        If ``average=True``, the chart will also contain the average performance, computed across the multiple
        ``EvalSet``\s provided (labeled as **Avg**).

        In case the provided ``eval_res`` contains a single ``EvalSet`` the visualization will contain the following
        curves:

        -   **Fractional Uplift** - fractional lift between the group in which the recommendations of the model
            intersect with the observed actions, and the group of observations associated with the neutral action
            (untreated).
        -   **Random** - the average fractional lift curve (just like the **Fractional Uplift** curve) calculated across
            the ``EvalSet`` objects contained in ``random_sets``, if provided.
        -   **FracLift Treated** - (relevant in case the ``EvalSet`` is associated with multiple actions) fractional
            lift between the treated group, disregarding the identity of the exact treatment, and the group of
            observations associated with the neutral action (untreated).
        -   **FracLift Vs Unrealized** - (relevant in case the ``EvalSet`` is associated with multiple actions)
            fractional lift between the group in which the recommendations of the model intersect with the observed
            actions (*realized*), and the group of observations in which the recommendations do not intersect with the
            observed actions (*unrealized*). Here, observations in the reference group can be associated with some
            non-neutral action, but just not the one recommended by the model.
        -   **UnexposedResponseRatio** - the ratio of average response in the complement region of the dataset, i.e.
            for each upper quantile ``q``, this curve takes into account the average responses of the treated group and
            the untreated group, in the lower ``(1-q)`` quantiles of the score, and subtracts between them.
            Values bigger than 1.0, imply that the average response of the untreated group, is higher than that of the
            treated group, for some lower quantile score of the dataset.

        Parameters
        ----------
        eval_res: Union[Dict[str, EvalSet], EvalSet]
            The input evaluated dataset(s).
        num_sets: Optional[Union[None, int]]
            The number of ``EvalSet`` objects in ``eval_res``. If not provided, inferred independently.
        average: Optional[bool]
            A boolean indicating whether to display the average performance as well. Relevant only if ``eval_res`` is
            a collection (``dict``)  of multiple ``EvalSet`` objects.
        title_suffix: Optional[str]
            Optional string to add to the title of the chart.
        random_sets: Optional[Union[List[EvalSet], None]]
            A list of randomly scored ``EvalSet`` objects, for benchmarking the performance associated with the
            evaluated dataset, if desired. Relevant only if ``eval_res`` contains a single ``EvalSet`` object.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        ax: Axes
            The axes on which the visualization was created, for further manipulation if required.

        """

        if not num_sets:
            average, num_sets = visualization.should_average(eval_res, average)

        # callable to apply on EvalSet.df for computing the fractional lift series
        frac_lift: Callable = lambda df: df['above_response_intersect'] / df['above_response_control']
        ax, lines = visualization.chart_display_template(eval_res=eval_res,
                                                         metric='Fractional Lift',
                                                         num_sets=num_sets,
                                                         average=average,
                                                         min_quantile=self.min_quantile,
                                                         func=frac_lift)

        if num_sets > 1:
            # compute the area under curve for each EvalSet, and label accordingly
            for (name, eval_set), line in zip(eval_res.items(), lines):
                auuc = self.integrate(x=frac_lift(eval_set.df), dx=eval_set.get_quantile_interval())
                line.set_label(f"{name} (AUUC={auuc:.3f})")
            ax.legend(fancybox=True, shadow=True)

        else:  # add more information
            if isinstance(eval_res, dict):  # a single entry dict
                name, eval_set = next(iter(eval_res))
            else:  # has to be an EvalSet
                name, eval_set = '', eval_res

            # compute area under the curve and label accordingly
            u: pd.Series = frac_lift(eval_set.df)
            u.index = eval_set.df['normalized_index']
            dx = eval_set.get_quantile_interval()
            auuc = self.integrate(x=u, dx=dx)
            lines[0].set_label(f"Fractional Uplift (AUUC={auuc:.3f})")
            # color fill for the area under the curve
            u = visualization.chop_lower_quantiles(u, q=self.min_quantile)
            visualization.area_fill(ax=ax, s=u, base=1., alpha=0.2)

            # in case randomization is required
            if random_sets is not None:
                # stack and average across the randomly scored datasets
                s = np.stack([frac_lift(rand_res.df).values for rand_res in random_sets], axis=-1).mean(
                    axis=1)
                auuc = self.integrate(x=s, dx=dx)
                visualization.single_curve_plot(signal=pd.Series(s, index=eval_set.df['normalized_index']),
                                                ax=ax, lw=1.5, min_quantile=self.min_quantile,
                                                label=f'Random (AUUC={auuc:.3f})', color='grey', linestyle='-.')

            # lines that are relevant only for multiple treatments
            if eval_set.is_multiple_actions:
                # FracLift Treated (see docstring)
                s = eval_set.df['above_response_treated'] / eval_set.df['above_response_control']
                s.index = eval_set.df['normalized_index']
                auuc = self.integrate(x=s, dx=dx)
                visualization.single_curve_plot(signal=s,
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                label=f'FracLift Treated (AUUC={auuc:.3f})')

                # FracLift Vs Unrealized (see docstring)
                s = eval_set.df['above_response_intersect'] / eval_set.df['above_response_unrealized']
                s.index = eval_set.df['normalized_index']
                auuc = self.integrate(x=s, dx=dx)
                visualization.single_curve_plot(signal=s,
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                label=f'FracLift Vs Unrealized (AUUC={auuc:.3f})')

            # Compute *UnexposedResponseRatio* curve (see docstring)
            unexposed_ratio = eval_set.df['below_response_control'] / eval_set.df['below_response_treated']
            unexposed_ratio.index = eval_set.df['normalized_index']
            visualization.single_curve_plot(signal=unexposed_ratio,
                                            ax=ax, lw=3, min_quantile=self.min_quantile,
                                            max_quantile=(1 - self.min_quantile),
                                            label=f'UnexposedResponseRatio')

            ax.set_ylabel('Fractional Lift')
            ax.grid(True)
            ax.legend(fancybox=True, shadow=True)

        visualization.emphasize_axes(ax=ax, base=(0, 1))
        ax.set_xlabel('Sample Score Quantiles (Descending) / Exposed Fraction')
        ax.set_title(f"Fractional Lift Curve\n{title_suffix}")

        return ax

    def display_gain(self,
                     eval_res: Union[Dict[str, EvalSet], EvalSet],
                     num_sets: Optional[Union[None, int]] = None,
                     average: Optional[bool] = False,
                     title_suffix: Optional[str] = '',
                     random_sets: Optional[Union[List[EvalSet], None]] = None,
                     **kwargs
                     ) -> Axes:
        """
        This method provides a visualization of the gain curve(s) implied by the input evaluated dataset(s).
        According to ``Evaluator.visualization_methods()``, this method fits the keyword ``gain``.

        The gain curve describes, for each upper quantile, the multiplication of two factors:

        -   difference in average response between the group of observations which were treated in accordance with the
            model recommendations, and a reference group of observations (e.g. observations which were not treated at
            all). The difference is computed for each upper quantile ``q``, i.e. we take the observations from the upper
            *q*-th quantile of the dataset, split into the relevant groups, and compute the difference in average
            response between these groups.
        -   The absolute number of untreated observations, found in the upper ``q``-th quantile, which grows bigger as
            ``q`` goes bigger.

        Multiplying these factors can be used to describe, the gain the model would yield, if it would have been *in
        action*, i.e. if we could count on the estimated difference in average response, and apply the actions
        recommended by the model, to the untreated observations, what is the quantity that will be added to the overall
        sum of responses.
        Combining the calculations of gain for all the quantiles, yields the gain curve.
        Hence, the x-axis corresponds to upper quantiles of the score distribution - i.e. after
        the scores on a given ``EvalSet`` object are sorted in a descending manner, as part of the evaluation procedure,
        we refer to a ``normalized_index`` which is simply a mapping between the scores and the range of (0,1], so that
        the highest score is mapped to nearest to zero as possible, and the lowest score is mapped to one. With this,
        the ``Exposed Fraction``, as noted in the charts labels, refers to calculations that include the observations
        up to (or from) a specific value of the ``normalized_index``.

        In case the provided ``eval_res`` contains multiple ``EvalSet``\s, the visualization will contain, for each of
        the ``EvalSet`` objects, the gain curve corresponding to the uplift between the group in which the
        recommendations of the model intersect with the observed actions, and the group of observations associated with
        the neutral action (untreated). Each curve will be labeled according to its corresponding key on the provided
        dictionary. If ``average=True``, the chart will also contain the average performance, computed across the
        multiple ``EvalSet``\s provided (labeled as **Avg**).

        In case the provided ``eval_res`` contains a single ``EvalSet`` the visualization will contain the following
        curves:

        -   **Intersection Gain** - gain assciated with the uplift between the group in which the recommendations
            of the model intersect with the observed actions, and the group of observations associated with the neutral
            action (untreated).
        -   **Treated Gain** - (relevant in case the ``EvalSet`` is associated with multiple actions) gain associated
            with the uplift between the treated group, disregarding the identity of the exact treatment, and the group
            of observations associated with the neutral action (untreated).
        -   **Random** - the average gain curve (just like the **Intersection Gain** curve) calculated across
            the ``EvalSet`` objects contained in ``random_sets``, if provided.
        -   In addition, the maximal gain points along the gain curves, are highlighted by markers, and labeled with the
            score value that corresponds to these points.

        Parameters
        ----------
        eval_res: Union[Dict[str, EvalSet], EvalSet]
            The input evaluated dataset(s).
        num_sets: Optional[Union[None, int]]
            The number of ``EvalSet`` objects in ``eval_res``. If not provided, inferred independently.
        average: Optional[bool]
            A boolean indicating whether to display the average performance as well. Relevant only if ``eval_res`` is
            a collection (``dict``)  of multiple ``EvalSet`` objects.
        title_suffix: Optional[str]
            Optional string to add to the title of the chart.
        random_sets: Optional[Union[List[EvalSet], None]]
            A list of randomly scored ``EvalSet`` objects, for benchmarking the performance associated with the
            evaluated dataset, if desired. Relevant only if ``eval_res`` contains a single ``EvalSet`` object.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        ax: Axes
            The axes on which the visualization was created, for further manipulation if required.

        """
        if not num_sets:
            average, num_sets = visualization.should_average(eval_res, average)

        ax, lines = visualization.chart_display_template(eval_res=eval_res,
                                                         metric='gain_intersection',
                                                         num_sets=num_sets,
                                                         average=average,
                                                         min_quantile=self.min_quantile)

        if num_sets == 1:
            # get the ``EvalSet`` object whether it was provided as input, or whether it was wrapped as part of a dict
            if isinstance(eval_res, dict):  # a single entry dict
                name, eval_set = next(iter(eval_res))
            else:  # has to be an EvalSet
                name, eval_set = '', eval_res

            points = []  # will collect maximal points of curves, which will be also displayed on the chart

            # modify the label associated with the curve already plotted for the single set
            lines[0].set_label(f"Intersection Gain")
            # accumulate its maximum
            points.append(self.get_max(eval_set, metric='gain_intersection'))

            if eval_set.is_multiple_actions:
                # add a curve for describing the expected response where the action space is considered as binary
                visualization.single_curve_plot(signal=eval_set.df,
                                                metric='gain_treated',
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                label=f'Treated Gain')
                # accumulate its maximum
                points.append(self.get_max(eval_set, metric='gain_treated'))

            if random_sets is not None:
                # stack and average across the randomly scored datasets
                s = np.stack([rand_res.df['gain_intersection'].values for rand_res in random_sets], axis=-1).mean(
                    axis=1)
                visualization.single_curve_plot(signal=pd.Series(s, index=eval_set.df['normalized_index']),
                                                ax=ax, lw=1.5, min_quantile=self.min_quantile,
                                                label=f'Random', color='grey', linestyle='-.')

            # highlight the maximal points we collected with corresponding markers and label with the
            # corresponding score value
            visualization.plot_points(ax=ax, points=points, legend_prefix='Score=', value_key='value')

            ax.grid(True)
            ax.legend(fancybox=True, shadow=True)

        ax.set_xlabel('Sample Score Quantiles (Descending) / Exposed Fraction')
        ax.set_ylabel('Gain')
        ax.set_title(f"Gain Curve\n{title_suffix}")

        return ax

    def display_avg_response(self,
                             eval_res: Union[Dict[str, EvalSet], EvalSet],
                             num_sets: Optional[Union[None, int]] = None,
                             average: Optional[bool] = False,
                             title_suffix: Optional[str] = '',
                             random_sets: Optional[Union[List[EvalSet], None]] = None,
                             **kwargs
                             ) -> Axes:
        """
        This method provides a visualization of the expected response curve(s) implied by the input evaluated
        dataset(s).
        According to ``Evaluator.visualization_methods()``, this method fits the keyword ``avg_response``.

        The expected response curve describes, for each quantile *q*, a weighted average between two factors:

        -   The average response, within the *q*-th upper score quantile, of the group of observations in which the
            actions recommended by the model intersect with the observed actions. This factor is weighted by ``q``.
        -   The average response, within the *(1-q)*-th lower score quantile, of the untreated group, i.e. the group of
            observations associated with the neutral action. This factor is weighted by ``1-q``.

        The expected response curve, can be used to estimate, for each qunatile ``q``, the average response of the
        **entire population**, if we *expose* the *q*-th upper qunatiles of the score distribution to the
        recommendations of the model.

        In case the provided ``eval_res`` contains multiple ``EvalSet``\s, the visualization will contain, for each of
        the ``EvalSet`` objects, the expected response curve in which intersections are considered only where the
        observed actions corresponds to the exact specific action recommended by the model. Each curve will be labeled
        according to its corresponding key on the provided dictionary. If ``average=True``, the chart will also contain
        the average performance, computed across the multiple ``EvalSet``\s provided (labeled as **Avg**).

        In case the provided ``eval_res`` contains a single ``EvalSet`` the visualization will contain the following
        curves:

        -   **IntersectionExpectedResponse** - the expected response curve in which intersections are considered only
            where the observed actions corresponds to the exact specific action recommended by the model. The maximal
            point along this curve is highlighted by a corresponding marker, and labeled with the corresponding score
            value.
        -   **TreatedExpectedResponse** - (relevant in case the ``EvalSet`` is associated with multiple actions) regards
            the entire set of non-neutral actions as a single binary action, and describes the expected response where
            intersections are counted accordingly. The maximal point along this curve is highlighted by a corresponding
            marker, and labeled with the corresponding score value.
        -   **Random** - the expected response curve (just like the **IntersectionExpectedResponse** curve) calculated
            across the ``EvalSet`` objects contained in ``random_sets``, if provided.
        -   **AvgResponseIntersectedTreatments** - for each upper quantile, *q*, describes the average response of the
            observations in which the recommended treatment by the model, intersects with the observed treatment. This
            curve is also accompanied by an uncertainty sleeve, describing the confidence interval according to the
            corredsponding standard error of the estimated average response.
        -   **AvgResponseUntreated** - for each upper quantile, *q*, describes the average response of the
            untreated observations. This curve is also accompanied by an uncertainty sleeve, describing the confidence
            interval according to the corredsponding standard error of the estimated average response.
        -   **OverallAvgResponse** - describes the average response observed for the entire dataset, disregarding
            score values.
        -   **UntreatedAvgResponse** - describes the average response among the untreated observations, disregarding
            score values.


        Parameters
        ----------
        eval_res: Union[Dict[str, EvalSet], EvalSet]
            The input evaluated dataset(s).
        num_sets: Optional[Union[None, int]]
            The number of ``EvalSet`` objects in ``eval_res``. If not provided, inferred independently.
        average: Optional[bool]
            A boolean indicating whether to display the average performance as well. Relevant only if ``eval_res`` is
            a collection (``dict``)  of multiple ``EvalSet`` objects.
        title_suffix: Optional[str]
            Optional string to add to the title of the chart.
        random_sets: Optional[Union[List[EvalSet], None]]
            A list of randomly scored ``EvalSet`` objects, for benchmarking the performance associated with the
            evaluated dataset, if desired. Relevant only if ``eval_res`` contains a single ``EvalSet`` object.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        ax: Axes
            The axes on which the visualization was created, for further manipulation if required.

        """
        if not num_sets:
            average, num_sets = visualization.should_average(eval_res, average)

        ax, lines = visualization.chart_display_template(eval_res=eval_res,
                                                         metric='expected_response_intersect',
                                                         num_sets=num_sets,
                                                         average=average,
                                                         min_quantile=self.min_quantile)

        if num_sets == 1:
            # get the ``EvalSet`` object whether it was provided as input, or whether it was wrapped as part of a dict
            if isinstance(eval_res, dict):  # a single entry dict
                name, eval_set = next(iter(eval_res))
            else:  # has to be an EvalSet
                name, eval_set = '', eval_res

            points = []  # will collect maximal points of curves, which will be also displayed on the chart

            # modify the label associated with the curve already plotted for the single set
            lines[0].set_label(f"IntersectionExpectedResponse")
            # accumulate its maximum
            points.append(self.get_max(eval_set, metric='expected_response_intersect'))

            if eval_set.is_multiple_actions:
                # add a curve for describing the expected response where the action space is considered as binary
                visualization.single_curve_plot(signal=eval_set.df,
                                                metric='expected_response_treated',
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                label=f'TreatedExpectedResponse')
                # accumulate its maximum
                points.append(self.get_max(eval_set, metric='expected_response_treated'))

            if random_sets is not None:
                # stack and average across the randomly scored datasets
                s = np.stack([rand_res.df['expected_response_intersect'].values for rand_res in random_sets],
                             axis=-1).mean(axis=1)
                visualization.single_curve_plot(signal=pd.Series(s, index=eval_set.df['normalized_index']),
                                                ax=ax, lw=1.5, min_quantile=self.min_quantile,
                                                label=f'Random', color='grey', linestyle='-.')

            # compute the corresponding standard error confidence intervals for the average response curves
            # corresponding to:
            # - intersections cases in the acceptance region.
            # - untreated cases in the rejection region.
            if eval_set.is_binary_response:
                # confidence interval based on standard error for proportion estimate
                se_exposed = utils.get_standard_error_proportion(  # intersections cases in the acceptance region
                    sample_size=eval_set.df['intersect_count'],
                    proportion_estimate=eval_set.df['above_response_intersect'])
                se_unexposed = utils.get_standard_error_proportion(  # untreated cases in the rejection region
                    sample_size=eval_set.df['control_count'].iloc[-1] - eval_set.df['control_count'],
                    proportion_estimate=eval_set.df['below_response_control'])

            else:  # in the case of continuous response

                # confidence interval based on standard error for mean estimate
                # For that we need the standarad deviation associated with each sample for which the mean was
                # estimated

                # intersection cases in the acceptance region
                exposed = eval_set.df['is_intersect'] > 0
                std_exposed = eval_set.df.loc[exposed, eval_set.response_field].expanding(
                    2).std(ddof=1).reindex(index=eval_set.df.index, method='ffill')
                se_exposed = utils.get_standard_error_mean(sample_size=eval_set.df['intersect_count'],
                                                           std=std_exposed)

                # untreated cases in the rejection region
                unexposed = eval_set.df[eval_set.observed_action_field] == eval_set.control_indicator
                unexposed_responses = eval_set.df.loc[unexposed, eval_set.response_field].iloc[::-1]
                std_unexposed = unexposed_responses.expanding(2).std(ddof=1).iloc[::-1].reindex(index=eval_set.df.index,
                                                                                                method='ffill')
                se_unexposed = utils.get_standard_error_mean(
                    sample_size=eval_set.df['control_count'].iloc[-1] - eval_set.df['control_count'],
                    std=std_unexposed)

            # display avg response for the intersection cases in the acceptance region
            # accompanied by uncertainty sleeve
            visualization.single_curve_plot(signal=eval_set.df,
                                            metric='above_response_intersect',
                                            ax=ax, lw=3,
                                            min_quantile=self.min_quantile,
                                            label=f'AvgResponseIntersectedTreatments')
            visualization.display_sleeve(ax=ax, eval_set=eval_set, metric='above_response_intersect',
                                         margin=se_exposed, color=ax.get_lines()[-1].get_color(),
                                         min_quantile=self.min_quantile * 2)

            # display avg response for the untreated cases in the rejection region
            # accompanied by uncertainty sleeve
            visualization.single_curve_plot(signal=eval_set.df,
                                            metric='below_response_control',
                                            ax=ax, lw=3,
                                            min_quantile=self.min_quantile,
                                            max_quantile=(1 - self.min_quantile),
                                            label=f'AvgResponseUntreated')
            visualization.display_sleeve(ax=ax, eval_set=eval_set, metric='below_response_control',
                                         margin=se_unexposed, color=ax.get_lines()[-1].get_color(),
                                         max_quantile=(1 - 2 * self.min_quantile))

            # highlight the maximal points we collected with corresponding markers and label with the
            # corresponding score value
            visualization.plot_points(ax=ax, points=points, legend_prefix='Score=', value_key='value')

            # add horizontal lines which will describe:
            # - the average response across the entire dataset.
            # - the average response associated with the untreated observations.
            where_untreated = eval_set.df[eval_set.observed_action_field] == eval_set.control_indicator
            ax.axhline(y=eval_set.df[eval_set.response_field].mean(), color='darkviolet', linestyle='--', lw=2,
                       label='OverallAvgResponse')
            ax.axhline(y=eval_set.df.loc[where_untreated, eval_set.response_field].mean(),
                       color='darkgoldenrod', linestyle=':', lw=2, label='UntreatedAvgResponse')

            ax.grid(True)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                      ncol=4, fancybox=True, shadow=True, fontsize=12)

        ax.set_xlabel('Sample Score Quantiles (Descending) / Exposed Fraction')
        ax.set_ylabel('AverageResponse Estimate')
        ax.set_title(f"Expected Response Estimate \n{title_suffix}")

        return ax

    def display_acceptance_region_stats(self,
                                        eval_res: Union[Dict[str, EvalSet], EvalSet],
                                        num_sets: Optional[Union[None, int]] = None,
                                        average: Optional[bool] = False,
                                        title_suffix: Optional[str] = '',
                                        random_sets: Optional[Union[List[EvalSet], None]] = None,
                                        **kwargs
                                        ) -> Axes:
        """
        This method provides a visualization of the estimated average response, among certain subgroups of the input
        ``EvalSet``(s), in the acceptance region, i.e. where the scores lie within some upper quantile.
        According to ``Evaluator.visualization_methods()``, this method fits the keyword ``targeted_region``.

        In case the provided ``eval_res`` contains multiple ``EvalSet``\s, the visualization will contain, for each of
        the ``EvalSet`` objects, the average response of the observations in which the recommended treatment by the
        model, intersects with the observed treatment, as a function of the upper quantile *q*, which defines the
        *acceptance region*. Each curve will be labeled according to its corresponding key on the provided dictionary.
        If ``average=True``, the chart will also contain the average performance, computed across the multiple
        ``EvalSet``\s provided (labeled as **Avg**).

        In case the provided ``eval_res`` contains a single ``EvalSet`` the visualization will contain the following
        curves:

        -   **Intersections** - the average response of the observations in which the recommended treatment by the
            model, intersects with the observed treatment, as a function of the upper quantile *q*, which defines the
            *acceptance region*.
        -   **Untreated** - the average response of the observations associated with the neutral action, as a function
            of the upper quantile *q*, which defines the *acceptance region*.
        -   **Treated** - (relevant in case the ``EvalSet`` is associated with multiple actions) regards
            the entire set of non-neutral actions as a single binary action, and describes the average response, where
            intersections are counted accordingly, within the *acceptance region* defined according to the upper
            quantile *q*.
        -   **Unrealized** - (relevant in case the ``EvalSet`` is associated with multiple actions) the average response
            among the group of observations in which the recommendations of the model do not intersect with the observed
            actions, within the *acceptance region* defined according to the upper quantile *q*.
        -   **OverallAvgResponse** - describes the average response observed for the entire dataset, disregarding
            score values.
        -   **UntreatedAvgResponse** - describes the average response among the untreated observations, disregarding
            score values.
        -   **pVal vs Untreated** - corresponding to the right y-axis, describes the result of applying a statistical
            hypothesis test to the difference between the estimated average response among the *Intersections* group and
            the *Untreated* group. In the case of binary response, the statistical test will be a `proportions test
            <https://stattrek.com/hypothesis-test/difference-in-proportions.aspx>`_, and in the case of a continuous
            response, hypothesis testing will be performed via `t-test
            <https://stattrek.com/hypothesis-test/difference-in-means.aspx?tutorial=AP>`_.

        the curves: **Intersections**, **Untreated**, **Treated**, **Unrealized** will also be accompanied by an
        uncertainty sleeve, describing the confidence interval according to the corredsponding standard error of the
        estimated average responses.

        Parameters
        ----------
        eval_res: Union[Dict[str, EvalSet], EvalSet]
            The input evaluated dataset(s).
        num_sets: Optional[Union[None, int]]
            The number of ``EvalSet`` objects in ``eval_res``. If not provided, inferred independently.
        average: Optional[bool]
            A boolean indicating whether to display the average performance as well. Relevant only if ``eval_res`` is
            a collection (``dict``)  of multiple ``EvalSet`` objects.
        title_suffix: Optional[str]
            Optional string to add to the title of the chart.
        random_sets: Optional[Union[List[EvalSet], None]]
            A list of randomly scored ``EvalSet`` objects, for benchmarking the performance associated with the
            evaluated dataset, if desired. Relevant only if ``eval_res`` contains a single ``EvalSet`` object.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        ax: Axes
            The axes on which the visualization was created, for further manipulation if required.

        """
        if not num_sets:
            average, num_sets = visualization.should_average(eval_res, average)

        if num_sets > 1:  # multiple EvalSet object
            ax, lines = visualization.chart_display_template(eval_res=eval_res,
                                                             metric='above_response_intersect',
                                                             num_sets=num_sets,
                                                             average=average,
                                                             min_quantile=self.min_quantile)
        else:
            fig, ax = plt.subplots()

            # get the ``EvalSet`` object whether it was provided as input, or whether it was wrapped as part of a dict
            if isinstance(eval_res, dict):  # a single entry dict
                name, eval_set = next(iter(eval_res))
            else:  # has to be an EvalSet
                name, eval_set = '', eval_res

            metrics = ['above_response_intersect', 'above_response_control']
            counts = ['intersect_count', 'control_count']
            indicators = ['is_intersect', 'is_control']
            labels = ['Intersections', 'Untreated']
            if eval_set.is_multiple_actions:
                metrics += ['above_response_treated', 'above_response_unrealized']
                counts += ['treated_count', 'unrealized_count']
                indicators += ['is_treated', 'is_unrealized']
                labels += ['Treated', 'Unrealized']
            stds = dict()

            for metric_col, count_col, indicator_col, label in zip(metrics, counts, indicators, labels):
                # for each combination of such
                # display the line
                visualization.single_curve_plot(signal=eval_set.df,
                                                metric=metric_col,
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                label=label)
                # compute the corresponding standard error confidence interval for the curve
                if eval_set.is_binary_response:
                    # confidence interval based on standard error for proportion estimate
                    se = utils.get_standard_error_proportion(
                        sample_size=eval_set.df[count_col],
                        proportion_estimate=eval_set.df[metric_col])
                else:
                    # confidence interval based on standard error for mean estimate
                    # For that we need the standarad deviation associated with each sample for which the mean was
                    # estimated
                    exposed = eval_set.df[indicator_col] > 0
                    std = eval_set.df.loc[exposed, eval_set.response_field].expanding(
                        2).std(ddof=1).reindex(index=eval_set.df.index, method='ffill')
                    stds[label] = std  # store std for further hypothesis testing
                    se = utils.get_standard_error_mean(sample_size=eval_set.df[count_col],
                                                       std=std)

                # display the uncertainty sleeve around the estimated avg response
                visualization.display_sleeve(ax=ax, eval_set=eval_set, metric=metric_col,
                                             margin=se, color=ax.get_lines()[-1].get_color(),
                                             min_quantile=self.min_quantile * 2)

            # add horizontal lines which will describe:
            # - the average response across the entire dataset.
            # - the average response associated with the untreated observations.
            where_untreated = eval_set.df[eval_set.observed_action_field] == eval_set.control_indicator
            ax.axhline(y=eval_set.df[eval_set.response_field].mean(), color='darkviolet', linestyle='--', lw=2,
                       label='OverallAvgResponse')
            ax.axhline(y=eval_set.df.loc[where_untreated, eval_set.response_field].mean(),
                       color='darkgoldenrod', linestyle=':', lw=2, label='UntreatedAvgResponse')

            # Statistical hypothesis testing
            if eval_set.is_binary_response:
                # perform proportions test
                pval = utils.proportions_test(sample_siz_1=eval_set.df['intersect_count'],
                                              sample_siz_2=eval_set.df['control_count'],
                                              proportion_est_1=eval_set.df['above_response_intersect'],
                                              proportion_est_2=eval_set.df['above_response_control'])
            else:
                # perform t-test
                pval = utils.t_test(mu_1=eval_set.df['above_response_intersect'],
                                    mu_2=eval_set.df['above_response_control'],
                                    sample_siz_1=eval_set.df['intersect_count'],
                                    sample_siz_2=eval_set.df['control_count'],
                                    std_1=stds['Intersections'],
                                    std_2=stds['Untreated']
                                    )

            # p-value will be displayed on the right y-axis
            pval.index = eval_set.df['normalized_index']
            ax2 = ax.twinx()
            pval.plot(ax=ax2, color='k', label='pVal vs Untreated', lw=1)
            ax2.set_yscale('log')
            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)
            ax2.set_ylabel('p-value Difference Test')
            ax.grid(True)
            ax.legend(fancybox=True, shadow=True)

        ax.set_xlabel('Sample Score Quantiles (Descending) / Exposed Fraction')
        ax.set_ylabel('AverageResponse Estimate')
        ax.set_title(f"Acceptance Region - Average Responses \n{title_suffix}")

        return ax

    def display_rejection_region_stats(self,
                                       eval_res: Union[Dict[str, EvalSet], EvalSet],
                                       num_sets: Optional[Union[None, int]] = None,
                                       average: Optional[bool] = False,
                                       title_suffix: Optional[str] = '',
                                       random_sets: Optional[Union[List[EvalSet], None]] = None,
                                       **kwargs
                                       ) -> Axes:
        """
        This method provides a visualization of the estimated average response, among certain subgroups of the input
        ``EvalSet``(s), outside the *acceptance region* (or inside the *rejection region*), i.e. where the scores lie
        within some lower quantile.
        According to ``Evaluator.visualization_methods()``, this method fits the keyword ``untargeted_region``.

        In case the provided ``eval_res`` contains multiple ``EvalSet``\s, the visualization will contain, for each of
        the ``EvalSet`` objects, the average response of the observations associated with the neutral action,
        as a function of the upper quantile *q*, which defines the *rejection region*, i.e. the range of scores where
        the recommendations of the model would have been rejected, if *q* would have been used as the score threhshold.
        Each curve will be labeled according to its corresponding key on the provided dictionary. If ``average=True``,
        the chart will also contain the average performance, computed across the multiple ``EvalSet``\s
        provided (labeled as **Avg**).

        In case the provided ``eval_res`` contains a single ``EvalSet`` the visualization will contain the following
        curves:

        -   **Untreated** - the average response of the observations associated with the neutral action, within the
            lower *(1-q)* quantiles of the score distribution, i.e. the *rejection region*.
        -   **Treated** - the average response of the observations associated with a non-neutral action, within the
            lower *(1-q)* quantiles of the score distribution, i.e. the *rejection region*.
        -   **OverallAvgResponse** - describes the average response observed for the entire dataset, disregarding
            score values.
        -   **UntreatedAvgResponse** - describes the average response among the untreated observations, disregarding
            score values.
        -   **pVal vs Untreated** - corresponding to the right y-axis, describes the result of applying a statistical
            hypothesis test to the difference between the estimated average response among the *Untreated* group and
            the *Treated* group, in the *rejection region*. In the case of binary response, the statistical test will
            be a `proportions test
            <https://stattrek.com/hypothesis-test/difference-in-proportions.aspx>`_, and in the case of a continuous
            response, hypothesis testing will be performed via `t-test
            <https://stattrek.com/hypothesis-test/difference-in-means.aspx?tutorial=AP>`_.

        the curves: **Untreated**, **Treated** will also be accompanied by an uncertainty sleeve, describing the
        confidence interval according to the corredsponding standard error of the estimated average responses.

        Parameters
        ----------
        eval_res: Union[Dict[str, EvalSet], EvalSet]
            The input evaluated dataset(s).
        num_sets: Optional[Union[None, int]]
            The number of ``EvalSet`` objects in ``eval_res``. If not provided, inferred independently.
        average: Optional[bool]
            A boolean indicating whether to display the average performance as well. Relevant only if ``eval_res`` is
            a collection (``dict``)  of multiple ``EvalSet`` objects.
        title_suffix: Optional[str]
            Optional string to add to the title of the chart.
        random_sets: Optional[Union[List[EvalSet], None]]
            A list of randomly scored ``EvalSet`` objects, for benchmarking the performance associated with the
            evaluated dataset, if desired. Relevant only if ``eval_res`` contains a single ``EvalSet`` object.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        ax: Axes
            The axes on which the visualization was created, for further manipulation if required.

        """
        if not num_sets:
            average, num_sets = visualization.should_average(eval_res, average)

        if num_sets > 1:  # multiple EvalSet object
            ax, lines = visualization.chart_display_template(eval_res=eval_res,
                                                             metric='below_response_control',
                                                             num_sets=num_sets,
                                                             average=average,
                                                             min_quantile=self.min_quantile,
                                                             max_quantile=(1 - self.min_quantile))
        else:
            fig, ax = plt.subplots()

            # get the ``EvalSet`` object whether it was provided as input, or whether it was wrapped as part of a dict
            if isinstance(eval_res, dict):
                name, eval_set = next(iter(eval_res))
            else:
                name, eval_set = '', eval_res

            metrics = ['below_response_control', 'below_response_treated']
            counts = ['control_count', 'treated_count']
            indicators = ['is_control', 'is_treated']
            labels = ['Untreated', 'Treated']
            stds, sample_sizes = dict(), dict()

            for metric_col, count_col, indicator_col, label in zip(metrics, counts, indicators, labels):
                # for each combination of such
                # display the line
                visualization.single_curve_plot(signal=eval_set.df,
                                                metric=metric_col,
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                max_quantile=(1 - self.min_quantile),
                                                label=label)
                # calculate the complement number of observations, by subtracting the cumulative count series,
                # from its final value
                sample_sizes[label] = eval_set.df[count_col].iloc[-1] - eval_set.df[count_col]

                # compute the corresponding standard error confidence interval for the curve
                if eval_set.is_binary_response:
                    # confidence interval based on standard error for proportion estimate
                    se = utils.get_standard_error_proportion(
                        sample_size=sample_sizes[label],
                        proportion_estimate=eval_set.df[metric_col])
                else:
                    # confidence interval based on standard error for mean estimate
                    # For that we need the standarad deviation associated with each sample for which the mean was
                    # estimated (in this case, from the end towards the beginning, gradually)
                    unexposed = eval_set.df[indicator_col] > 0
                    unexposed_responses = eval_set.df.loc[unexposed, eval_set.response_field].iloc[::-1]
                    std_unexposed = unexposed_responses.expanding(2).std(ddof=1).iloc[::-1].reindex(
                        index=eval_set.df.index,
                        method='ffill')
                    stds[label] = std_unexposed  # store std for further hypothesis testing
                    se = utils.get_standard_error_mean(
                        sample_size=sample_sizes[label],
                        std=std_unexposed)

                # display the uncertainty sleeve around the estimated avg response
                visualization.display_sleeve(ax=ax, eval_set=eval_set, metric=metric_col,
                                             margin=se, color=ax.get_lines()[-1].get_color(),
                                             min_quantile=self.min_quantile * 2,
                                             max_quantile=(1 - 2 * self.min_quantile))

            # add horizontal lines which will describe:
            # - the average response across the entire dataset.
            # - the average response associated with the untreated observations.
            where_untreated = eval_set.df[eval_set.observed_action_field] == eval_set.control_indicator
            ax.axhline(y=eval_set.df[eval_set.response_field].mean(), color='darkviolet', linestyle='--', lw=2,
                       label='OverallAvgResponse')
            ax.axhline(y=eval_set.df.loc[where_untreated, eval_set.response_field].mean(),
                       color='darkgoldenrod', linestyle=':', lw=2, label='UntreatedAvgResponse')

            # Statistical hypothesis testing
            if eval_set.is_binary_response:
                # perform proportions test
                pval = utils.proportions_test(sample_siz_1=sample_sizes['Untreated'],
                                              sample_siz_2=sample_sizes['Treated'],
                                              proportion_est_1=eval_set.df['below_response_control'],
                                              proportion_est_2=eval_set.df['below_response_treated'])
            else:
                # perform t-test
                pval = utils.t_test(mu_1=eval_set.df['below_response_control'],
                                    mu_2=eval_set.df['below_response_treated'],
                                    sample_siz_1=sample_sizes['Untreated'],
                                    sample_siz_2=sample_sizes['Treated'],
                                    std_1=stds['Untreated'],
                                    std_2=stds['Treated']
                                    )

            # p-value will be displayed on the right y-axis
            pval.index = eval_set.df['normalized_index']
            ax2 = ax.twinx()
            pval.plot(ax=ax2, color='k', label='pVal vs Untreated', lw=1)
            ax2.set_yscale('log')
            ax2.legend(loc='upper center', bbox_to_anchor=(0.4, 1.05), fancybox=True, shadow=True)
            ax2.set_ylabel('p-value Difference Test')
            ax.grid(True)
            ax.legend(loc='upper center', bbox_to_anchor=(0.75, 1.00), fancybox=True, shadow=True)

        ax.set_xlabel('Sample Score Quantiles (Descending) / Exposed Fraction')
        ax.set_ylabel('AverageResponse Estimate')
        ax.set_title(f"Rejection Region - Average Responses \n{title_suffix}")

        return ax

    def display_agreement_stats(self,
                                eval_res: Union[Dict[str, EvalSet], EvalSet],
                                num_sets: Optional[Union[None, int]] = None,
                                average: Optional[bool] = False,
                                title_suffix: Optional[str] = '',
                                random_sets: Optional[Union[List[EvalSet], None]] = None,
                                **kwargs
                                ) -> Axes:
        """
        This method provides a visualization of the agreements/intersections statistics implied by the input evaluated
        dataset(s), as a function of the score quantile.

        In case the provided ``eval_res`` contains multiple ``EvalSet``\s, the visualization will contain, for each of
        the ``EvalSet`` objects, the rate in which intersections occur, for each *acceptance region*, which is defined
        according to the upper quantile *q*. Each curve will be labeled according to its corresponding key on the
        provided dictionary. If ``average=True``, the chart will also contain the average performance, computed across
        the multiple``EvalSet``\s provided (labeled as **Avg**).

        In case the provided ``eval_res`` contains a single ``EvalSet`` the visualization will contain the following
        curves:

        -   **AgreementRate** - the rate in which intersections occur, for each *acceptance region*, which is defined
            according to the upper quantile *q*.
        -   **Random** - the agreemnt rate curve (just like the **AgreementRate** curve) calculated
            across the ``EvalSet`` objects contained in ``random_sets``, if provided.
        -   **BinaryAgreementRate** - (relevant in case the ``EvalSet`` is associated with multiple actions) regards
            the entire set of non-neutral actions as a single binary action, and describes the agreement rate where
            intersections are counted accordingly.
        -   **# of Agreements** - corresponds to the y-axis, displays the absolute number of intersections/agreements,
            w.r.t to each *acceptance region*, defined according to the upper quantile *q*.


        Parameters
        ----------
        eval_res: Union[Dict[str, EvalSet], EvalSet]
            The input evaluated dataset(s).
        num_sets: Optional[Union[None, int]]
            The number of ``EvalSet`` objects in ``eval_res``. If not provided, inferred independently.
        average: Optional[bool]
            A boolean indicating whether to display the average performance as well. Relevant only if ``eval_res`` is
            a collection (``dict``)  of multiple ``EvalSet`` objects.
        title_suffix: Optional[str]
            Optional string to add to the title of the chart.
        random_sets: Optional[Union[List[EvalSet], None]]
            A list of randomly scored ``EvalSet`` objects, for benchmarking the performance associated with the
            evaluated dataset, if desired. Relevant only if ``eval_res`` contains a single ``EvalSet`` object.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        ax: Axes
            The axes on which the visualization was created, for further manipulation if required.

        """
        if not num_sets:
            average, num_sets = visualization.should_average(eval_res, average)

        fig, ax = plt.subplots()
        # callable to apply on EvalSet object for finding the intersection series
        decision_agreement = lambda es: (es.df[es.proposed_action_field] == es.df[es.observed_action_field]).astype(int)

        if num_sets > 1:
            series_dict = dict()  # will collection the plotted series for each EvalSet
            for name, single_res in eval_res.items():  # for each ``EvalSet`` object
                # compute agreement series and append the corresponding line
                agreement_rate = decision_agreement(single_res)
                agreement_rate.index = single_res.df['normalized_index']
                # expanding mean is used to compute the running average
                series_dict[name], line = visualization.single_curve_plot(agreement_rate.expanding().mean(),
                                                                          ax=ax, lw=1,
                                                                          label=name,
                                                                          min_quantile=self.min_quantile)
            if average:
                # a new dataframe will interpolate the accumulated signals, and compute
                # the average of them, for each quantile
                combined_df = pd.DataFrame(series_dict)
                combined_df = combined_df.interpolate('index')
                aggregated = combined_df.mean(axis=1)
                aggregated.plot(ax=ax, lw=3, label='Avg', color='k', alpha=0.5)

        else:  # add more information
            if isinstance(eval_res, dict):  # a single entry dict
                name, eval_set = next(iter(eval_res))
            else:  # has to be an EvalSet
                name, eval_set = '', eval_res

            # main signal
            u: pd.Series = decision_agreement(eval_set).expanding().mean()
            u.index = eval_set.df['normalized_index']
            visualization.single_curve_plot(u,
                                            ax=ax, lw=3,
                                            label='AgreementRate',
                                            min_quantile=self.min_quantile)

            # in case randomization is required
            if random_sets is not None:
                # stack and average across the randomly scored datasets
                s = np.stack([decision_agreement(rand_res).expanding().mean().values for rand_res in random_sets],
                             axis=-1).mean(axis=1)
                visualization.single_curve_plot(signal=pd.Series(s, index=eval_set.df['normalized_index']),
                                                ax=ax, lw=1.5, min_quantile=self.min_quantile,
                                                label=f'Random', color='grey', linestyle='-.')

            # lines that are relevant only for multiple treatments
            if eval_set.is_multiple_actions:
                # binary agreement is defined when both the observed action and the recommended action are diffrent
                # from the neutral action
                u = np.logical_and(eval_set.df[eval_set.proposed_action_field] != eval_set.control_indicator,
                                   eval_set.df[eval_set.observed_action_field] != eval_set.control_indicator).astype(
                    int).expanding().mean()
                u.index = eval_set.df['normalized_index']
                visualization.single_curve_plot(u,
                                                ax=ax, lw=3,
                                                label='BinaryAgreementRate',
                                                min_quantile=self.min_quantile)

            # on the second y- axis
            # add the absolute number of aggrements found, by *summing* aggrements gradually, as qunatile grows
            ax2 = ax.twinx()
            u = decision_agreement(eval_set).expanding().sum()
            u.index = eval_set.df['normalized_index']
            visualization.single_curve_plot(u,
                                            ax=ax2, lw=3,
                                            label='# of Agreements',
                                            color='darkviolet',
                                            min_quantile=self.min_quantile)
            ax2.legend(fancybox=True, shadow=True)
            ax2.set_ylabel('# of Intersections')

        ax.grid(True)
        ax.legend(fancybox=True, shadow=True)
        ax.set_xlabel('Sample Score Quantiles (Descending) / Exposed Fraction')
        ax.set_ylabel('Rate of Intersections (In Acceptance Region)')
        ax.set_title(f"Agreement/Intersection Statistics\n{title_suffix}")
        return ax

    def get_distribution_by_threshold(self,
                                      eval_res: Union[Dict[str, EvalSet], EvalSet],
                                      thresh: float,
                                      title_suffix: Optional[str] = '') -> Axes:
        """
        This method provides a visualization of the distribution of recommended treatments, according to a certain
        input score threshold, and with respect to the outputs of the model as they are represented in the input
        ``EvalSet`` object.

        When we consider a certain score threshold, every recommendation of the model that is associated with scores
        higher than the threshold are taken into account, and recommendations associated with lower scores, are
        considered as a recommendation for a neutral action.

        This visualization considers three groups:

        -   ``Observed``: The distribution of the observed/actual treatments/actions in the entire dataset (here, the
            threshold has no affect, as this distribution was observed regardless of the scores associated with each
            observation).
        -   ``Recommended``: The distribution of the recommendations of the model, where scores lower than the threshold
            are considered as a recommendation for the neutral action.
        -   ``Intersection``: The distribution of actions in the intersection set between the observed actions and the
            ones recommended by the model.

        The upper chart describes the rate, or fraction, associated with each treatment for each of the groups listed
        above.
        The lower chart describes the absolute number of occurences of each treatment for each of the groups listed
        above.

        Parameters
        ----------
        eval_res: Union[Dict[str, EvalSet], EvalSet]
            The input evaluated dataset. If provided as a dictionary, must contain only a single entry.
        thresh: float
            The score threshold according to which the distribution of recommendations will be visualized.
        title_suffix: Optional[str]
            Optional string to add to the title of the chart.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        ax: Axes
            The axes on which the visualization was created, for further manipulation if required.

        """

        assert isinstance(eval_res, EvalSet), "The input ``eval_res`` must be of type EvalSet"
        assert eval_res.df[eval_res.score_field].min() <= thresh <= eval_res.df[eval_res.score_field].max(), \
            "The provided threshold does not lie within the range of scores"

        fig, ax_arr = plt.subplots(nrows=2)

        # observed actions
        observed = eval_res.df[eval_res.observed_action_field]
        # recommended actions
        recommendations = eval_res.df[eval_res.proposed_action_field].copy()
        recommendations[eval_res.df[eval_res.score_field] < thresh] = eval_res.control_indicator
        # intersection of recommendations and observed actions
        intersected_recommendations = recommendations.loc[recommendations == observed]

        # The loop will create two visualizations
        # where the difference will be in the normalization of the counts
        for norm_bool, ax in zip([True, False], ax_arr):
            # merge distribution of observed and recommendations
            distribs = pd.merge(observed.value_counts(normalize=norm_bool).rename('Observed'),
                                recommendations.value_counts(normalize=norm_bool).rename('Recommended'),
                                left_index=True, right_index=True)
            # merge the result with the distribution of the intersections
            distribs = pd.merge(distribs,
                                intersected_recommendations.value_counts(normalize=norm_bool).rename('Intersection'),
                                left_index=True, right_index=True)

            distribs.sort_index().plot(kind='bar', ax=ax)
            ax.grid(True)

            # add annotations
            for p in ax.patches:
                h = p.get_height()
                ax.annotate(f"{h}" if isinstance(h, np.int64) else f"{h:.2f}",
                            (p.get_x() * 1.005, p.get_height() * 1.005), fontsize=12)

        ax.set_xlabel('Treatment')
        ax_arr[0].set_ylabel('Rate')
        ax_arr[0].set_title('Treatment Distribution Rates')

        ax_arr[1].set_ylabel('# (Absolute)')
        ax_arr[1].set_title('Treatment Distribution - Absolute Quantities')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f"Threshold={thresh} - Treatment Recommendation Distribution\n{title_suffix}")

        return ax
