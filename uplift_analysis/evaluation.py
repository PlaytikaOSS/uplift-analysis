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
    Its primary interface is the method ``evaluate_set()``.

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
            evaluation, summary = self.evaluate_set(input_set=set_to_evaluate)
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

    @staticmethod
    def integrate(x: Union[np.ndarray, pd.Series], dx: float):
        if isinstance(x, pd.Series):
            x = x.values
        if np.isnan(x).mean() < 1.0:
            x = x[:, np.newaxis]
            mask = np.logical_or(np.isnan(x), ~np.isfinite(x))
            idx = np.where(~mask, np.arange(mask.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            x[mask] = x[np.nonzero(mask)[0], idx[mask]]
            x = np.squeeze(x)
            return simps(x[np.logical_and(~np.isnan(x), np.isfinite(x))], dx=dx)
        else:
            return np.nan

    @staticmethod
    def get_max(eval_set: EvalSet, metric: str):
        idxmax = eval_set.df[metric].idxmax()
        return {
            'point': [
                eval_set.df['normalized_index'][idxmax],
                eval_set.df[metric].max()
            ],
            'value': eval_set.df[eval_set.score_field][idxmax]
        }

    def visualization_methods(self) -> Dict[str, Callable]:
        return {
            'uplift': self.display_uplift_curve,
            'fractional_lift': self.display_fractional_lift_curve,
            'gain': self.display_gain,
            'avg_response': self.display_avg_response,
            'targeted_region': self.display_targeted_region_stats,
            'untargeted_region': self.display_untargeted_region_stats,
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
        This method provides a set of charts for the description of the evaluation results.

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

        # make sure all the specified chart, if provided are listed
        if specify:
            assert all([elem in self.visualization_methods() for elem in specify]), \
                "Every element provided as part of `specify` argument must be listed under `visualization_methods()`"

        # generate random sets for benchmarking
        if show_random and num_sets == 1:
            random_res = self.create_random_sets(eval_res, num_random_rep)
        else:
            random_res = None

        # for each visualization method
        viz_methods = self.visualization_methods()
        for chart_name, func in viz_methods.items():
            if (specify is None) or (chart_name in specify):
                func(eval_res=eval_res,
                     num_sets=num_sets,
                     average=average,
                     title_suffix=title_suffix,
                     random_sets=random_res)

    def eval_and_show(self, data=Union[pd.DataFrame, EvalSet, Dict[str, Union[pd.DataFrame, EvalSet]]], **kwargs):
        if isinstance(data, dict):
            eval_res, summary = self.evaluate_multiple(data)
        else:
            eval_res, summary = self.evaluate_set(data)

        self.visualize(eval_res=eval_res, **kwargs)

        return eval_res, summary

    def create_random_sets(self, eval_res: Union[EvalSet, Dict[str, EvalSet]], num_random_rep: int) -> List[EvalSet]:
        if isinstance(eval_res, dict):
            _, eval_set = next(iter(eval_res))
        else:
            eval_set = eval_res

        num_samples = len(eval_set.df)
        # list possible actions
        action_set = list(
            set(eval_set.df[eval_set.observed_action_field].unique().tolist()) - set([eval_set.control_indicator]))

        random_sets: List[EvalSet] = []
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
                                   **kwargs):
        if not num_sets:
            average, num_sets = visualization.should_average(eval_res)

        fig, ax = plt.subplots()
        palette = itertools.cycle(sns.color_palette())

        if num_sets > 1:
            for (name, eval_set) in eval_res.items():
                s = eval_set.df[eval_set.score_field]
                sns.histplot(s, ax=ax, kde=True, bins=visualization.get_bin_quantity(s), stat='density', alpha=0.3,
                             label=name, color=next(palette))

            ax.set_xlabel('Uplift Score')

        else:
            if isinstance(eval_res, dict):  # a single entry dict
                name, eval_set = next(iter(eval_res))
            else:  # has to be an EvalSet
                name, eval_set = '', eval_res

            s = eval_set.df[eval_set.score_field]
            sns.histplot(s, ax=ax, kde=True, bins=visualization.get_bin_quantity(s), stat='density', alpha=0.3,
                         label='Overall', color=next(palette))

            s = eval_set.df.loc[
                eval_set.df[eval_set.observed_action_field] != eval_set.control_indicator, eval_set.score_field]
            sns.histplot(s, ax=ax, kde=True, bins=visualization.get_bin_quantity(s), stat='density', alpha=0.3,
                         label='Treated', color=next(palette))

            s = eval_set.df.loc[
                eval_set.df[eval_set.observed_action_field] == eval_set.control_indicator, eval_set.score_field]
            sns.histplot(s, ax=ax, kde=True, bins=visualization.get_bin_quantity(s), stat='density', alpha=0.3,
                         label='Untreated', color=next(palette))

            if eval_set.is_multiple_actions:
                for grp, sub_df in eval_set.df.groupby(eval_set.observed_action_field):
                    s = sub_df[eval_set.score_field]
                    sns.histplot(s, ax=ax, kde=True, bins=visualization.get_bin_quantity(s), stat='density', alpha=0.3,
                                 label=f'Treatment: {grp}', color=next(palette))

            ax.set_xlabel(eval_set.score_field)

        ax.grid(True)
        ax.legend(fancybox=True, shadow=True)
        ax.set_title(f"Score Distribution\n{title_suffix}")

    def display_uplift_curve(self,
                             eval_res: Union[Dict[str, EvalSet], EvalSet],
                             num_sets: Optional[Union[None, int]] = None,
                             average: Optional[bool] = False,
                             title_suffix: Optional[str] = '',
                             random_sets: Optional[Union[List[EvalSet], None]] = None,
                             **kwargs
                             ):
        if not num_sets:
            average, num_sets = visualization.should_average(eval_res, average)

        ax, lines = visualization.chart_display_template(eval_res=eval_res,
                                                         metric='uplift_intersection',
                                                         num_sets=num_sets,
                                                         average=average,
                                                         min_quantile=self.min_quantile)

        if num_sets > 1:
            # compute the area under curve for each EvalSet
            for (name, eval_set), line in zip(eval_res.items(), lines):
                auuc = self.integrate(x=eval_set.df['uplift_intersection'], dx=eval_set.get_quantile_interval())
                line.set_label(f"{name} (AUUC={auuc:.3f})")
            ax.legend(fancybox=True, shadow=True)
        else:  # add more information
            if isinstance(eval_res, dict):  # a single entry dict
                name, eval_set = next(iter(eval_res))
            else:  # has to be an EvalSet
                name, eval_set = '', eval_res

            # main signal
            dx = eval_set.get_quantile_interval()
            auuc = self.integrate(x=eval_set.df['uplift_intersection'], dx=dx)
            lines[0].set_label(f"Intersection Uplift (AUUC={auuc:.3f})")
            u = visualization.chop_lower_quantiles(eval_set.df.set_index('normalized_index')['uplift_intersection'],
                                                   q=self.min_quantile)
            visualization.area_fill(ax=ax, s=u, base=0., alpha=0.2)

            # in case randomization is required
            if random_sets is not None:
                s = np.stack([rand_res.df['uplift_intersection'].values for rand_res in random_sets], axis=-1).mean(
                    axis=1)
                auuc = self.integrate(x=s, dx=dx)
                visualization.single_curve_plot(signal=pd.Series(s, index=eval_set.df['normalized_index']),
                                                ax=ax, lw=1.5, min_quantile=self.min_quantile,
                                                label=f'Random (AUUC={auuc:.3f})', color='grey', linestyle='-.')

            # lines that are relevant only for multiple treatments
            if eval_set.is_multiple_actions:
                auuc = self.integrate(x=eval_set.df['uplift_treated'], dx=dx)
                visualization.single_curve_plot(signal=eval_set.df,
                                                metric='uplift_treated',
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                label=f'Treated Uplift (AUUC={auuc:.3f})')

                auuc = self.integrate(x=eval_set.df['uplift_against_unrealized'], dx=dx)
                visualization.single_curve_plot(signal=eval_set.df,
                                                metric='uplift_against_unrealized',
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                label=f'Realized Vs Unrealized Uplift (AUUC={auuc:.3f})')

            unexposed_diff = eval_set.df['below_response_control'] - eval_set.df['below_response_treated']
            unexposed_diff.index = eval_set.df['normalized_index']
            visualization.single_curve_plot(signal=unexposed_diff,
                                            ax=ax, lw=3, min_quantile=self.min_quantile,
                                            max_quantile=(1 - self.min_quantile),
                                            label=f'UnexposedResponseDiff')

            ax.set_ylabel('Uplift Estimate')
            ax.grid(True)
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
                                      ):
        if not num_sets:
            average, num_sets = visualization.should_average(eval_res, average)

        frac_lift: Callable = lambda df: df['above_response_intersect'] / df['above_response_control']
        ax, lines = visualization.chart_display_template(eval_res=eval_res,
                                                         metric='Fractional Lift',
                                                         num_sets=num_sets,
                                                         average=average,
                                                         min_quantile=self.min_quantile,
                                                         func=frac_lift)

        if num_sets > 1:
            # compute the area under curve for each EvalSet
            for (name, eval_set), line in zip(eval_res.items(), lines):
                auuc = self.integrate(x=frac_lift(eval_set.df), dx=eval_set.get_quantile_interval())
                line.set_label(f"{name} (AUUC={auuc:.3f})")
            ax.legend(fancybox=True, shadow=True)
        else:  # add more information
            if isinstance(eval_res, dict):  # a single entry dict
                name, eval_set = next(iter(eval_res))
            else:  # has to be an EvalSet
                name, eval_set = '', eval_res

            # main signal
            dx = eval_set.get_quantile_interval()
            u: pd.Series = frac_lift(eval_set.df)
            u.index = eval_set.df['normalized_index']
            auuc = self.integrate(x=u, dx=dx)
            lines[0].set_label(f"Fractional Uplift (AUUC={auuc:.3f})")
            u = visualization.chop_lower_quantiles(u, q=self.min_quantile)
            visualization.area_fill(ax=ax, s=u, base=1., alpha=0.2)

            # in case randomization is required
            if random_sets is not None:
                s = np.stack([frac_lift(rand_res.df).values for rand_res in random_sets], axis=-1).mean(
                    axis=1)
                auuc = self.integrate(x=s, dx=dx)
                visualization.single_curve_plot(signal=pd.Series(s, index=eval_set.df['normalized_index']),
                                                ax=ax, lw=1.5, min_quantile=self.min_quantile,
                                                label=f'Random (AUUC={auuc:.3f})', color='grey', linestyle='-.')

            # lines that are relevant only for multiple treatments
            if eval_set.is_multiple_actions:
                s = eval_set.df['above_response_treated'] / eval_set.df['above_response_control']
                s.index = eval_set.df['normalized_index']
                auuc = self.integrate(x=s, dx=dx)
                visualization.single_curve_plot(signal=s,
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                label=f'FracLift - Treated (AUUC={auuc:.3f})')

                s = eval_set.df['above_response_intersect'] / eval_set.df['above_response_unrealized']
                s.index = eval_set.df['normalized_index']
                auuc = self.integrate(x=s, dx=dx)
                visualization.single_curve_plot(signal=s,
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                label=f'FracLift Vs Unrealized Uplift (AUUC={auuc:.3f})')

            unexposed_diff = eval_set.df['below_response_control'] / eval_set.df['below_response_treated']
            unexposed_diff.index = eval_set.df['normalized_index']
            visualization.single_curve_plot(signal=unexposed_diff,
                                            ax=ax, lw=3, min_quantile=self.min_quantile,
                                            max_quantile=(1 - self.min_quantile),
                                            label=f'UnexposedResponseRatio')

            ax.set_ylabel('Uplift Estimate')
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
                     ):
        if not num_sets:
            average, num_sets = visualization.should_average(eval_res, average)

        ax, lines = visualization.chart_display_template(eval_res=eval_res,
                                                         metric='gain_intersection',
                                                         num_sets=num_sets,
                                                         average=average,
                                                         min_quantile=self.min_quantile)

        if num_sets == 1:
            if isinstance(eval_res, dict):  # a single entry dict
                name, eval_set = next(iter(eval_res))
            else:  # has to be an EvalSet
                name, eval_set = '', eval_res

            points = []

            lines[0].set_label(f"Intersection Gain")
            points.append(self.get_max(eval_set, metric='gain_intersection'))

            if eval_set.is_multiple_actions:
                visualization.single_curve_plot(signal=eval_set.df,
                                                metric='gain_treated',
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                label=f'Treated Gain')
                points.append(self.get_max(eval_set, metric='gain_treated'))

            if random_sets is not None:
                s = np.stack([rand_res.df['gain_intersection'].values for rand_res in random_sets], axis=-1).mean(
                    axis=1)
                visualization.single_curve_plot(signal=pd.Series(s, index=eval_set.df['normalized_index']),
                                                ax=ax, lw=1.5, min_quantile=self.min_quantile,
                                                label=f'Random', color='grey', linestyle='-.')

            visualization.plot_points(ax=ax, points=points, legend_prefix='Score=', value_key='value')

            ax.grid(True)
            ax.legend(fancybox=True, shadow=True)

        ax.set_xlabel('Sample Score Quantiles (Descending) / Exposed Fraction')
        ax.set_title(f"Gain Curve\n{title_suffix}")

        return ax

    def display_avg_response(self,
                             eval_res: Union[Dict[str, EvalSet], EvalSet],
                             num_sets: Optional[Union[None, int]] = None,
                             average: Optional[bool] = False,
                             title_suffix: Optional[str] = '',
                             random_sets: Optional[Union[List[EvalSet], None]] = None,
                             **kwargs
                             ):
        if not num_sets:
            average, num_sets = visualization.should_average(eval_res, average)

        ax, lines = visualization.chart_display_template(eval_res=eval_res,
                                                         metric='expected_response_intersect',
                                                         num_sets=num_sets,
                                                         average=average,
                                                         min_quantile=self.min_quantile)

        if num_sets == 1:
            if isinstance(eval_res, dict):  # a single entry dict
                name, eval_set = next(iter(eval_res))
            else:  # has to be an EvalSet
                name, eval_set = '', eval_res

            points = []

            lines[0].set_label(f"IntersectionExpectedResponse")
            points.append(self.get_max(eval_set, metric='expected_response_intersect'))

            if eval_set.is_multiple_actions:
                visualization.single_curve_plot(signal=eval_set.df,
                                                metric='expected_response_treated',
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                label=f'TreatedExpectedResponse')
                points.append(self.get_max(eval_set, metric='expected_response_treated'))

            if random_sets is not None:
                s = np.stack([rand_res.df['expected_response_intersect'].values for rand_res in random_sets],
                             axis=-1).mean(axis=1)
                visualization.single_curve_plot(signal=pd.Series(s, index=eval_set.df['normalized_index']),
                                                ax=ax, lw=1.5, min_quantile=self.min_quantile,
                                                label=f'Random', color='grey', linestyle='-.')

            if eval_set.is_binary_response:
                se_exposed = utils.get_standard_error_proportion(
                    sample_size=eval_set.df['intersect_count'],
                    proportion_estimate=eval_set.df['above_response_intersect'])
                se_unexposed = utils.get_standard_error_proportion(
                    sample_size=eval_set.df['control_count'].iloc[-1] - eval_set.df['control_count'],
                    proportion_estimate=eval_set.df['below_response_control'])
            else:
                exposed = eval_set.df['is_intersect'] > 0
                std_exposed = eval_set.df.loc[exposed, eval_set.response_field].expanding(
                    2).std(ddof=1).reindex(index=eval_set.df.index, method='ffill')
                se_exposed = utils.get_standard_error_mean(sample_size=eval_set.df['intersect_count'],
                                                           std=std_exposed)

                unexposed = eval_set.df[eval_set.observed_action_field] == eval_set.control_indicator
                unexposed_responses = eval_set.df.loc[unexposed, eval_set.response_field].iloc[::-1]
                std_unexposed = unexposed_responses.expanding(2).std(ddof=1).iloc[::-1].reindex(index=eval_set.df.index,
                                                                                                method='ffill')
                se_unexposed = utils.get_standard_error_mean(
                    sample_size=eval_set.df['control_count'].iloc[-1] - eval_set.df['control_count'],
                    std=std_unexposed)

            visualization.single_curve_plot(signal=eval_set.df,
                                            metric='above_response_intersect',
                                            ax=ax, lw=3,
                                            min_quantile=self.min_quantile,
                                            label=f'AvgResponseIntersectedTreatments')
            visualization.display_sleeve(ax=ax, eval_set=eval_set, metric='above_response_intersect',
                                         margin=se_exposed, color=ax.get_lines()[-1].get_color(),
                                         min_quantile=self.min_quantile * 2)

            visualization.single_curve_plot(signal=eval_set.df,
                                            metric='below_response_control',
                                            ax=ax, lw=3,
                                            min_quantile=self.min_quantile,
                                            max_quantile=(1 - self.min_quantile),
                                            label=f'AvgResponseUntreated')
            visualization.display_sleeve(ax=ax, eval_set=eval_set, metric='below_response_control',
                                         margin=se_unexposed, color=ax.get_lines()[-1].get_color(),
                                         max_quantile=(1 - 2 * self.min_quantile))

            visualization.plot_points(ax=ax, points=points, legend_prefix='Score=', value_key='value')

            where_untreated = eval_set.df[eval_set.observed_action_field] == eval_set.control_indicator
            ax.axhline(y=eval_set.df[eval_set.response_field].mean(), color='darkviolet', linestyle='--', lw=2,
                       label='OverallAvgResponse')
            ax.axhline(y=eval_set.df.loc[where_untreated, eval_set.response_field].mean(),
                       color='darkgoldenrod', linestyle=':', lw=2, label='UntreatedAvgResponse')

            ax.grid(True)
            ax.legend(fancybox=True, shadow=True)

        ax.set_xlabel('Sample Score Quantiles (Descending) / Exposed Fraction')
        ax.set_ylabel('AverageResponse Estimate')
        ax.set_title(f"Expected Response Estimate \n{title_suffix}")

        return ax

    def display_targeted_region_stats(self,
                                      eval_res: Union[Dict[str, EvalSet], EvalSet],
                                      num_sets: Optional[Union[None, int]] = None,
                                      average: Optional[bool] = False,
                                      title_suffix: Optional[str] = '',
                                      random_sets: Optional[Union[List[EvalSet], None]] = None,
                                      **kwargs
                                      ):
        if not num_sets:
            average, num_sets = visualization.should_average(eval_res, average)

        if num_sets > 1:
            ax, lines = visualization.chart_display_template(eval_res=eval_res,
                                                             metric='above_response_intersect',
                                                             num_sets=num_sets,
                                                             average=average,
                                                             min_quantile=self.min_quantile)
        else:
            fig, ax = plt.subplots()

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
                visualization.single_curve_plot(signal=eval_set.df,
                                                metric=metric_col,
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                label=label)
                if eval_set.is_binary_response:
                    se = utils.get_standard_error_proportion(
                        sample_size=eval_set.df[count_col],
                        proportion_estimate=eval_set.df[metric_col])
                else:
                    exposed = eval_set.df[indicator_col] > 0
                    std = eval_set.df.loc[exposed, eval_set.response_field].expanding(
                        2).std(ddof=1).reindex(index=eval_set.df.index, method='ffill')
                    stds[label] = std
                    se = utils.get_standard_error_mean(sample_size=eval_set.df[count_col],
                                                       std=std)

                visualization.display_sleeve(ax=ax, eval_set=eval_set, metric=metric_col,
                                             margin=se, color=ax.get_lines()[-1].get_color(),
                                             min_quantile=self.min_quantile * 2)

            where_untreated = eval_set.df[eval_set.observed_action_field] == eval_set.control_indicator
            ax.axhline(y=eval_set.df[eval_set.response_field].mean(), color='darkviolet', linestyle='--', lw=2,
                       label='OverallAvgResponse')
            ax.axhline(y=eval_set.df.loc[where_untreated, eval_set.response_field].mean(),
                       color='darkgoldenrod', linestyle=':', lw=2, label='UntreatedAvgResponse')

            # pvalue
            if eval_set.is_binary_response:
                pval = utils.proportions_test(sample_siz_1=eval_set.df['intersect_count'],
                                              sample_siz_2=eval_set.df['control_count'],
                                              proportion_est_1=eval_set.df['above_response_intersect'],
                                              proportion_est_2=eval_set.df['above_response_control'])
            else:
                pval = utils.t_test(mu_1=eval_set.df['above_response_intersect'],
                                    mu_2=eval_set.df['above_response_control'],
                                    sample_siz_1=eval_set.df['intersect_count'],
                                    sample_siz_2=eval_set.df['control_count'],
                                    std_1=stds['Intersections'],
                                    std_2=stds['Untreated']
                                    )
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
        ax.set_title(f"Targeted Region - Average Responses \n{title_suffix}")

        return ax

    def display_untargeted_region_stats(self,
                                        eval_res: Union[Dict[str, EvalSet], EvalSet],
                                        num_sets: Optional[Union[None, int]] = None,
                                        average: Optional[bool] = False,
                                        title_suffix: Optional[str] = '',
                                        random_sets: Optional[Union[List[EvalSet], None]] = None,
                                        **kwargs
                                        ):
        if not num_sets:
            average, num_sets = visualization.should_average(eval_res, average)

        if num_sets > 1:
            ax, lines = visualization.chart_display_template(eval_res=eval_res,
                                                             metric='below_response_control',
                                                             num_sets=num_sets,
                                                             average=average,
                                                             min_quantile=self.min_quantile,
                                                             max_quantile=(1 - self.min_quantile))
        else:
            fig, ax = plt.subplots()

            if isinstance(eval_res, dict):  # a single entry dict
                name, eval_set = next(iter(eval_res))
            else:  # has to be an EvalSet
                name, eval_set = '', eval_res

            metrics = ['below_response_control', 'below_response_treated']
            counts = ['control_count', 'treated_count']
            indicators = ['is_control', 'is_treated']
            labels = ['Untreated', 'Treated']
            stds, sample_sizes = dict(), dict()

            for metric_col, count_col, indicator_col, label in zip(metrics, counts, indicators, labels):
                visualization.single_curve_plot(signal=eval_set.df,
                                                metric=metric_col,
                                                ax=ax, lw=3,
                                                min_quantile=self.min_quantile,
                                                max_quantile=(1 - self.min_quantile),
                                                label=label)
                sample_sizes[label] = eval_set.df[count_col].iloc[-1] - eval_set.df[count_col]

                if eval_set.is_binary_response:
                    se = utils.get_standard_error_proportion(
                        sample_size=sample_sizes[label],
                        proportion_estimate=eval_set.df[metric_col])
                else:
                    unexposed = eval_set.df[indicator_col] > 0
                    unexposed_responses = eval_set.df.loc[unexposed, eval_set.response_field].iloc[::-1]
                    std_unexposed = unexposed_responses.expanding(2).std(ddof=1).iloc[::-1].reindex(
                        index=eval_set.df.index,
                        method='ffill')
                    stds[label] = std_unexposed
                    se = utils.get_standard_error_mean(
                        sample_size=sample_sizes[label],
                        std=std_unexposed)

                visualization.display_sleeve(ax=ax, eval_set=eval_set, metric=metric_col,
                                             margin=se, color=ax.get_lines()[-1].get_color(),
                                             min_quantile=self.min_quantile * 2,
                                             max_quantile=(1 - 2 * self.min_quantile))

            where_untreated = eval_set.df[eval_set.observed_action_field] == eval_set.control_indicator
            ax.axhline(y=eval_set.df[eval_set.response_field].mean(), color='darkviolet', linestyle='--', lw=2,
                       label='OverallAvgResponse')
            ax.axhline(y=eval_set.df.loc[where_untreated, eval_set.response_field].mean(),
                       color='darkgoldenrod', linestyle=':', lw=2, label='UntreatedAvgResponse')

            # pval
            if eval_set.is_binary_response:
                pval = utils.proportions_test(sample_siz_1=sample_sizes['Untreated'],
                                              sample_siz_2=sample_sizes['Treated'],
                                              proportion_est_1=eval_set.df['below_response_control'],
                                              proportion_est_2=eval_set.df['below_response_treated'])
            else:
                pval = utils.t_test(mu_1=eval_set.df['below_response_control'],
                                    mu_2=eval_set.df['below_response_treated'],
                                    sample_siz_1=sample_sizes['Untreated'],
                                    sample_siz_2=sample_sizes['Treated'],
                                    std_1=stds['Untreated'],
                                    std_2=stds['Treated']
                                    )
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
        ax.set_title(f"UnTargeted Region - Average Responses \n{title_suffix}")

        return ax

    def display_agreement_stats(self,
                                eval_res: Union[Dict[str, EvalSet], EvalSet],
                                num_sets: Optional[Union[None, int]] = None,
                                average: Optional[bool] = False,
                                title_suffix: Optional[str] = '',
                                random_sets: Optional[Union[List[EvalSet], None]] = None,
                                **kwargs
                                ):
        if not num_sets:
            average, num_sets = visualization.should_average(eval_res, average)

        fig, ax = plt.subplots()
        decision_agreement = lambda es: (es.df[es.proposed_action_field] == es.df[es.observed_action_field]).astype(int)

        if num_sets > 1:
            series_dict = dict()
            for name, single_res in eval_res.items():
                agreement_rate = decision_agreement(single_res)
                agreement_rate.index = single_res.df['normalized_index']
                series_dict[name], line = visualization.single_curve_plot(agreement_rate.expanding().mean(),
                                                                          ax=ax, lw=1,
                                                                          label=name,
                                                                          min_quantile=self.min_quantile)
            if average:
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
                s = np.stack([decision_agreement(rand_res).expanding().mean().values for rand_res in random_sets],
                             axis=-1).mean(axis=1)
                visualization.single_curve_plot(signal=pd.Series(s, index=eval_set.df['normalized_index']),
                                                ax=ax, lw=1.5, min_quantile=self.min_quantile,
                                                label=f'Random', color='grey', linestyle='-.')

            # lines that are relevant only for multiple treatments
            if eval_set.is_multiple_actions:
                u = np.logical_and(eval_set.df[eval_set.proposed_action_field] != eval_set.control_indicator,
                                   eval_set.df[eval_set.observed_action_field] != eval_set.control_indicator).astype(
                    int).expanding().mean()
                u.index = eval_set.df['normalized_index']
                visualization.single_curve_plot(u,
                                                ax=ax, lw=3,
                                                label='BinaryAgreementRate',
                                                min_quantile=self.min_quantile)

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
        ax.set_ylabel('Rate of Intersections (Among Targeted)')
        ax.set_title(f"Agreement/Intersection Statistics\n{title_suffix}")
        return ax

    def get_distribution_by_threshold(self,
                                      eval_res: Union[Dict[str, EvalSet], EvalSet],
                                      thresh: float,
                                      title_suffix: Optional[str] = ''):

        assert isinstance(eval_res, EvalSet), "The input ``eval_res`` must be of type EvalSet"
        assert eval_res.df[eval_res.score_field].min() <= thresh <= eval_res.df[eval_res.score_field].max(), \
            "The provided threshold does not lie within the range of scores"

        fig, ax_arr = plt.subplots(nrows=2)

        observed = eval_res.df[eval_res.observed_action_field]

        recommendations = eval_res.df[eval_res.proposed_action_field].copy()
        recommendations[eval_res.df[eval_res.score_field] >= thresh] = eval_res.control_indicator

        intersected_recommendations = recommendations.loc[recommendations == observed]

        for norm_bool, ax in zip([True, False], ax_arr):
            distribs = pd.merge(observed.value_counts(normalize=norm_bool).rename('Observed'),
                                recommendations.value_counts(normalize=norm_bool).rename('Recommended'),
                                left_index=True, right_index=True)
            distribs = pd.merge(distribs,
                                intersected_recommendations.value_counts(normalize=norm_bool).rename('Intersection'),
                                left_index=True, right_index=True)

            distribs.sort_index().plot(kind='bar', ax=ax)
            ax.grid(True)
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
