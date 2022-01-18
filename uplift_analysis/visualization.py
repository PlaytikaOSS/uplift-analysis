from typing import Union, Dict, Optional, Tuple, List, Callable
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from .data import EvalSet

BASE_MIN_QUANTILE = 0.001  # visualize metrics starting with this percentile to avoid noisy signals.
BASE_MAX_QUANTILE = 1.0  # visualize metrics finishing with this percentile to avoid noisy signals.


def visualize_selection_distribution(eval_res: Union[pd.DataFrame, EvalSet],
                                     column_name: Optional[Union[str, None]] = None) -> Axes:
    """
    Implements a utility function for displaying the selection distribution of multiple treatments, inferred according
    to outputs of an evaluated model.
    This visualization describes how selections diverge among the different treatments in a cumulative and gradual
    manner, as we lower the acceptance threshold (w.r.t the scores of the model).

    Parameters
    ----------
    eval_res: Union[pd.DataFrame, EvalSet]
        The evaluated dataset, whether it is represented by a dataframe or an EvalSet object.
    column_name: Optional[Union[str, None]]
        The name of the field containing the recommended action according to the evaluated model.
        If not provided, inferred according to the input EvalSet.

    Returns
    -------
    Axes
        The axes in which the visualization is created.
    """
    if isinstance(eval_res, EvalSet):
        assert eval_res.is_evaluated, "The provided dataset must be evaluated first."
        column_name = column_name or eval_res.proposed_action_field
        eval_res = eval_res.df
    elif isinstance(eval_res, pd.DataFrame):
        assert 'normalized_index' in list(eval_res.columns), "The provided dataset must be evaluated first."
        assert column_name, "When providing dataframe, the argument `column_name` must be specified."
    else:
        raise TypeError("The provided dataset must be a Pandas DataFrame or an EvalSet object.")

    fig, ax = plt.subplots()

    cumulative = pd.get_dummies(eval_res.set_index('normalized_index')[column_name]).cumsum(axis=0)
    cumulative.plot(ax=ax)

    ax.set_xlabel('Fraction of population')
    ax.set_ylabel('No. Assignments per Action')
    ax.set_title('Cumulative Action assignment Distribution')
    ax.grid(True)

    return ax


def chart_display_template(eval_res: Union[Dict[str, EvalSet], EvalSet],
                           metric: str,
                           func: Union[Callable, None] = None,
                           num_sets: Optional[Union[None, int]] = None,
                           average: Optional[bool] = False,
                           min_quantile: Optional[Union[float, None]] = None,
                           max_quantile: Optional[Union[float, None]] = None,
                           ) -> Tuple[Axes, List[Line2D]]:
    if not num_sets:
        average, num_sets = should_average(eval_res, average)
    lw = 1 if num_sets > 1 else 3

    fig, ax = plt.subplots()
    lines = []

    if isinstance(eval_res, EvalSet):
        _, line = single_curve_plot(eval_res.df, ax=ax, metric=metric, lw=lw, label=eval_res.name or metric,
                                    min_quantile=min_quantile, max_quantile=max_quantile,
                                    func=func)
        lines.append(line)
    else:
        series_dict = dict()
        for name, single_res in eval_res.items():
            series_dict[name], line = single_curve_plot(single_res.df, ax=ax, metric=metric, lw=lw, label=name,
                                                        min_quantile=min_quantile, max_quantile=max_quantile,
                                                        func=func)
            lines.append(line)

        if average:
            combined_df = pd.DataFrame(series_dict)
            combined_df = combined_df.interpolate('index')
            aggregated = combined_df.mean(axis=1)
            aggregated.plot(ax=ax, lw=3, label='Avg', color='k', alpha=0.5)

    ax.grid(True)
    ax.legend(fancybox=True, shadow=True)
    ax.set_xlabel('Sample Score Quantiles (Descending) / Exposed Fraction')
    ax.set_ylabel(metric)
    ax.set_title(metric)

    return ax, lines


def area_fill(ax: Axes, s: pd.Series, base: Union[int, float] = 0., **kwargs) -> None:
    ax.fill_between(s.index.values, base, s, where=s >= base, color='green', interpolate=True, **kwargs)
    ax.fill_between(s.index.values, base, s, where=s < base, color='red', interpolate=True, **kwargs)


def emphasize_axes(ax: Axes, base: Tuple[Union[float, int], Union[float, int]]) -> None:
    ax.axvline(x=base[0], lw=2, color='k', label=None)
    ax.axhline(y=base[1], lw=2, color='k', label=None)


def plot_points(ax: Axes, points: List[Dict], legend_prefix: str, value_key: str = 'value') -> None:
    for point, color in zip(points, mcolors.TABLEAU_COLORS):
        ax.plot(point['point'][0], point['point'][1], marker='o',markersize=10, markeredgecolor='k', color=color,
                label=f"{legend_prefix}{point[value_key]:.3f}")


def display_sleeve(ax: Axes, eval_set: EvalSet, metric: str, margin: pd.Series, color: str,
                   min_quantile: float = BASE_MIN_QUANTILE, max_quantile: float = BASE_MAX_QUANTILE) -> None:
    lb = eval_set.df[metric] - margin
    ub = eval_set.df[metric] + margin
    lb.index = eval_set.df['normalized_index']
    ub.index = eval_set.df['normalized_index']

    lb = chop_lower_quantiles(lb, q=min_quantile)
    ub = chop_lower_quantiles(ub, q=min_quantile)
    if max_quantile is not None and max_quantile != BASE_MAX_QUANTILE:
        lb = chop_upper_quantiles(lb, q=max_quantile)
        ub = chop_upper_quantiles(ub, q=max_quantile)

    ax.fill_between(lb.index.values, lb, ub, facecolor=color, alpha=0.2, label=None)


def visualize_metric(eval_results, metric, **kwargs):
    fig, ax = plt.subplots()

    # in case of a single results set
    if isinstance(eval_results, pd.DataFrame):
        s = single_curve_plot(eval_results, metric, ax)
    else:
        if isinstance(eval_results, list):
            series_list = []
            for single_res in eval_results:
                series_list.append(single_curve_plot(single_res, metric, ax))

            combined_df = pd.concat(series_list, axis=1)
        elif isinstance(eval_results, dict):
            series_dict = dict()
            for name, single_res in eval_results.items():
                series_dict[name] = single_curve_plot(single_res, metric, ax, label=name)
            combined_df = pd.DataFrame(series_dict)
        else:
            raise ValueError

        avg = kwargs.get('avg', False)
        if avg:
            combined_df = combined_df.interpolate('index')
            aggregated = combined_df.mean(axis=1)
            aggregated.plot(ax=ax, lw=3, label='Avg', color='k', alpha=0.5)

    ax.grid(True)
    ax.legend(fancybox=True, shadow=True)
    ax.set_xlabel('Fraction of the population exposed')
    ax.set_ylabel(metric)
    ax.set_title(metric)

    return ax


def single_curve_plot(signal: Union[pd.DataFrame, pd.Series],
                      ax: Axes,
                      metric: Union[str, None] = None,
                      func: Union[Callable, None] = None,
                      **kwargs) -> Tuple[pd.Series, Line2D]:
    assert metric is not None or isinstance(signal, pd.Series)
    if isinstance(signal, pd.DataFrame):
        assert metric is not None or func is not None
        if func is not None:
            s = func(signal)
            s.index = signal['normalized_index']
        else:
            s = signal.set_index('normalized_index')[metric]
    else:
        s = signal

    s = chop_lower_quantiles(s=s, q=kwargs.pop('min_quantile', BASE_MIN_QUANTILE))
    max_quantile = kwargs.pop('max_quantile', None)
    if max_quantile and max_quantile != BASE_MAX_QUANTILE:
        s = chop_upper_quantiles(s=s, q=max_quantile)
    s[~s.isin([np.nan, np.inf, -np.inf])].plot(ax=ax, lw=kwargs.pop('lw', .5), **kwargs)
    return s, ax.get_lines()[-1]


def chop_lower_quantiles(s: pd.Series, q: float = None):
    return s.loc[s.index > (q or BASE_MIN_QUANTILE)]


def chop_upper_quantiles(s: pd.Series, q: float = None):
    return s.loc[s.index <= (q or BASE_MAX_QUANTILE)]


def get_bin_quantity(s: pd.Series, max_bins: int = 200, bin_rate: float = .05) -> int:
    return min(max_bins, bin_rate * s.nunique())


def should_average(eval_res: Union[Dict[str, EvalSet], EvalSet], average: Optional[bool] = False):
    num_sets = len(eval_res) if isinstance(eval_res, Dict) else 1
    average = average if num_sets > 1 else False

    return average, num_sets
