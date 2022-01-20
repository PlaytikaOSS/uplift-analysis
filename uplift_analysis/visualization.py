# -*- coding: utf-8 -*-
"""
This module implements functional utilities which are helpful for visualizing the evaluation and analysis results.
"""

from typing import Union, Dict, Optional, Tuple, List, Callable
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from .data import EvalSet

BASE_MIN_QUANTILE = 0.001  # the default quantile from which the quantile-dependent signal will be plotted
BASE_MAX_QUANTILE = 1.0  # the default quantile up to  which the quantile-dependent signal will be plotted


def visualize_selection_distribution(eval_res: Union[pd.DataFrame, EvalSet],
                                     column_name: Optional[Union[str, None]] = None) -> Axes:
    """
    Implements a utility function for displaying the selection distribution of multiple treatments, inferred according
    to outputs of an evaluated model.
    This visualization describes how selections diverge among the different treatments in a cumulative and gradual
    manner, as we lower the acceptance threshold (w.r.t the scores of the model).
    The x-axis corresponds to the quantile of scores, in a descending manner, where each progression towards the right
    implies including a wider upper quantile. The y-axis corresponds to the number of appearances for each treatment,
    i.e. the number of *decisions* made by the model to recommend each treatment.

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

    # get the cumulative count of each unique entity in the designated column
    cumulative = pd.get_dummies(eval_res.set_index('normalized_index')[column_name]).cumsum(axis=0)
    cumulative.plot(ax=ax)

    ax.set_xlabel('Fraction of population')
    ax.set_ylabel('No. Assignments per Action')
    ax.set_title('Cumulative Action assignment Distribution')
    ax.grid(True)

    return ax


def chart_display_template(eval_res: Union[Dict[str, EvalSet], EvalSet],
                           metric: str,
                           func: Optional[Union[Callable, None]] = None,
                           num_sets: Optional[Union[None, int]] = None,
                           average: Optional[bool] = False,
                           min_quantile: Optional[Union[float, None]] = None,
                           max_quantile: Optional[Union[float, None]] = None,
                           ) -> Tuple[Axes, List[Line2D]]:
    """
    Implements a utility function called by ``evaluation.Evaluator``, for handling the required abstraction
    for plotting signals from multiple/signal ``data.EvalSet`` objects.

    Parameters
    ----------
    eval_res: Union[Dict[str, EvalSet], EvalSet]
        A single ``data.EvalSet`` object, or a collection of such, containing the signal(s) to be plotted.
    metric: str
        The name of the signal. Generally, corresponds to a column on the dataframe hosted in the ``EvalSet`` object.
    func: Optional[Union[Callable, None]]
        In cases where the required signal is not part of the mentioned dataframe, the caller can provide a ``Callable``
        which will be responsible for generating a custom ``pd.Series``.
    num_sets: Optional[Union[None, int]]
        The number of ``EvalSet`` objects the input argument ``eval_res`` holds.
    average: Optional[bool]
        A boolean indicating whether averaging of the signals is required. Relevant only in case ``num_sets > 1``.
    min_quantile: Optional[Union[float, None]]
        The quantile from which the quantile-dependent signal(s) will be plotted, for avoiding the noise on the edge
        of the signal.
    max_quantile
        The quantile up to which the quantile-dependent signal(s) will be plotted, for avoiding the noise on the edge
        of the signal.

    Returns
    -------
    ax: Axes
        The axes object on which the lines were plotted.
    lines: List[Line2D]
        The list of ``Line2D`` objects plotted by the function, for further editing, if required.

    """

    if not num_sets:  # if not provided, infer independently
        average, num_sets = should_average(eval_res, average)

    # set wider line-width in case there is only a single ``EvalSet`` object
    lw = 1 if num_sets > 1 else 3

    fig, ax = plt.subplots()
    lines = []  # will be populated by the ``Line2D`` objects created

    if isinstance(eval_res, EvalSet):  # a single ``EvalSet`` object
        _, line = single_curve_plot(eval_res.df, ax=ax, metric=metric, lw=lw, label=eval_res.name or metric,
                                    min_quantile=min_quantile, max_quantile=max_quantile,
                                    func=func)
        lines.append(line)
    else:  # a dict of ``EvalSet`` objects
        series_dict = dict()  # a mapping of the plotted signals
        for name, single_res in eval_res.items():  # for each ``EvalSet`` object
            series_dict[name], line = single_curve_plot(single_res.df, ax=ax, metric=metric, lw=lw, label=name,
                                                        min_quantile=min_quantile, max_quantile=max_quantile,
                                                        func=func)
            lines.append(line)

        if average:
            # a new dataframe will interpolate the accumulated signals, and compute
            # the average of them, for each quantile
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
    """
    A visualization utility, employed by ``evaluation.Evaluator`` for filling areas under curves.
    The colors of the fill varies according to some configurable horizontal line, ``base``.
    Areas above ``base`` will be filled with green color, and ares below ``base`` will be filled with
    red color.

    Parameters
    ----------
    ax: Axes
        The axes object on which the visualization will be appended.
    s: pd.Series
        The curve, according to which an area will be filled.
    base: Union[int, float]
        The y-axis value of the horizontal line according to which the area will be colored.
    kwargs:
        Arbitrary keyword arguments, which will be sent to matplotlib backend.

    """
    ax.fill_between(s.index.values, base, s, where=s >= base, color='green', interpolate=True, **kwargs)
    ax.fill_between(s.index.values, base, s, where=s < base, color='red', interpolate=True, **kwargs)


def emphasize_axes(ax: Axes, base: Tuple[Union[float, int], Union[float, int]]) -> None:
    """
    A visualization utility, employed by ``evaluation.Evaluator`` for emphasizing axes,
    by plotting bold lines on designated coordinates.

    Parameters
    ----------
    ax: Axes
        The axes object on which the visualization will be appended.
    base: Tuple[Union[float, int]
        The tuple of values defining the axes, (x_value,y_value).
    """

    ax.axvline(x=base[0], lw=2, color='k', label=None)
    ax.axhline(y=base[1], lw=2, color='k', label=None)


def plot_points(ax: Axes, points: List[Dict], legend_prefix: str, value_key: Optional[str] = 'value') -> None:
    """
    A visualization utility, employed by ``evaluation.Evaluator`` for plotting a set of independet markers,
    labeled by their corresponding values.

    Parameters
    ----------
    ax: Axes
        The axes object on which the visualization will be appended.
    points: List[Dict]
        A list of dictionaries, containing the points to be plotted. Each dictionary must contain a key ``point``, for
        which the corresponding value will be a 2D tuple, containing the coordinates of the point.
        It also  must contain a key corresponding to the argument ``value_key``. The value of this key will be used for
        labeling this point.
    legend_prefix: str
        The string which will be added to the beginning of the label of each plotted point, followed by the
        corresponding value.
    value_key: Optional[str]
        The name of the key, in each of the ``dict`` objects listed under ``points``, in which the value of the point
        is stored. This value will be used for labeling the point.
    """

    for point, color in zip(points, mcolors.TABLEAU_COLORS):
        ax.plot(point['point'][0], point['point'][1], marker='o', markersize=10, markeredgecolor='k', color=color,
                label=f"{legend_prefix}{point[value_key]:.3f}")


def display_sleeve(ax: Axes,
                   eval_set: EvalSet,
                   metric: str,
                   margin: pd.Series,
                   color: str,
                   min_quantile: Optional[float] = BASE_MIN_QUANTILE,
                   max_quantile: Optional[float] = BASE_MAX_QUANTILE) -> None:
    """
    A visualization utility, employed by ``evaluation.Evaluator`` for displaying an uncertainty sleeve around a
    specified line.

    Parameters
    ----------
    ax: Axes
        The axes object on which the visualization will be appended.
    eval_set: EvalSet
        The ``EvalSet`` object containing the signal around which the sleeve will be visualized.
    metric: str
        The name of the signal, corresponding to the column in the dataframe of the provided ``eval_set``. This will
        be the reference signal, around which the sleeve will be visualized.
    margin: pd.Series
        The one-sided margin, which will be added, and subtracted from the reference signal, for creating the sleeve.
    color: str
        The color of the sleeve.
    min_quantile:
        The quantile from which the quantile-dependent signal(s) will be plotted, for avoiding the noise on the edge
        of the signal.
    max_quantile
        The quantile up to which the quantile-dependent signal(s) will be plotted, for avoiding the noise on the edge
        of the signal.
    """

    # compute lower and upper bounds of the sleeve, and index them accordingly
    lb = eval_set.df[metric] - margin  # lower bound
    ub = eval_set.df[metric] + margin  # upper bound
    lb.index = eval_set.df['normalized_index']
    ub.index = eval_set.df['normalized_index']

    # use ``min_quantile`` and ``max_quantile`` for chopping the bounding signals
    lb = chop_lower_quantiles(lb, q=min_quantile)
    ub = chop_lower_quantiles(ub, q=min_quantile)
    if max_quantile is not None and max_quantile != BASE_MAX_QUANTILE:
        lb = chop_upper_quantiles(lb, q=max_quantile)
        ub = chop_upper_quantiles(ub, q=max_quantile)

    # color the area in between
    ax.fill_between(lb.index.values, lb, ub, facecolor=color, alpha=0.2, label=None)


def single_curve_plot(signal: Union[pd.DataFrame, pd.Series],
                      ax: Axes,
                      metric: Optional[Union[str, None]] = None,
                      func: Optional[Union[Callable, None]] = None,
                      **kwargs) -> Tuple[pd.Series, Line2D]:
    """
    A visualization utility for plotting a single curve, on a given ``Axes`` object.
    The signal can be an independent ``pandas.Series``, but it also can be a column of a ``pandas.Datframe``, or a
    result of a function applied to an input ``pandas.Dataframe``, for creating a new ``pandas.Series``.

    Parameters
    ----------
    signal: Union[pd.DataFrame, pd.Series]
        The input data according to which the curve will be plotted.
    ax: Axes
        The axes object on which the visualization will be appended.
    metric: Optional[Union[str, None]]
        The name of the column holding the desired signal, on the input dataframe. Irrelevant when ``signal`` is a
        ``pandas.Series`` object.
    func: Optional[Union[Callable, None]]
        A function to apply on the input dataframe, for generating a new ``pandas.Series``. Irrelevant when ``signal``
        is a ``pandas.Series`` object.
    kwargs
        Arbitrary keyword arguments.
    Returns
    -------
    pd.Series
        The series which was eventually plotted by the function.
    Line2D
        The plotted line object, for further manipulation.
    """

    assert metric is not None or isinstance(signal, pd.Series), \
        "If the input argument ``metric`` is not provided, ``signal`` must be a ``pandas.Series`` object."

    if isinstance(signal, pd.DataFrame):
        assert metric is not None or func is not None, \
            "In case the input `signal` is a ``pandas.DataFrame`` object, either ``metric`` or ``func`` must provided."

        # if a callable function was provided - apply it and index the resulting series
        if func is not None:
            s = func(signal)
            s.index = signal['normalized_index']
        else:  # metric must be provided
            s = signal.set_index('normalized_index')[metric]
    else:  # if it's not a dataframe, it must be a series
        s = signal

    # lose lower quantiles, to avoid noisy line
    s = chop_lower_quantiles(s=s, q=kwargs.pop('min_quantile', BASE_MIN_QUANTILE))

    # if max_quantile is provided, lose also upper quantiles, for the same reason
    max_quantile = kwargs.pop('max_quantile', None)
    if max_quantile and max_quantile != BASE_MAX_QUANTILE:
        s = chop_upper_quantiles(s=s, q=max_quantile)

    # display the line, while ignoring problematic values
    s[~s.isin([np.nan, np.inf, -np.inf])].plot(ax=ax, lw=kwargs.pop('lw', .5), **kwargs)

    # return the chopped series, and the plotted line
    return s, ax.get_lines()[-1]


def chop_lower_quantiles(s: pd.Series, q: Optional[float] = None) -> pd.Series:
    """
    A utility function for filtering out lower quantiles of a given signal, represented as a ``pandas.Series``,
    and indexed by quantiles in a descending manner. This is done for avoiding noisy estimations which might
    occur in lower quantiles.

    Parameters
    ----------
    s: pd.Series
        The original ``pandas.Series``, indexed by quantiles in a descending manner.
    q: Optional[float]
        The quantile up to which value of the original series will be filtered out.

    Returns
    -------
    pd.Series
        The series after filtering out the specified quantiles.
    """
    return s.loc[s.index > (q or BASE_MIN_QUANTILE)]


def chop_upper_quantiles(s: pd.Series, q: float = None) -> pd.Series:
    """
    A utility function for filtering out upper quantiles of a given signal, represented as a ``pandas.Series``,
    and indexed by quantiles in a descending manner. This is done for avoiding noisy estimations which might
    occur in upper quantiles.

    Parameters
    ----------
    s: pd.Series
        The original ``pandas.Series``, indexed by quantiles in a descending manner.
    q: Optional[float]
        The quantile from which value of the original series will be filtered out.

    Returns
    -------
    pd.Series
        The series after filtering out the specified quantiles.
    """
    return s.loc[s.index <= (q or BASE_MAX_QUANTILE)]


def get_bin_quantity(s: pd.Series, max_bins: Optional[int] = 200, bin_rate: Optional[float] = .05) -> int:
    """
    A utility function for retrieving the number of bins to use for a density plot or an histogram.
    This number will be based on the number of unique values from which the signal will be composed, and bounded
    from above by a configured maximal value.

    Parameters
    ----------
    s: pd.Series
        The series for which a denisty plot is required.
    max_bins: Optional[int]
        The maximal number of bins to allow, regardless of the number of distinct values.
    bin_rate: Optional[float]
        The fraction between the number of bins and the number of distinct values in the input series.

    Returns
    -------
    pd.Series
        The number of bins to use in a denisty plot / histogram.
    """

    return min(max_bins, int(bin_rate * s.nunique()))


def should_average(eval_res: Union[Dict[str, EvalSet], EvalSet], average: Optional[bool] = False):
    """
    A function for determining the number of ``EvalSet`` objects in the input ``eval_res``, and accordingly determine if
    averaging is required.

    Parameters
    ----------
    eval_res: Union[Dict[str, EvalSet], EvalSet]
        The input data, which might contain single or multiple ``EvalSet`` objects.
    average: Optional[bool]
        The upper setting of the demand for averaging.

    Returns
    -------
    average: bool
        Averaging is required (``True``) only if the input ``average=True``, and ``num_sets > 1``.
    num_sets: int
        The number of ``EvalSet`` objects in the input ``eval_res``.
    """
    num_sets = len(eval_res) if isinstance(eval_res, Dict) else 1
    # average will be true only if initially provided as such *and* if ``num_sets`` > 1
    average = average if num_sets > 1 else False

    return average, num_sets
