import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd

BASE_MIN_PERC = 0.001  # visualize metrics starting with this percentile to avoid noisy signals.


def visualize_selection_distribution(eval_res: pd.DataFrame, column_name: str = 'proposed_action') -> Axes:
    """
    Implements a utility function for displaying the selection distribution of multiple treatments, inferred according
    to outputs of an evaluated model.
    This visualization describes how selections diverge among the different treatments in a cumulative and gradual
    manner, as we lower the acceptance threshold (w.r.t the scores of the model).

    Parameters
    ----------
    eval_res: pd.DataFrame
        A dataframe of the evaluated dataset.
    column_name: str
        The name of the field containing the recommended action according to the evaluated model.

    Returns
    -------
    Axes
        The axes in which the visualization is created.
    """

    fig, ax = plt.subplots()

    cumulative = pd.get_dummies(eval_res.set_index('normalized_index')[column_name]).cumsum(axis=0)
    cumulative.plot(ax=ax)

    ax.set_xlabel('Fraction of population')
    ax.set_ylabel('No. Assignments per Action')
    ax.set_title('Cumulative Action assignment Distribution')
    ax.grid(True)

    return ax


def visualize_binary_uplift_metrics(eval_results, **kwargs):
    metrics_list = ['uplift_treated',
                    'gain_treated',
                    'expected_response_treated',
                    'relative_lift_treated']
    visualize_multiple_metrics(eval_results, metrics_list, **kwargs)


def visualize_multiple_actions_uplift_metrics(eval_results, **kwargs):
    metrics_list = ['uplift_intersection',
                    'gain_intersection',
                    'expected_response_intersect',
                    'relative_lift_intersect']
    visualize_multiple_metrics(eval_results, metrics_list, **kwargs)


def visualize_multiple_metrics(eval_results, metrics_list, **kwargs):
    for metric in metrics_list:
        visualize_metric(eval_results, metric, **kwargs)


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
    ax.legend()
    ax.set_xlabel('Fraction of the population exposed')
    ax.set_ylabel(metric)
    ax.set_title(metric)

    return ax


def single_curve_plot(df: pd.DataFrame, metric: str, ax, label=None):
    s = df.loc[df.normalized_index >= BASE_MIN_PERC].set_index('normalized_index')[metric]
    s.plot(ax=ax, lw=.5, label=label)
    return s


def visualize_fraction_of_action_data(eval_results, **kwargs):
    ax = visualize_metric(eval_results, 'frac_of_overall_treated', **kwargs)
    ax.plot([0, 1], [0, 1], linestyle='--', color='k', lw=.5, label=None)
    ax.set_ylabel('Fraction of action data')
    ax.set_title('Fraction of Action Data vs Score')
