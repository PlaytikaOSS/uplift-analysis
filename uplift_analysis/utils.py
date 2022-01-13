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
