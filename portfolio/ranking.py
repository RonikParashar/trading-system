import pandas as pd

def rank_and_select(scores: dict,
                    tradable: dict,
                    top_n: int = 10) -> pd.DataFrame:
    """
    Cross-sectional ranking of stocks.

    Parameters
    ----------
    scores : dict[str, pd.Series]
        score series per ticker
    tradable : dict[str, pd.Series]
        tradable mask per ticker
    top_n : int
        number of stocks to select per day

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame: True if stock is selected on that date
    """

    # align all scores into one DataFrame
    score_df = pd.DataFrame(scores)
    tradable_df = pd.DataFrame(tradable)

    # invalid stocks get very low score
    score_df = score_df.where(tradable_df, -1e9)

    # daily ranks (higher score = better)
    ranks = score_df.rank(axis=1, ascending=False, method="first")

    # select top-N
    selected = ranks <= top_n

    return selected.fillna(False)
