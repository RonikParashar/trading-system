import pandas as pd
import numpy as np


def information_coefficient(preds, y):
    df = pd.concat([preds, y], axis=1).dropna()
    return df.groupby(level=0).apply(
        lambda x: x.iloc[:, 0].corr(x.iloc[:, 1])
    )


def rank_ic(preds, y):
    df = pd.concat([preds, y], axis=1).dropna()
    return df.groupby(level=0).apply(
        lambda x: x.iloc[:, 0].rank().corr(x.iloc[:, 1].rank())
    )
