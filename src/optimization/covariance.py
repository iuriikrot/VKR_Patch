"""
Оценка ковариации для портфельной оптимизации.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def compute_covariance(returns, method="sample", annualize=252):
    """
    Рассчитать ковариационную матрицу.

    Args:
        returns: DataFrame с доходностями (дневные лог-доходности)
        method: 'sample' или 'ledoit_wolf'
        annualize: множитель для годового масштаба

    Returns:
        np.ndarray ковариационная матрица
    """
    if method == "sample":
        cov = returns.cov().values
    elif method == "ledoit_wolf":
        lw = LedoitWolf().fit(returns.values)
        cov = lw.covariance_
    else:
        raise ValueError(f"Неизвестный метод ковариации: {method}")

    return cov * annualize
