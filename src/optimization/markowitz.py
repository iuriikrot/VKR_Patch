"""
Оптимизатор портфеля по Марковицу.
Максимизация коэффициента Шарпа.
"""

import numpy as np
from scipy.optimize import minimize


def maximize_sharpe(
    mu,
    cov,
    rf=0.02,
    min_weight=0.0,
    max_weight=1.0,
    long_only=True,
    fully_invested=True,
    gross_exposure=None
):
    """
    Максимизация коэффициента Шарпа.

    max (w'μ - rf) / sqrt(w'Σw)
    s.t. sum(w) = 1, w >= 0

    Args:
        mu: вектор ожидаемых доходностей (годовых)
        cov: ковариационная матрица (годовая)
        rf: безрисковая ставка (годовая, по умолчанию 2%)

    Returns:
        weights: оптимальные веса портфеля
    """
    n = len(mu)

    # Начальные веса — равные
    w0 = np.ones(n) / n

    # Функция для минимизации (отрицательный Шарп)
    def neg_sharpe(w):
        port_return = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        if port_vol < 1e-10:
            return 1e10  # Избегаем деления на ноль
        return -(port_return - rf) / port_vol

    if long_only and min_weight < 0:
        min_weight = 0.0

    # Ограничения
    constraints = []
    if fully_invested:
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # sum(w) = 1
    if not long_only and gross_exposure is not None:
        constraints.append({'type': 'ineq', 'fun': lambda w: gross_exposure - np.sum(np.abs(w))})

    # Границы весов
    bounds = [(min_weight, max_weight) for _ in range(n)]

    # Оптимизация
    result = minimize(
        neg_sharpe,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x


def portfolio_performance(weights, mu, cov, rf=0.02):
    """
    Расчёт характеристик портфеля.

    Args:
        weights: веса портфеля
        mu: ожидаемые доходности (годовые)
        cov: ковариационная матрица (годовая)
        rf: безрисковая ставка

    Returns:
        dict с return, volatility, sharpe
    """
    port_return = np.dot(weights, mu)
    port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0

    return {
        'return': port_return,
        'volatility': port_vol,
        'sharpe': sharpe
    }


if __name__ == "__main__":
    # Тест на простом примере
    mu = np.array([0.10, 0.12, 0.08])  # Ожидаемые доходности
    cov = np.array([
        [0.04, 0.01, 0.005],
        [0.01, 0.09, 0.02],
        [0.005, 0.02, 0.025]
    ])  # Ковариационная матрица

    weights = maximize_sharpe(mu, cov, rf=0.02)
    perf = portfolio_performance(weights, mu, cov, rf=0.02)

    print("Оптимальные веса:")
    print(weights.round(4))
    print(f"\nДоходность: {perf['return']:.2%}")
    print(f"Волатильность: {perf['volatility']:.2%}")
    print(f"Sharpe: {perf['sharpe']:.2f}")
