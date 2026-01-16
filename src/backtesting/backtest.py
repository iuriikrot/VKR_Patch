"""
Бэктест классического Марковица со скользящим окном.
Baseline 1: μ = историческое среднее
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

# Добавляем путь к src
sys.path.append(str(Path(__file__).parent.parent))

from optimization.markowitz import maximize_sharpe
from optimization.covariance import compute_covariance

# Загружаем конфигурацию
config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Параметры из config
TRAIN_WINDOW = config['backtest']['train_window']  # 1260 дней (5 лет)
TEST_WINDOW = config['backtest']['test_window']    # 21 день (1 месяц)
RF = config['optimization']['risk_free_rate']      # 0.02
CV_METHOD = config['optimization'].get('covariance', 'sample')
CONSTRAINTS = config.get('optimization', {}).get('constraints', {})
MIN_WEIGHT = CONSTRAINTS.get('min_weight', 0.0)
MAX_WEIGHT = CONSTRAINTS.get('max_weight', 1.0)
LONG_ONLY = CONSTRAINTS.get('long_only', True)
FULLY_INVESTED = CONSTRAINTS.get('fully_invested', True)
GROSS_EXPOSURE = CONSTRAINTS.get('gross_exposure')


def run_backtest(returns, save_weights_path=None):
    """
    Запуск бэктеста со скользящим окном.

    Args:
        returns: DataFrame с лог-доходностями

    Returns:
        portfolio_returns: Series с доходностями портфеля
    """
    n = len(returns)
    portfolio_returns = []
    dates = []
    weights_list = [] if save_weights_path else None

    print(f"Всего дней: {n}")
    print(f"Train окно: {TRAIN_WINDOW} дней (5 лет)")
    print(f"Test окно: {TEST_WINDOW} дней (1 месяц)")

    step = 0
    i = 0

    while i + TRAIN_WINDOW + TEST_WINDOW <= n:
        # Train данные
        train_data = returns.iloc[i:i + TRAIN_WINDOW]

        # Test данные
        test_data = returns.iloc[i + TRAIN_WINDOW:i + TRAIN_WINDOW + TEST_WINDOW]

        # Считаем μ и Σ на train (годовые)
        mu = train_data.mean() * 252
        cov = compute_covariance(train_data, method=CV_METHOD, annualize=252)

        # Оптимизируем веса
        weights = maximize_sharpe(
            mu.values,
            cov,
            rf=RF,
            min_weight=MIN_WEIGHT,
            max_weight=MAX_WEIGHT,
            long_only=LONG_ONLY,
            fully_invested=FULLY_INVESTED,
            gross_exposure=GROSS_EXPOSURE
        )

        # Доходность портфеля на test (ребалансировка раз в месяц)
        asset_gross = np.exp(test_data.sum(axis=0).values)
        portfolio_gross = np.dot(weights, asset_gross)
        if not FULLY_INVESTED:
            portfolio_gross += (1 - weights.sum())
        month_return = np.log(portfolio_gross)

        portfolio_returns.append(month_return)
        dates.append(test_data.index[0])
        if weights_list is not None:
            weights_list.append(weights)

        step += 1
        i += TEST_WINDOW  # Сдвигаем на месяц

        if step % 20 == 0:
            print(f"Шаг {step}: {test_data.index[0].date()}")

    print(f"\nВсего периодов: {len(portfolio_returns)}")

    if weights_list is not None:
        weights_df = pd.DataFrame(weights_list, index=dates, columns=returns.columns)
        weights_df.to_csv(save_weights_path)

    return pd.Series(portfolio_returns, index=dates)


def calculate_metrics(returns, rf=0.02):
    """
    Расчёт метрик портфеля.

    Args:
        returns: Series с месячными лог-доходностями
        rf: безрисковая ставка (годовая)

    Returns:
        dict с метриками
    """
    # Переводим в простые доходности для корректного расчёта
    simple_returns = np.exp(returns) - 1
    monthly_rf = (1 + rf) ** (1 / 12) - 1
    excess = simple_returns - monthly_rf

    # Годовая доходность (CAGR)
    if len(simple_returns) > 0:
        annual_return = (1 + simple_returns).prod() ** (12 / len(simple_returns)) - 1
    else:
        annual_return = 0

    # Годовая волатильность
    annual_vol = simple_returns.std() * np.sqrt(12)

    # Sharpe Ratio
    sharpe = (excess.mean() / simple_returns.std() * np.sqrt(12)) if simple_returns.std() > 0 else 0

    # Max Drawdown
    cumulative = (1 + simple_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Общая доходность за период
    total_return = (1 + simple_returns).prod() - 1

    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Total Return': total_return,
        'Num Periods': len(returns)
    }


if __name__ == "__main__":
    # Загружаем данные
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "log_returns.csv"
    returns = pd.read_csv(data_path, index_col=0, parse_dates=True)
    results_path = Path(__file__).parent.parent.parent / "results"
    results_path.mkdir(exist_ok=True)

    print("="*50)
    print("БЭКТЕСТ: Классический Марковиц")
    print("="*50)
    print(f"Данные: {returns.index[0].date()} — {returns.index[-1].date()}")
    print(f"Активы: {list(returns.columns)}")
    print()

    # Запускаем бэктест
    portfolio_returns = run_backtest(
        returns,
        save_weights_path=results_path / "baseline1_weights.csv"
    )

    # Считаем метрики
    metrics = calculate_metrics(portfolio_returns, rf=RF)

    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ")
    print("="*50)
    for name, value in metrics.items():
        if 'Return' in name or 'Volatility' in name or 'Drawdown' in name:
            print(f"{name}: {value:.2%}")
        elif 'Ratio' in name:
            print(f"{name}: {value:.2f}")
        else:
            print(f"{name}: {value}")
