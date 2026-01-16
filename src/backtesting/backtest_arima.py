"""
Бэктест Baseline 2: StatsForecast AutoARIMA прогноз для оценки μ.
"""

import pandas as pd
from pathlib import Path
import sys
import warnings
import yaml
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))

from backtesting.backtest_statsforecast import run_backtest, calculate_metrics

# Загружаем конфигурацию
config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

RF = config['optimization']['risk_free_rate']


if __name__ == "__main__":
    # Загружаем данные
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "log_returns.csv"
    returns = pd.read_csv(data_path, index_col=0, parse_dates=True)

    print("="*60)
    print("БЭКТЕСТ: StatsForecast AutoARIMA")
    print("="*60)
    print(f"Данные: {returns.index[0].date()} — {returns.index[-1].date()}")
    print()

    results_path = Path(__file__).parent.parent.parent / "results"
    results_path.mkdir(exist_ok=True)

    # Бэктест
    portfolio_returns = run_backtest(
        returns,
        save_weights_path=results_path / "statsforecast_weights.csv"
    )

    # Метрики
    metrics = calculate_metrics(portfolio_returns, rf=RF)

    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ: StatsForecast AutoARIMA")
    print("="*60)
    for name, value in metrics.items():
        if 'Return' in name or 'Volatility' in name or 'Drawdown' in name:
            print(f"{name}: {value:.2%}")
        elif 'Ratio' in name:
            print(f"{name}: {value:.2f}")
        else:
            print(f"{name}: {value}")
