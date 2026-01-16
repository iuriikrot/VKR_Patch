"""
Запуск всех моделей и сохранение результатов.

Использование:
    python run_all.py           # Запустить модели
    python run_all.py --fast    # Быстрый режим PatchTST (для отладки)

Результаты сохраняются в results/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from optimization.markowitz import maximize_sharpe
from optimization.covariance import compute_covariance
try:
    from backtesting.backtest_statsforecast import run_backtest as run_statsforecast
except ImportError as exc:
    raise ImportError(
        "statsforecast не установлен. Установите: pip install statsforecast"
    ) from exc

# Загружаем конфигурацию
config_path = Path(__file__).parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Параметры
TRAIN_WINDOW = config['backtest']['train_window']
TEST_WINDOW = config['backtest']['test_window']
RF = config['optimization']['risk_free_rate']
CV_METHOD = config['optimization'].get('covariance', 'sample')
CONSTRAINTS = config.get('optimization', {}).get('constraints', {})
MIN_WEIGHT = CONSTRAINTS.get('min_weight', 0.0)
MAX_WEIGHT = CONSTRAINTS.get('max_weight', 1.0)
LONG_ONLY = CONSTRAINTS.get('long_only', True)
FULLY_INVESTED = CONSTRAINTS.get('fully_invested', True)
GROSS_EXPOSURE = CONSTRAINTS.get('gross_exposure')

 


def calculate_metrics(returns, rf=0.02):
    """Расчёт метрик портфеля (returns — месячные лог-доходности)."""
    simple_returns = np.exp(returns) - 1
    monthly_rf = (1 + rf) ** (1 / 12) - 1
    excess = simple_returns - monthly_rf

    if len(simple_returns) > 0:
        annual_return = (1 + simple_returns).prod() ** (12 / len(simple_returns)) - 1
    else:
        annual_return = 0
    annual_vol = simple_returns.std() * np.sqrt(12)
    sharpe = (excess.mean() / simple_returns.std() * np.sqrt(12)) if simple_returns.std() > 0 else 0

    cumulative = (1 + simple_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    total_return = (1 + simple_returns).prod() - 1

    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Total Return': total_return,
        'Num Periods': len(returns)
    }


def compute_monthly_log_return(test_data, weights, fully_invested=True):
    """Доходность за месяц при ребалансировке раз в месяц (buy-and-hold)."""
    asset_gross = np.exp(test_data.sum(axis=0).values)
    portfolio_gross = np.dot(weights, asset_gross)
    if not fully_invested:
        portfolio_gross += (1 - weights.sum())
    return np.log(portfolio_gross)


# ============================================================
# BASELINE 1: Историческое среднее
# ============================================================

def run_baseline1(returns, save_weights_path=None):
    """Бэктест: μ = историческое среднее."""
    n = len(returns)
    portfolio_returns = []
    dates = []
    weights_list = [] if save_weights_path else None

    i = 0
    while i + TRAIN_WINDOW + TEST_WINDOW <= n:
        train_data = returns.iloc[i:i + TRAIN_WINDOW]
        test_data = returns.iloc[i + TRAIN_WINDOW:i + TRAIN_WINDOW + TEST_WINDOW]

        mu = train_data.mean().values * 252
        cov = compute_covariance(train_data, method=CV_METHOD, annualize=252)
        weights = maximize_sharpe(
            mu,
            cov,
            rf=RF,
            min_weight=MIN_WEIGHT,
            max_weight=MAX_WEIGHT,
            long_only=LONG_ONLY,
            fully_invested=FULLY_INVESTED,
            gross_exposure=GROSS_EXPOSURE
        )

        month_return = compute_monthly_log_return(
            test_data,
            weights,
            fully_invested=FULLY_INVESTED
        )

        portfolio_returns.append(month_return)
        dates.append(test_data.index[0])
        if weights_list is not None:
            weights_list.append(weights)
        i += TEST_WINDOW

    if weights_list is not None:
        weights_df = pd.DataFrame(weights_list, index=dates, columns=returns.columns)
        weights_df.to_csv(save_weights_path)

    return pd.Series(portfolio_returns, index=dates)


# ============================================================
# BASELINE 2: StatsForecast AutoARIMA
# ============================================================

def run_baseline2(returns, save_weights_path=None):
    """Бэктест: μ = прогноз StatsForecast AutoARIMA."""
    return run_statsforecast(returns, save_weights_path=save_weights_path)


# ============================================================
# PATCHTST
# ============================================================

def run_patchtst(returns, mode='full', save_weights_path=None):
    """Бэктест: μ = прогноз PatchTST."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    # Загружаем параметры PatchTST
    patchtst_config = config['models']['patchtst'][mode]

    INPUT_LEN = patchtst_config['input_length']
    PRED_LEN = patchtst_config['pred_length']
    PATCH_LEN = patchtst_config['patch_length']
    STRIDE = patchtst_config['stride']
    D_MODEL = patchtst_config['d_model']
    N_HEADS = patchtst_config['n_heads']
    N_LAYERS = patchtst_config['n_layers']
    D_FF = patchtst_config['d_ff']
    DROPOUT = patchtst_config['dropout']
    USE_REVIN = patchtst_config['use_revin']
    MASK_RATIO = patchtst_config['mask_ratio']
    PRETRAIN_EPOCHS = patchtst_config['pretrain_epochs']
    FINETUNE_EPOCHS = patchtst_config.get('finetune_epochs', 5)  # Fine-tuning epochs
    PRETRAIN_LR = patchtst_config['pretrain_lr']
    BATCH_SIZE = patchtst_config['batch_size']

    def select_device():
        if torch.backends.mps.is_available():
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    device = select_device()

    # Импортируем модель
    from models.patchtst import (
        PatchTST_SelfSupervised, pretrain_patchtst, finetune_patchtst,
        forecast_patchtst, create_sequences
    )

    n = len(returns)
    portfolio_returns = []
    dates = []
    weights_list = [] if save_weights_path else None

    total_steps = (n - TRAIN_WINDOW - TEST_WINDOW) // TEST_WINDOW + 1
    i = 0
    step = 0

    while i + TRAIN_WINDOW + TEST_WINDOW <= n:
        train_data = returns.iloc[i:i + TRAIN_WINDOW]
        test_data = returns.iloc[i + TRAIN_WINDOW:i + TRAIN_WINDOW + TEST_WINDOW]

        step += 1
        if step % 5 == 0 or step == 1:
            print(f"    PatchTST: шаг {step}/{total_steps}")

        forecasts = []
        for ticker in train_data.columns:
            series = train_data[ticker].values

            model = PatchTST_SelfSupervised(
                input_len=INPUT_LEN,
                pred_len=PRED_LEN,
                patch_len=PATCH_LEN,
                stride=STRIDE,
                d_model=D_MODEL,
                n_heads=N_HEADS,
                n_layers=N_LAYERS,
                d_ff=D_FF,
                dropout=DROPOUT,
                mask_ratio=MASK_RATIO,
                use_revin=USE_REVIN
            ).to(device)

            model = pretrain_patchtst(
                model, series,
                epochs=PRETRAIN_EPOCHS,
                lr=PRETRAIN_LR,
                batch_size=BATCH_SIZE,
                verbose=False
            )

            # Fine-tuning: обучаем prediction head (официальный подход)
            # После pretraining переносим веса encoder и дообучаем prediction head
            X_train, y_train = create_sequences(series, INPUT_LEN, PRED_LEN)
            if len(X_train) > 0:
                model = finetune_patchtst(
                    model, X_train, y_train,
                    epochs=FINETUNE_EPOCHS,
                    lr=PRETRAIN_LR * 0.1,  # Меньший lr для fine-tuning
                    batch_size=BATCH_SIZE,
                    verbose=False
                )

            last_input = series[-INPUT_LEN:]
            forecast = forecast_patchtst(model, last_input)
            annual_return = forecast.mean() * 252
            forecasts.append(annual_return)

        mu = np.array(forecasts)
        cov = compute_covariance(train_data, method=CV_METHOD, annualize=252)
        weights = maximize_sharpe(
            mu,
            cov,
            rf=RF,
            min_weight=MIN_WEIGHT,
            max_weight=MAX_WEIGHT,
            long_only=LONG_ONLY,
            fully_invested=FULLY_INVESTED,
            gross_exposure=GROSS_EXPOSURE
        )

        month_return = compute_monthly_log_return(
            test_data,
            weights,
            fully_invested=FULLY_INVESTED
        )

        portfolio_returns.append(month_return)
        dates.append(test_data.index[0])
        if weights_list is not None:
            weights_list.append(weights)
        i += TEST_WINDOW

    if weights_list is not None:
        weights_df = pd.DataFrame(weights_list, index=dates, columns=returns.columns)
        weights_df.to_csv(save_weights_path)

    return pd.Series(portfolio_returns, index=dates)


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Запуск всех моделей')
    parser.add_argument('--fast', action='store_true', help='Быстрый режим PatchTST')
    args = parser.parse_args()

    def prompt_yes_no(prompt, default=False):
        suffix = " [Y/n]: " if default else " [y/N]: "
        while True:
            ans = input(prompt + suffix).strip().lower()
            if ans == "":
                return default
            if ans in ("y", "yes", "да", "д"):
                return True
            if ans in ("n", "no", "нет", "н"):
                return False
            print("Введите 'y' или 'n'.")

    def prompt_models():
        print("Выберите модели для запуска:")
        print("  1 - Baseline 1 (Историческое среднее)")
        print("  2 - StatsForecast AutoARIMA")
        print("  3 - PatchTST")
        while True:
            ans = input("Введите номера через запятую (Enter = все): ").strip()
            if ans == "":
                return {"baseline1", "baseline2", "patchtst"}
            parts = [p.strip() for p in ans.replace(" ", "").split(",") if p.strip()]
            mapping = {"1": "baseline1", "2": "baseline2", "3": "patchtst"}
            selected = {mapping[p] for p in parts if p in mapping}
            if selected:
                return selected
            print("Не удалось распознать выбор. Пример: 1,3")

    patchtst_mode = 'fast' if args.fast else 'full'

    # Создаём папку results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Параметры берутся из config/config.yaml")
    if prompt_yes_no("Скачать данные заново?", default=False):
        from data.downloader import download_and_prepare_data
        download_and_prepare_data()

    # Загружаем данные
    data_path = Path(__file__).parent / "data" / "raw" / "log_returns.csv"
    if not data_path.exists():
        print("Ошибка: данные не найдены!")
        print(f"Ожидается файл: {data_path}")
        print("\nЗапустите сначала загрузку данных:")
        print("  python src/data/downloader.py")
        return

    returns = pd.read_csv(data_path, index_col=0, parse_dates=True)

    selected_models = prompt_models()

    print("=" * 60)
    print("ЗАПУСК ВСЕХ МОДЕЛЕЙ")
    print("=" * 60)
    print(f"Данные: {returns.index[0].date()} — {returns.index[-1].date()}")
    print(f"Акций: {len(returns.columns)}")
    print(f"Train: {TRAIN_WINDOW} дней, Test: {TEST_WINDOW} дней")
    print(f"PatchTST режим: {patchtst_mode.upper()}")
    print()

    results = {}

    total_steps = len(selected_models)
    step_num = 0

    # Baseline 1
    if "baseline1" in selected_models:
        step_num += 1
        print(f"[{step_num}/{total_steps}] Baseline 1: Историческое среднее...")
        baseline1_returns = run_baseline1(
            returns,
            save_weights_path=results_dir / f"baseline1_weights_{timestamp}.csv"
        )
        results['baseline1'] = {
            'returns': baseline1_returns,
            'metrics': calculate_metrics(baseline1_returns, rf=RF)
        }
        print(f"      Sharpe: {results['baseline1']['metrics']['Sharpe Ratio']:.2f}")
        print()

    # Baseline 2
    if "baseline2" in selected_models:
        step_num += 1
        print(f"[{step_num}/{total_steps}] Baseline 2: StatsForecast AutoARIMA...")
        baseline2_returns = run_baseline2(
            returns,
            save_weights_path=results_dir / f"statsforecast_weights_{timestamp}.csv"
        )
        results['baseline2'] = {
            'returns': baseline2_returns,
            'metrics': calculate_metrics(baseline2_returns, rf=RF)
        }
        print(f"      Sharpe: {results['baseline2']['metrics']['Sharpe Ratio']:.2f}")
        print()

    # PatchTST
    if "patchtst" in selected_models:
        step_num += 1
        print(f"[{step_num}/{total_steps}] PatchTST Self-Supervised ({patchtst_mode})...")
        patchtst_returns = run_patchtst(
            returns,
            mode=patchtst_mode,
            save_weights_path=results_dir / f"patchtst_weights_{timestamp}.csv"
        )
        results['patchtst'] = {
            'returns': patchtst_returns,
            'metrics': calculate_metrics(patchtst_returns, rf=RF)
        }
        print(f"      Sharpe: {results['patchtst']['metrics']['Sharpe Ratio']:.2f}")
        print()

    # CSV с доходностями
    if "baseline1" in results:
        results["baseline1"]["returns"].to_csv(results_dir / f"baseline1_returns_{timestamp}.csv")
    if "baseline2" in results:
        results["baseline2"]["returns"].to_csv(results_dir / f"statsforecast_returns_{timestamp}.csv")
    if "patchtst" in results:
        results["patchtst"]["returns"].to_csv(results_dir / f"patchtst_returns_{timestamp}.csv")

    # Сводная таблица
    comparison_data = {}
    if "baseline1" in results:
        comparison_data['Baseline 1 (Hist Mean)'] = results['baseline1']['metrics']
    if "baseline2" in results:
        comparison_data['Baseline 2 (StatsForecast)'] = results['baseline2']['metrics']
    if "patchtst" in results:
        comparison_data['PatchTST'] = results['patchtst']['metrics']
    comparison = pd.DataFrame(comparison_data).T
    comparison.to_csv(results_dir / f"comparison_{timestamp}.csv")

    # JSON с метриками
    metrics_json = {
        'timestamp': timestamp,
        'config': {
            'train_window': TRAIN_WINDOW,
            'test_window': TEST_WINDOW,
            'risk_free_rate': RF,
            'patchtst_mode': patchtst_mode
        },
        'metrics': {}
    }
    if "baseline1" in results:
        metrics_json['metrics']['baseline1'] = results['baseline1']['metrics']
    if "baseline2" in results:
        metrics_json['metrics']['baseline2'] = results['baseline2']['metrics']
    if "patchtst" in results:
        metrics_json['metrics']['patchtst'] = results['patchtst']['metrics']
    with open(results_dir / f"metrics_{timestamp}.json", 'w') as f:
        json.dump(metrics_json, f, indent=2, default=str)

    # Вывод результатов
    print("=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    labels = []
    if "baseline1" in results:
        labels.append(("baseline1", "Baseline 1"))
    if "baseline2" in results:
        labels.append(("baseline2", "StatsF"))
    if "patchtst" in results:
        labels.append(("patchtst", "PatchTST"))

    header = f"\n{'Метрика':<25}" + "".join([f"{label:>12}" for _, label in labels])
    print(header)
    print("-" * (25 + 12 * len(labels)))
    for metric in ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Total Return']:
        if 'Ratio' in metric:
            row = f"{metric:<25}" + "".join(
                [f"{results[key]['metrics'][metric]:>12.2f}" for key, _ in labels]
            )
        else:
            row = f"{metric:<25}" + "".join(
                [f"{results[key]['metrics'][metric]:>11.2%}" for key, _ in labels]
            )
        print(row)

    print()
    print(f"Результаты сохранены в: {results_dir}/")
    print(f"  - comparison_{timestamp}.csv")
    print(f"  - metrics_{timestamp}.json")
    print(f"  - *_returns_{timestamp}.csv")


if __name__ == "__main__":
    main()
