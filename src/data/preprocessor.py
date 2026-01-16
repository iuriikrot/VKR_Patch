"""
Модуль для предобработки данных: расчёт доходностей, ковариации, скользящие окна.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Generator
import logging

logger = logging.getLogger(__name__)


def calculate_returns(
    prices: pd.DataFrame,
    method: str = "log"
) -> pd.DataFrame:
    """
    Расчёт доходностей из цен.

    Args:
        prices: DataFrame с ценами (индекс - дата, колонки - тикеры)
        method: Метод расчёта ('log' - логарифмические, 'simple' - простые)

    Returns:
        DataFrame с доходностями
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Неизвестный метод: {method}. Используйте 'log' или 'simple'")

    # Удаляем первую строку с NaN
    returns = returns.dropna()

    logger.info(f"Рассчитаны {method} доходности: {returns.shape}")
    return returns


def calculate_covariance(
    returns: pd.DataFrame,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Расчёт ковариационной матрицы.

    Args:
        returns: DataFrame с доходностями
        annualize: Приводить к годовой (умножить на 252)
        trading_days: Количество торговых дней в году

    Returns:
        Ковариационная матрица
    """
    cov = returns.cov()

    if annualize:
        cov = cov * trading_days

    return cov


def calculate_mean_returns(
    returns: pd.DataFrame,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.Series:
    """
    Расчёт средних доходностей.

    Args:
        returns: DataFrame с доходностями
        annualize: Приводить к годовой
        trading_days: Количество торговых дней в году

    Returns:
        Series со средними доходностями
    """
    mean_ret = returns.mean()

    if annualize:
        mean_ret = mean_ret * trading_days

    return mean_ret


def create_rolling_windows(
    data: pd.DataFrame,
    train_size: int,
    test_size: int,
    step: int = None
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp], None, None]:
    """
    Генератор скользящих окон для бэктестинга.

    Args:
        data: DataFrame с данными (цены или доходности)
        train_size: Размер тренировочного окна (в днях)
        test_size: Размер тестового окна (в днях)
        step: Шаг сдвига окна (по умолчанию = test_size)

    Yields:
        Tuple[train_data, test_data, test_start_date]
    """
    if step is None:
        step = test_size

    n = len(data)
    start_idx = 0

    while start_idx + train_size + test_size <= n:
        train_end_idx = start_idx + train_size
        test_end_idx = train_end_idx + test_size

        train_data = data.iloc[start_idx:train_end_idx]
        test_data = data.iloc[train_end_idx:test_end_idx]
        test_start_date = data.index[train_end_idx]

        yield train_data, test_data, test_start_date

        start_idx += step


def normalize_data(
    data: pd.DataFrame,
    method: str = "zscore"
) -> Tuple[pd.DataFrame, dict]:
    """
    Нормализация данных для ML моделей.

    Args:
        data: DataFrame для нормализации
        method: Метод ('zscore', 'minmax')

    Returns:
        Tuple[normalized_data, params] - нормализованные данные и параметры для обратного преобразования
    """
    params = {}

    if method == "zscore":
        params["mean"] = data.mean()
        params["std"] = data.std()
        normalized = (data - params["mean"]) / params["std"]
    elif method == "minmax":
        params["min"] = data.min()
        params["max"] = data.max()
        normalized = (data - params["min"]) / (params["max"] - params["min"])
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    params["method"] = method
    return normalized, params


def denormalize_data(
    data: pd.DataFrame,
    params: dict
) -> pd.DataFrame:
    """
    Обратное преобразование нормализованных данных.

    Args:
        data: Нормализованные данные
        params: Параметры нормализации

    Returns:
        Денормализованные данные
    """
    method = params["method"]

    if method == "zscore":
        return data * params["std"] + params["mean"]
    elif method == "minmax":
        return data * (params["max"] - params["min"]) + params["min"]
    else:
        raise ValueError(f"Неизвестный метод: {method}")


if __name__ == "__main__":
    from downloader import get_data
    from pathlib import Path
    import yaml

    # Загружаем конфиг
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Загружаем цены
    cache_path = Path(__file__).parent.parent.parent / "data" / "raw" / "prices.csv"
    prices = get_data(
        tickers=config["data"]["tickers"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        cache_path=str(cache_path)
    )

    # Рассчитываем доходности
    returns = calculate_returns(prices, method="log")

    # Сохраняем доходности
    returns_path = Path(__file__).parent.parent.parent / "data" / "processed" / "returns.csv"
    returns_path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(returns_path)

    print("\n" + "="*50)
    print("Логарифмические доходности:")
    print("="*50)
    print(f"Форма: {returns.shape}")
    print(f"\nСтатистика:")
    print(returns.describe())

    # Ковариационная матрица (годовая)
    cov = calculate_covariance(returns, annualize=True)
    print(f"\nГодовая ковариационная матрица:")
    print(cov.round(4))

    # Средние доходности (годовые)
    mean_ret = calculate_mean_returns(returns, annualize=True)
    print(f"\nГодовые средние доходности:")
    print(mean_ret.round(4))

    # Тест скользящих окон
    print(f"\n" + "="*50)
    print("Тест скользящих окон:")
    print("="*50)

    train_window = config["backtest"]["train_window"]
    test_window = config["backtest"]["test_window"]

    windows = list(create_rolling_windows(returns, train_window, test_window))
    print(f"Количество окон: {len(windows)}")
    print(f"Первое окно: train {windows[0][0].index[0].date()} - {windows[0][0].index[-1].date()}")
    print(f"             test  {windows[0][1].index[0].date()} - {windows[0][1].index[-1].date()}")
