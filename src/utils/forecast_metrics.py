"""
Метрики качества прогнозов для сравнения моделей.

Сравниваем месячные прогнозы (sum за 21 день) с фактическими доходностями.
"""

import numpy as np
import pandas as pd


def calculate_forecast_metrics(actual, predicted):
    """
    Рассчитать метрики качества прогноза.

    Args:
        actual: Series или array — фактические месячные доходности по тикерам
        predicted: Series или array — прогнозные месячные доходности по тикерам

    Returns:
        dict с метриками: RMSE, MAE, Hit Rate
    """
    # Приводим к numpy arrays
    if isinstance(actual, pd.Series):
        actual = actual.values
    if isinstance(predicted, pd.Series):
        predicted = predicted.values

    # Убираем NaN
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'hit_rate': np.nan}

    # RMSE
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # MAE
    mae = np.mean(np.abs(actual - predicted))

    # Hit Rate (совпадение знаков)
    # Игнорируем нули
    nonzero_mask = (actual != 0) & (predicted != 0)
    if nonzero_mask.sum() > 0:
        hits = np.sign(actual[nonzero_mask]) == np.sign(predicted[nonzero_mask])
        hit_rate = hits.mean()
    else:
        hit_rate = np.nan

    return {
        'rmse': rmse,
        'mae': mae,
        'hit_rate': hit_rate
    }


def aggregate_forecast_metrics(forecasts_df):
    """
    Агрегировать метрики по всем периодам бэктеста.

    Args:
        forecasts_df: DataFrame с колонками:
            - date: дата ребалансировки
            - ticker: тикер
            - actual: фактическая месячная доходность
            - predicted: прогнозная месячная доходность

    Returns:
        dict с агрегированными метриками
    """
    actual = forecasts_df['actual'].values
    predicted = forecasts_df['predicted'].values

    return calculate_forecast_metrics(actual, predicted)


def create_forecast_record(date, tickers, actual_monthly, predicted_monthly, model_name):
    """
    Создать записи прогнозов для одного периода.

    Args:
        date: дата ребалансировки
        tickers: список тикеров
        actual_monthly: Series/array — фактические месячные доходности
        predicted_monthly: Series/array — прогнозные месячные доходности
        model_name: название модели

    Returns:
        list of dicts для добавления в DataFrame
    """
    records = []
    for i, ticker in enumerate(tickers):
        records.append({
            'date': date,
            'ticker': ticker,
            'actual': actual_monthly[i] if hasattr(actual_monthly, '__getitem__') else actual_monthly.iloc[i],
            'predicted': predicted_monthly[i] if hasattr(predicted_monthly, '__getitem__') else predicted_monthly.iloc[i],
            'model': model_name
        })
    return records
