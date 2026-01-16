"""
Пример подбора ARIMA для одной акции.
Используем statsmodels (более стабильный).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA

# Загружаем данные
data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "log_returns.csv"
returns = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Берём одну акцию - AAPL
ticker = "AAPL"
series = returns[ticker]

print(f"Акция: {ticker}")
print(f"Всего наблюдений: {len(series)}")
print(f"Период: {series.index[0].date()} — {series.index[-1].date()}")

# Берём последние 5 лет для обучения (как в бэктесте)
train_window = 1260
train_data = series.iloc[-train_window-21:-21]  # 5 лет до последнего месяца
test_data = series.iloc[-21:]  # последний месяц для проверки

# Данные уже содержат только торговые дни (252 в году)
# Предупреждения statsmodels игнорируем — на результат не влияют

print(f"\nTrain: {len(train_data)} торговых дней")
print(f"Test: {len(test_data)} дней")

# Подбираем ARIMA - перебираем комбинации (p, d, q)
print("\n" + "="*50)
print("Подбор параметров ARIMA...")
print("="*50)

best_aic = np.inf
best_order = None
best_model = None

# Для доходностей d=0 (ряд уже стационарный)
# Перебираем p и q от 0 до 3
for p in range(4):
    for q in range(4):
        try:
            model = ARIMA(train_data, order=(p, 0, q))
            fitted = model.fit()
            aic = fitted.aic

            if aic < best_aic:
                best_aic = aic
                best_order = (p, 0, q)
                best_model = fitted

            print(f"ARIMA({p},0,{q}): AIC = {aic:.2f}")
        except:
            print(f"ARIMA({p},0,{q}): не удалось обучить")

print("\n" + "="*50)
print("Результат:")
print("="*50)
print(f"Лучшая модель: ARIMA{best_order}")
print(f"AIC: {best_aic:.2f}")

# Прогноз на 21 день вперёд
forecast = best_model.forecast(steps=21)

print(f"\nПрогноз средней дневной доходности: {forecast.mean():.6f}")
print(f"Прогноз месячной доходности (сумма): {forecast.sum():.4f} ({forecast.sum()*100:.2f}%)")

# Сравним с реальностью
actual_sum = test_data.sum()

print(f"\nФактическая средняя дневной доходности: {test_data.mean():.6f}")
print(f"Фактическая месячная доходность: {actual_sum:.4f} ({actual_sum*100:.2f}%)")

# Ошибка
print(f"\nОшибка прогноза: {abs(forecast.sum() - actual_sum):.4f}")
