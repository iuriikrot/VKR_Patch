"""
Модуль для работы с данными.
"""

from .downloader import download_stock_data, save_data, load_data, get_data
from .preprocessor import (
    calculate_returns,
    calculate_covariance,
    calculate_mean_returns,
    create_rolling_windows,
    normalize_data,
    denormalize_data
)

__all__ = [
    "download_stock_data",
    "save_data",
    "load_data",
    "get_data",
    "calculate_returns",
    "calculate_covariance",
    "calculate_mean_returns",
    "create_rolling_windows",
    "normalize_data",
    "denormalize_data"
]
