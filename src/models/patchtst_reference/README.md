# PatchTST Reference Code

Официальный код PatchTST из репозитория:
https://github.com/yuqinie98/PatchTST

## Файлы

- `PatchTST_backbone.py` - Backbone архитектура из supervised версии
- `PatchTST_layers.py` - Вспомогательные слои (RevIN, Positional Encoding)
- `patchTST_selfsupervised.py` - Self-Supervised версия с маскированием патчей

## Лицензия

Apache License 2.0

## Ссылки

- Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
- GitHub: https://github.com/yuqinie98/PatchTST

## Использование в проекте

Эти файлы используются как референс для нашей адаптированной реализации
в `src/models/patchtst.py`.
