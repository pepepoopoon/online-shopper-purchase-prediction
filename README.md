# Предсказание покупки посетителем интернет-магазина

## Описание задачи

Бинарная классификация сессии до решения о рекламном контакте. Цель `Revenue` показывает, завершилась ли сессия покупкой; accuracy не используется как единственный критерий из-за дисбаланса.

## Цель проекта

Ранжировать сессии по вероятности покупки и выбрать ограниченную долю наиболее перспективных посетителей. Основная метрика — PR-AUC, рабочий порог задаётся бюджетом контакта.

## Архитектура решения

Валидация схемы и достаточности каждого класса → стратифицированные train/validation/test →
`ColumnTransformer`, обучаемый только на train → Dummy, Logistic Regression, Decision Tree,
Random Forest и Gradient Boosting → выбор модели и порога на validation → неизменяемая оценка
test. Артефакт хранит pipeline, порог и seed.

## Структура каталогов

`src/online_shopper` содержит генератор, данные и CLI; `tests` — автономные тесты; `data` — инструкция; `artifacts` и `reports` создаются локально; `.github/workflows` — CI.

## Используемые технологии

Python 3.11, NumPy, pandas, scikit-learn, joblib; pytest и Ruff для разработки.

## Требования к окружению

Python 3.11.15. Версии библиотек зафиксированы в `pyproject.toml`.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Подготовка данных

Контрольный источник — [UCI Online Shoppers Purchasing Intention, id 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset),
лицензия CC BY 4.0. Реальные данные в репозиторий не включены, поэтому локальные версия
файла, размер и SHA-256 не зафиксированы; их нужно записать перед полным экспериментом.
Для технической проверки используйте `make smoke`: команда создаёт синтетические данные.

## Запуск обучения

```bash
shopper-train --data data/smoke.csv --artifact artifacts/model.joblib --report reports/validation_metrics.json
```

## Запуск оценки

```bash
shopper-evaluate --data data/smoke.csv --artifact artifacts/model.joblib --metrics reports/test_metrics.json --errors reports/test_errors.csv
```

## Запуск инференса

```bash
shopper-predict --data data/smoke.csv --artifact artifacts/model.joblib --output reports/predictions.csv
```

## Метрики

PR-AUC — основная; дополнительно сохраняются ROC-AUC, precision, recall, F1, accuracy,
confusion matrix и доля выбранных сессий. Контактная очередь содержит ровно
`ceil(n × budget_fraction)` строк; равные score разрешаются стабильным исходным порядком.

## Тестирование

`make check` запускает Ruff и pytest. Тесты генерируют данные локально и ничего не скачивают.

## Ограничения

Публичный датасет отражает исторический контекст и не доказывает причинный эффект рекламы. Синтетический smoke-набор проверяет только исполнение. Перед применением нужны временная валидация, мониторинг дрейфа и оценка стоимости контакта.

## Полученные результаты

Результаты на реальном UCI-наборе пока не запускались и не заявляются. После централизованного запуска сюда можно перенести только фактически полученные test-метрики; smoke-метрики не являются результатом исследования.

## Статус проекта

Инженерный каркас завершён; итоговая оценка на зафиксированной версии UCI-данных ожидает централизованного запуска.
