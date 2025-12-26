# Paddy Disease - классификация болезней риса по фото листа

В сельском хозяйстве раннее обнаружение заболеваний растений напрямую влияет на урожайность и экономические потери. В этом проекте мы строим модель, которая по фотографии листика риса определяет один из 10 классов (9 заболеваний и здоровый рис). В соответсвующем соревновании на каггле говорится, что выращивание риса требует постоянного контроля, поскольку некоторые болезни и вредители могут поражать рисовые культуры, что приводит к потере урожая на 70%. Для борьбы с этими болезнями и предотвращения гибели урожая обычно требуется квалифицированный надзор. При ограниченном количестве специалистов по защите растений ручная диагностика заболеваний является трудоемкой и дорогостоящей. Таким образом, становится все более важным автоматизировать процесс идентификации заболеваний, используя методы, основанные на компьютерном зрении, которые позволили достичь многообещающих результатов в различных областях.

---

## Постановка задачи

### Что делаем

Мы обучаем классификатор изображений на основе ResNet34 с заменнеым полносвязанным слоем, модели, которая по входному фото листа риса выдаёт наиболее вероятное заболевание кустика риса.

### Цель

Построить воспроизводимый пайплайн обучения и инференса. Подготовить модель к развёртыванию. Предоставить простой способ взаимодействия с моделью через **Web UI**.

### Зачем это нужно

- Практическая ценность: быстрый скрининг заболеваний по фото может помочь в принятии решений по обработке посевов.
- Образовательная цель: демонстрация полного MLOps-цикла (data → train → log → export → deploy → inference UI).
- Потенциал расширения: в дальнейшем можно добавлять новые сорта и регионы и улучшить модель на более современную.

---

## Формат входных и выходных данных

### Вход

`.jpg` изображение листа риса (RGB). В предобработке используется нормализация ImageNet и приведение к размеру `224×224` и аугментации, например вращения, изменения контрасности и яркости и небольшие повороты.

### Выход

Вектор логитов/вероятностей по 10 классам и как итоговое предсказание: top-k классов с вероятностями, которые видны в Web UI.

---

## Метрики

- **Accuracy** (`val_acc`) — базовая метрика, отражает долю верных предсказаний.
- **Macro F1** (`val_f1_macro`) — дополнительно учитывает качество по каждому классу и полезна при возможном дисбалансе классов.

---

## Источник данных

Kaggle competition: [`paddy-disease-classification`](https://www.kaggle.com/competitions/paddy-disease-classification)
Скачивание встроено в проект, оно запускается при запуске тренировочной функции, если не было скачано до этого.

---

## Моделирование

### Подготовка данных

- **Train/val split** (фиксируем seed для воспроизводимости)
- **Преобразования**:
  - Resize до `224×224`
  - Перевод в тензор
  - Различные аугментации, такие как повороты, флипы, вращения, изменения яркости, контарстности и насыщения.
  - Нормализация посчитанная на ImageNet

### Архитектура модели

Мы используем **ResNet34** (предобученную на ImageNet), заменяя финальный классификатор на слой под 10 классов и добавляя dropout.

---

## TL;DR (быстрый старт)

```bash
# 0) Установить uv (если ещё не установлен)
# https://docs.astral.sh/uv/

# 1) Установить зависимости
uv sync

# 2) Проверка кодстайла
uv run pre-commit install
uv run pre-commit run -a

# 3) Скачивание данных с Kaggle и DVC
uv run python -m paddy_disease.commands download_data

# 4) Обучение
uv run python -m paddy_disease.commands train

# 5) Экспорт ONNX
uv run python -m paddy_disease.commands export_onnx

# 6) Экспорт labels (индекс -> имя класса, нужно для инференса)
uv run python -m paddy_disease.commands export_labels

# 7) Копирование модели для Triton
cp -f models/onnx/paddy_resnet34.onnx triton/model_repository/paddy_resnet34/1/
# если появился внешний файл весов:
cp -f models/onnx/paddy_resnet34.onnx.data triton/model_repository/paddy_resnet34/1/

# 8) Triton в Docker
sudo docker compose up -d triton
curl -sf http://localhost:8000/v2/health/ready >/dev/null && echo "READY"

# 9) Web UI
uv run python -m paddy_disease.commands web --port=8081
# открыть http://localhost:8081/
```

---

## Структура репозитория

```
paddy_disease/
  commands.py                 # CLI
  config.py                   # dataclasses конфигов
  data/                       # датасет и loaders и transforms
  training/
    datamodule.py             # LightningDataModule
    lightning_module.py       # LightningModule
    train.py                  # train()
    plot_callback.py          # сохранение графиков
  export/
    onnx_export.py            # export_onnx_main()
    tensorrt_export.py        # export_tensorrt_main()
    tensorrt.sh
  inference/
    triton_client.py          # HTTP Triton client
    labels_export.py          # export_labels -> models/labels.json
  utils/
    git.py                    # git commit hash для логирования, для понимания версии кода

configs/
  config.yaml
  data/*.yaml
  model/*.yaml
  optim/*.yaml
  train/*.yaml
  checkpoint/*.yaml
  export/
    onnx.yaml
    tensorrt.yaml
  logging/
    mlflow.yaml

triton/
  model_repository/
    paddy_resnet34/
      1/                      # модель нужно положить сюда для подъема тритона
      config.pbtxt

models/
  checkpoints/ (.gitkeep)
  onnx/        (.gitkeep)
  trt/         (.gitkeep)

plots/         # графики
```

---

## Setup окружения

### Требования

- Python 3.13
- `uv`
- `git`
- `docker` (для Triton)
- `kaggle` CLI (для скачивания датасета)

### Установка зависимостей

```bash
uv sync
```

### Pre-commit (кодстайл)

```bash
uv run pre-commit install
uv run pre-commit run -a
```

---

## Данные: Kaggle и DVC

### Kaggle API token

Нужен `kaggle.json`:

- Linux: `~/.kaggle/kaggle.json` (права `chmod 600`)

Проверка:

```bash
kaggle --version
```

### Скачать данные и подключить DVC

```bash
uv run python -m paddy_disease.commands download_data
```

Структура `data/raw`:

```
data/raw/
  train.csv
  train_images/
  test_images/
```

---

## Обучение

Запуск:

```bash
uv run python -m paddy_disease.commands train
```

Так же можно написать с кастомными параметрами м помощью Hydra override:

```bash
uv run python -m paddy_disease.commands train train.epochs=1 data.loader.batch_size=8 data.loader.num_workers=0
```

Выходы:

- `models/checkpoints/` — чекпоинты
- `plots/` — графики
- `mlruns/` — локальное хранилище MLflow

---

## Логирование (MLflow + графики)

В проекте логируются:

- метрики: `train_loss`, `val_loss`, `val_acc`, `val_f1_macro`
- гиперпараметры (Hydra config)
- `git commit id`

Если нужно поднять локально MLflow UI:

```bash
uv run mlflow ui --backend-store-uri ./mlruns --port 8080
```

Открыть: `http://127.0.0.1:8080`

---

## Экспорт модели

### ONNX

```bash
uv run python -m paddy_disease.commands export_onnx
```

### TensorRT

Требуется хост с GPU:

```bash
uv run python -m paddy_disease.commands export_tensorrt
```

---

## Triton (HTTP) + Web UI

### Подготовка model repository Triton

Сначала нужно скопировать .onnx и .onnx.data в директорию /1. Получится структура:

```
triton/model_repository/paddy_resnet34/
  1/
    paddy_resnet34.onnx
    paddy_resnet34.onnx.data (если есть)
  config.pbtxt
```

### Запуск Triton

```bash
sudo docker compose up -d triton
sudo docker logs -f triton_paddy
```

Проверка readiness:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v2/health/ready
# Ожидаем 200, тогда все работает
```

Проверка модели:

```bash
curl -s http://localhost:8000/v2/models/paddy_resnet34 | head
# Ожидаем параметры модели, скопированной в triton/model_repository/paddy_resnet34/1/
```

### Labels (индекс -> класс)

Это понадобится для инференса, чтобы класс был человекочитаем:

```bash
uv run python -m paddy_disease.commands export_labels
```

### Web UI

```bash
uv run python -m paddy_disease.commands web --port=8081
```

Открыть:

- `http://localhost:8081/`

---

## CLI

Единая точка входа:

```bash
uv run python -m paddy_disease.commands --help
```

Основные команды:

- `download_data`
- `train`
- `export_onnx`
- `export_tensorrt`
- `export_labels`
- `web`
