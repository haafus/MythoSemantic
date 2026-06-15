# How To: MythoSemantic

Краткая карта проекта: что делает каждый модуль, какие файлы он читает и пишет, и как его запускать. Все команды ниже предполагают запуск из корня проекта.

## Структура проекта

```
config/          — статические конфиги, шаблоны, download_list.json
outputs/         — всё, что генерируется при запуске (corpus, analysis, logs, …)
src/             — исходный код (все Python-пакеты, settings.py, main.py, cli.py)
docs/            — документация
tests/           — тесты
pyproject.toml   — конфигурация проекта, зависимости, ruff, mypy
```

## Подготовка окружения

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install --upgrade pip
pip install -e ".[all,dev]"
```

Часть команд скачивает модели, обращается к внешним сайтам или пишет большие артефакты в `outputs/cache/`, `outputs/chroma_db/`, `outputs/analysis/`, `outputs/corpus_chunked/`, `outputs/graphs/` и `outputs/logs/`.

## Конфигурация

Конфигурация устроена в два слоя:

- **`src/settings.py`** — единый источник путей и окружения. Все директории (`outputs/corpus`, `outputs/chroma_db`, …), модель и chunking по умолчанию. Переопределяется через переменные окружения с префиксом `MYTHO_` или файл `.env` (например, `MYTHO_CORPUS_DIR=/data/corpus`).
- **`config/*.yaml`** — доменные параметры каждого модуля (батчи, ретраи, LLM-эндпоинты). Файлы поставляются полностью закомментированными: раскомментируйте ключ, чтобы переопределить значение из кода. Пути в YAML дублировать не нужно — модульные конфиги берут их из `settings.py`.

## CLI

Все команды проекта доступны через единую точку входа `mytho`:

```bash
mytho --help
mytho embeddings --help
mytho projection --help
mytho cluster --help
mytho graphs --help
mytho server --help
mytho pipeline --help
```

## corpus

Модуль сборки корпуса из `config/download_list.json`. Тексты с Project Gutenberg автоматически очищаются от лицензионных заголовков и хвостов при скачивании.

Основные файлы:
- `src/corpus/downloader.py` скачивает источники.
- `src/corpus/utils.py` извлекает текст из HTML/PDF/TXT и нормализует его.
- `src/corpus/builder.py` строит структуру `outputs/corpus/`, метаданные и каталог.
- `src/corpus/clean_gutenberg.py` автоматически удаляет Gutenberg-боллерплейт.

Возможности:
- Скачать и обработать источники.
- Автоматически очистить Gutenberg-тексты (по маркерам в содержимом).
- Сохранить тексты в `outputs/corpus/<major>/<tradition>/<title>/<title>.txt`.
- Создать `outputs/corpus/corpus_metadata.json`, `outputs/corpus/traditions_info.json`.

Запуск сборки всего корпуса:

```bash
mytho corpus --type all
```

Только переводы:

```bash
mytho corpus --type translation
```

Пересобрать с перезаписью:

```bash
mytho corpus --type all --force
```

## embedding

Модуль генерации эмбеддингов и записи в Chroma DB.

Основные файлы:
- `config/embedding.yaml` задает пути, модели, chunking и batch size.
- `src/embedding/cli.py` содержит click-команды (generate, query, test, compare и др.).
- `src/embedding/builder.py` читает корпус, режет тексты на чанки, считает эмбеддинги и пишет в Chroma.
- `src/embedding/chunking.py` содержит стратегии chunking.
- `src/embedding/cache_utils.py` и `cache_validator.py` работают с кешем.

Возможности:
- Построить эмбеддинги для нескольких моделей.
- Сохранить чанки в `outputs/corpus_chunked/`.
- Сохранить индекс в `outputs/chroma_db/`.
- Кешировать эмбеддинги в `outputs/cache/`.
- Делать запросы к Chroma.

Посмотреть конфиг:

```bash
mytho embeddings show-config
```

Сгенерировать эмбеддинги по конфигу:

```bash
mytho embeddings generate
```

Сгенерировать для конкретной модели:

```bash
mytho embeddings generate --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

Выбрать chunking и тип текста:

```bash
mytho embeddings generate --chunking paragraph --text-type all
```

Поиск по индексу:

```bash
mytho embeddings query "creation of the world" --model "BAAI/bge-m3" --top-k 5
```

Проверить кеш:

```bash
mytho embeddings validate-cache
```

Удалить коллекцию модели:

```bash
mytho embeddings clear-cache --model "BAAI/bge-m3"
```

## projection

Модуль анализа эмбеддингов из Chroma DB и генерации HTML/CSV/JSON-артефактов в `outputs/analysis/`.

Основные файлы:
- `src/projection/loader.py` читает данные из Chroma.
- `src/projection/analyzer.py` собирает статистику.
- `src/projection/visualization.py` строит PCA, UMAP, t-SNE, heatmap и dashboard.
- `config/projection.yaml` задает пути и параметры визуализации.

Возможности:
- Получить статистику по модели.
- Сохранить `model_info.json`, `models.json`, `embeddings_data.csv`.
- Построить интерактивные графики семантического пространства.

Запустить анализ всех доступных моделей:

```bash
mytho projection
```

Запустить анализ одной модели:

```bash
mytho projection --model "BAAI/bge-m3"
```

Только статистика, без графиков:

```bash
mytho projection --model "BAAI/bge-m3" --no-plots
```

## clustering

Модуль кластеризации эмбеддингов и сравнения алгоритмов.

Основные файлы:
- `src/clustering/models.py` содержит KMeans, HDBSCAN из sklearn, Spectral, Birch, GMM, MeanShift, OPTICS.
- `src/clustering/metrics.py` считает метрики кластеризации.
- `src/clustering/visualization.py` строит HTML-графики и матрицы.
- `src/clustering/run_clustering.py` содержит логику запуска анализа.

Возможности:
- Кластеризовать эмбеддинги одной или всех моделей.
- Сохранить метрики и labels в `outputs/analysis/<model>/clustering/`.
- Построить `clusters_*.html`, `confusion_matrix_*.html`, `metrics_dashboard.html`.

Запустить все алгоритмы для всех доступных моделей:

```bash
mytho cluster
```

Запустить один алгоритм для одной модели:

```bash
mytho cluster --model "BAAI/bge-m3" --single-model --algorithm kmeans
```

Запустить без визуализаций:

```bash
mytho cluster --single-model --algorithm kmeans --no-viz
```

## graphs

Модуль извлечения персонажей, отношений, мест и времени через LLM и генерации графов.

Основные файлы:
- `config/graphs.yaml` задает LLM, пути и параметры чанков.
- `config/graphs_prompts.json` содержит промпты.
- `src/graphs/llm_processing.py` вызывает OpenAI-compatible API.
- `src/graphs/run_graph_generation.py` режет тексты и агрегирует сущности.
- `src/graphs/graph_generator.py` строит HTML-граф через NetworkX и Cytoscape.

Возможности:
- Пройти по книгам из `outputs/corpus/corpus_metadata.json`.
- Извлечь сущности и связи через локальный или внешний LLM.
- Сохранить графы в `outputs/graphs/<book_id>/characters.html`.

Запуск по конфигу:

```bash
mytho graphs
```

Запуск с перезаписью готовых графов:

```bash
mytho graphs --force
```

Перед запуском проверьте `config/graphs.yaml`: по умолчанию выбран локальный OpenAI-compatible сервер `http://127.0.0.1:1234/v1/`.

## server

Современный FastAPI-сервер и SPA-интерфейс.

Возможности:
- API для списка моделей, корпуса, географии, похожих фрагментов и кластеризации.
- Раздача веб-интерфейса из `src/server/web`.
- Раздача готовых HTML-артефактов из `outputs/analysis/`, `config/template/`, `outputs/corpus/`, `outputs/corpus_chunked/`.

Запуск:

```bash
mytho server
```

С явным указанием хоста и порта:

```bash
mytho server --host 0.0.0.0 --port 9000
```

Проверка:

```bash
curl http://127.0.0.1:8000/api/health
```

Открыть интерфейс: `http://127.0.0.1:8000/`.

## config/template

HTML-шаблоны для старого UI.

Возможности:
- Страницы `home.html`, `corpus.html`, `geography.html`, `embeddings_analysis.html`, `cluster_analysis.html`.
- Общая навигация `navbar.html`.
- Логотип `logo.jpg`.

## server/web

Современный SPA-фронтенд.

Основные файлы:
- `index.html` подключает стили и JS.
- `assets/app.js` содержит маршруты и экраны.
- `assets/core.js` содержит API helpers и состояние.
- `assets/plot-utils.js` работает с Plotly-графиками.
- `assets/app.css` содержит стили.

Запускается через:

```bash
mytho server
```

## Директории outputs/

Все генерируемые данные хранятся в `outputs/`:

- `outputs/corpus/` — основной текстовый корпус с метаданными и каталогом. Создается через `mytho corpus`.
- `outputs/corpus_chunked/` — корпус после разбиения на чанки. Создается через `mytho embeddings generate`.
- `outputs/chroma_db/` — локальная Chroma DB с векторными коллекциями. Создается через `mytho embeddings generate`.
- `outputs/analysis/` — результаты анализа: `models.json`, HTML-графики, кластеризация. Создается через `mytho projection` и `mytho cluster`.
- `outputs/graphs/` — готовые HTML-графы персонажей и связей. Создается через `mytho graphs`.
- `outputs/cache/` — кеш эмбеддингов в `.npy` и `.json`. Создается через `mytho embeddings generate`.
- `outputs/logs/` — логи всех пайплайнов.

## Типовой пайплайн

Запустить всё одной командой:

```bash
mytho pipeline --model "BAAI/bge-m3" --text-type all
```

Или по шагам:

```bash
# 1. Собрать корпус (Gutenberg-тексты очищаются автоматически)
mytho corpus --type all

# 2. Построить эмбеддинги и Chroma DB
mytho embeddings generate

# 3. Построить визуальный анализ эмбеддингов
mytho projection

# 4. Построить кластеризацию
mytho cluster

# 5. Запустить веб-интерфейс
mytho server
```

Можно пропускать отдельные шаги:

```bash
mytho pipeline --skip-corpus --skip-graphs
```
