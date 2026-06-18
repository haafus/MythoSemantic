# How To: MythoScope

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

Часть команд скачивает модели, обращается к внешним сайтам или пишет большие артефакты в `outputs/cache/`, `outputs/chroma/`, `outputs/analysis/`, `outputs/corpus_chunked/`, `outputs/graphs/` и `outputs/logs/`.

## Конфигурация

- **`src/settings.py`** — единый источник путей и параметров. Все директории (`outputs/corpus`, `outputs/chroma`, …), параметры chunking, LLM, сервера и т.д. Переопределяется через переменные окружения с префиксом `MYTHO_` или файл `.env` / `config/.env` (например, `MYTHO_CORPUS_DIR=/data/corpus`). Вложенные параметры через `__`: `MYTHO_LLM__MODEL=gpt4o-mini`. Полный список переменных — в `.env.example`.
- **`config/models.json`** — реестр LLM-провайдеров (base_url, model, env_key) и алиасов embedding-моделей. Алиасы позволяют писать `bge-m3` вместо `BAAI/bge-m3` в CLI и конфигах.
- **`config/corpus.json`** — каталог текстов корпуса (источники, традиции, URL).
- **`config/traditions.json`** — описания традиций и их группировка.
- **`config/graphs_prompts.json`** — промпты для LLM-извлечения сущностей.

## CLI

Все команды проекта доступны через единую точку входа `mytho`:

```bash
mytho --help
mytho corpus --help
mytho embeddings --help
mytho projection --help
mytho graphs --help
mytho server --help
mytho build --help
mytho status
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
- `src/embedding/builder.py` читает корпус, режет тексты на чанки, считает эмбеддинги и пишет в Chroma.
- `src/embedding/build_embeddings.py` оркестрирует генерацию для нескольких моделей.
- `src/embedding/chunking.py` содержит стратегии chunking.
- `config/models.json` задает модели и алиасы.

Возможности:
- Построить эмбеддинги для нескольких моделей.
- Сохранить индекс в `outputs/chroma/`.

Сгенерировать эмбеддинги для всех активных моделей:

```bash
mytho embeddings
```

Сгенерировать для конкретной модели:

```bash
mytho embeddings --model bge-m3
```

Пересоздать с нуля:

```bash
mytho embeddings --model bge-m3 --force
```

## projection

Модуль анализа эмбеддингов из Chroma DB и генерации HTML/CSV/JSON-артефактов в `outputs/analysis/`.

Основные файлы:
- `src/projection/loader.py` читает данные из Chroma.
- `src/projection/analyzer.py` собирает статистику.
- `src/projection/visualization.py` строит UMAP-проекцию, heatmap и distribution chart.
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
- `outputs/chroma/` — локальная Chroma DB с векторными коллекциями. Создается через `mytho embeddings`.
- `outputs/analysis/` — результаты анализа: `models.json`, HTML-графики, кластеризация. Создается через `mytho projection` и `mytho cluster`.
- `outputs/graphs/` — готовые HTML-графы персонажей и связей. Создается через `mytho graphs`.
- `outputs/logs/` — логи всех пайплайнов.

## Типовой пайплайн

Запустить всё одной командой:

```bash
mytho build --model bge-m3
```

Или по шагам:

```bash
# 1. Собрать корпус (Gutenberg-тексты очищаются автоматически)
mytho corpus --type all

# 2. Построить эмбеддинги и Chroma DB
mytho embeddings

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
