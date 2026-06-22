# How To: MythoScope

Краткая карта проекта: что делает каждый модуль, какие файлы он читает и пишет, и как его запускать. Все команды ниже предполагают запуск из корня проекта.

## Структура проекта

```
config/          — статические конфиги, шаблоны, models.json, corpus.json
outputs/         — всё, что генерируется при запуске (corpus, embeddings, projections, …)
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

Часть команд скачивает модели, обращается к внешним сайтам или пишет большие артефакты в `outputs/embeddings/`, `outputs/projections/`, `outputs/corpus/`, `outputs/graphs/` и `outputs/logs/`.

## Конфигурация

- **`src/settings.py`** — единый источник путей и параметров. Все директории (`outputs/corpus`, `outputs/embeddings`, …), параметры chunking, LLM, сервера и т.д. Переопределяется через переменные окружения с префиксом `MYTHO_` или файл `.env` / `config/.env` (например, `MYTHO_CORPUS_DIR=/data/corpus`). Вложенные параметры через `__`: `MYTHO_LLM__MODEL=gpt4o-mini`. Полный список переменных — в `.env.example`.
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
mytho projections --help
mytho graphs --help
mytho server --help
mytho build --help
mytho status
mytho clean
```

## corpus

Модуль сборки корпуса из `config/download_list.json`. Тексты с Project Gutenberg автоматически очищаются от лицензионных заголовков и хвостов при скачивании.

Основные файлы:
- `src/corpus/downloader.py` скачивает источники.
- `src/corpus/extraction.py` извлекает текст из HTML/PDF/TXT.
- `src/corpus/utils.py` утилиты: пути, нормализация текста, подсчёт слов/предложений, работа с традициями.
- `src/corpus/iterator.py` итерация по файлам корпуса (`iter_files`, `CorpusFileInfo`).
- `src/corpus/builder.py` строит структуру `outputs/corpus/`, метаданные и каталог.
- `src/corpus/clean_gutenberg.py` автоматически удаляет Gutenberg-боллерплейт.

Возможности:
- Скачать и обработать источники.
- Автоматически очистить Gutenberg-тексты (по маркерам в содержимом).
- Сохранить тексты в `outputs/corpus/<major>/<tradition>/<title>.txt`.
- Создать `outputs/corpus/corpus.json` (метаданные), `outputs/corpus/traditions.json`.

Запуск сборки корпуса:

```bash
mytho corpus
```

Пересобрать с перезаписью:

```bash
mytho corpus --force
```

## embedding

Модуль генерации эмбеддингов и записи в Chroma DB.

Основные файлы:
- `src/embeddings/build_embeddings.py` оркестрирует генерацию для нескольких моделей (skip/resume по метаданным коллекции).
- `src/embeddings/builder.py` читает корпус, режет тексты на чанки, считает эмбеддинги и пишет в Chroma.
- `src/embeddings/chunking.py` содержит стратегии chunking (character, sentence, paragraph).
- `src/embeddings/chroma_manager.py` хранилище ChromaDB: module-level функции (создание/удаление коллекций, список моделей) и `ChromaCollection` (upsert, загрузка данных, existing_ids).
- `src/embeddings/model_manager.py` загрузка/выгрузка SentenceTransformer моделей (`EmbeddingEncoder`).
- `src/model_registry.py` резолвит алиасы моделей из `config/models.json`.

Возможности:
- Построить эмбеддинги для нескольких моделей.
- Сохранить индекс в `outputs/embeddings/`.

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

## projections

Модуль анализа эмбеддингов из Chroma DB и генерации JSON-артефактов в `outputs/projections/`.

Основные файлы:
- `src/projections/analyzer.py` загружает данные из Chroma и собирает статистику.
- `src/projections/visualization.py` вычисляет UMAP-проекции, heatmap расстояний и distribution, сохраняет как JSON.
- `src/projections/run_analysis.py` оркестрирует анализ для нескольких моделей.

Возможности:
- Получить статистику по модели.
- Сохранить `model_info.json`, `models.json`, `embeddings_data.csv`.
- Построить интерактивные графики семантического пространства.

Запустить анализ всех доступных моделей:

```bash
mytho projections
```

Запустить анализ одной модели:

```bash
mytho projections --model bge-m3
```

## graphs

Модуль извлечения персонажей, отношений, мест и времени через LLM и генерации графов.

Основные файлы:
- `config/graphs_prompts.json` содержит промпты для извлечения сущностей.
- `src/graphs/run_graph_generation.py` оркестрирует генерацию: итерация по текстам, чанкинг, агрегация.
- `src/graphs/extraction.py` извлекает сущности через LLM и дедуплицирует их.
- `src/graphs/chunking.py` разбивает тексты на чанки с перекрытием.
- `src/graphs/checkpointing.py` сохранение/загрузка промежуточных результатов.
- `src/graphs/graph_generator.py` строит граф через NetworkX и сохраняет JSON.
- `src/llm_client.py` вызывает OpenAI-compatible API (`LLMProcessor`).

Возможности:
- Пройти по книгам из `outputs/corpus/corpus.json`.
- Извлечь сущности и связи через локальный или внешний LLM.
- Сохранить три графа на книгу в `outputs/graphs/<text_id>/`:
  - `characters.json` — персонажи и отношения между ними.
  - `realms.json` — локации и связи смежности (поле "Adjacent to").
  - `ages.json` — эпохи, связанные через общих ключевых персонажей (KeyActors).

Запуск с моделью по умолчанию (из `settings.py` → `llm.model`):

```bash
mytho graphs
```

Запуск с конкретной LLM из реестра `config/models.json`:

```bash
mytho graphs --model gemini25-flash
```

Запуск с перезаписью готовых графов:

```bash
mytho graphs --force
```

## status

Показывает текущее состояние пайплайна: что построено, чего не хватает, размеры на диске.

```bash
mytho status
```

Вывод содержит секции Corpus, Embeddings, Projections, Graphs с итоговым размером.

## clean

Поиск и удаление осиротевших файлов: corpus-тексты без записи в каталоге, embedding-коллекции без модели в реестре, чанки без текста в корпусе, projections-директории без коллекции, graph-директории без текста в корпусе.

По умолчанию — dry run (только показывает что будет удалено):

```bash
mytho clean
```

Удалить:

```bash
mytho clean --apply
```

## server

FastAPI-сервер и SPA-интерфейс.

Основные файлы:
- `src/server/run_server.py` создание приложения, middleware, статика.
- `src/server/api/models.py` список доступных embedding-моделей.
- `src/server/api/corpus.py` каталог текстов, чтение документов, скачивание архива.
- `src/server/api/geography.py` традиции с географией.
- `src/server/api/graphs.py` данные графов (персонажи, связи).
- `src/server/api/projections.py` JSON-данные проекций.
- `src/server/api/points.py` информация о точках эмбеддингов и соседи.
- `src/server/api/search.py` семантический поиск (синхронный и асинхронный), warmup.
- `src/server/schemas.py` Pydantic-схемы запросов и ответов.
- `src/server/services/` сервисный слой (кэширование каталога, ZIP-архив, загрузка проекций и индексов).

### API эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/api/models` | Список embedding-моделей |
| GET | `/api/corpus/catalog` | Каталог текстов корпуса |
| GET | `/api/corpus/documents` | Текст документа по title |
| GET | `/api/corpus/archive` | ZIP-архив корпуса |
| GET | `/api/geography/traditions` | Традиции с координатами |
| GET | `/api/graphs/` | Список книг с доступными графами |
| GET | `/api/graphs/{book_id}/{graph_type}` | JSON-данные графа (nodes + edges) |
| GET | `/api/similarity/projections/{model}/{method}` | JSON-данные проекции |
| GET | `/api/similarity/points/{model}/{text_id}` | Информация о точке (+ соседи через `?chunk_index=N&neighbors=N&offset=M`) |
| POST | `/api/similarity/search` | Семантический поиск |
| POST | `/api/similarity/search/warmup` | Предзагрузка модели и индекса |

### Интерактивная документация

FastAPI автоматически генерирует документацию из Pydantic-схем:

- `http://localhost:8000/docs` — Swagger UI (интерактивное тестирование эндпоинтов)
- `http://localhost:8000/redoc` — ReDoc (читаемая документация)
- `http://localhost:8000/openapi.json` — OpenAPI-схема (для кодогенерации или Postman)

### Запуск

```bash
mytho server
```

С явным указанием хоста и порта:

```bash
mytho server --host 0.0.0.0 --port 9000
```

Открыть интерфейс: `http://127.0.0.1:8000/`.

### Публикация в интернет (Caddy)

По умолчанию сервер слушает `127.0.0.1` и недоступен извне. Для публичного доступа используйте [Caddy](https://caddyserver.com) как reverse proxy — он автоматически получает и обновляет HTTPS-сертификаты от Let's Encrypt.

Установка (Ubuntu/Debian):

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install caddy
```

Создайте `/etc/caddy/Caddyfile`:

```
mythoscope.example.com {
    reverse_proxy localhost:8000
}
```

Запуск:

```bash
sudo systemctl enable caddy
sudo systemctl start caddy
```

Caddy сам получит TLS-сертификат, настроит редирект HTTP → HTTPS и проксирует запросы в uvicorn. В проекте ничего менять не нужно — `mytho server` продолжает слушать localhost.

## server/web

Современный SPA-фронтенд.

Основные файлы:
- `index.html` подключает стили и JS.
- `assets/app.js` содержит маршруты и экраны.
- `assets/core.js` содержит API helpers и состояние.
- `assets/app.css` содержит стили.

Запускается через:

```bash
mytho server
```

## Директории outputs/

Все генерируемые данные хранятся в `outputs/`:

- `outputs/corpus/` — основной текстовый корпус с метаданными (`corpus.json`) и описаниями традиций (`traditions.json`). Создается через `mytho corpus`.
- `outputs/embeddings/` — локальная Chroma DB с векторными коллекциями. Создается через `mytho embeddings`.
- `outputs/projections/` — результаты анализа: JSON-данные проекций (UMAP, heatmap, distribution). Создается через `mytho projections`.
- `outputs/graphs/` — JSON-графы (characters, realms, ages) для каждого текста. Создается через `mytho graphs`.
- `outputs/logs/` — логи всех пайплайнов.

## Типовой пайплайн

Запустить всё одной командой:

```bash
mytho build --model bge-m3
```

Или по шагам:

```bash
# 1. Собрать корпус (Gutenberg-тексты очищаются автоматически)
mytho corpus

# 2. Построить эмбеддинги
mytho embeddings

# 3. Построить визуальный анализ эмбеддингов
mytho projections

# 4. Извлечь графы персонажей через LLM
mytho graphs

# 5. Запустить веб-интерфейс
mytho server
```

Каждый шаг идемпотентен — если результат уже есть, он будет пропущен. Для принудительной пересборки:

```bash
mytho build --force
```

Проверить состояние и найти мусор:

```bash
mytho status
mytho clean
```
