# План рефакторинга конфигурации

## Статус: ВЫПОЛНЕНО

- [x] Фаза 1: corpus — dataclass + YAML override
- [x] Фаза 2: graphs — dataclass, больше не падает без YAML
- [x] Фаза 3: projection — dataclass с backward-compatible API
- [x] Фаза 4: embedding — извлечение хардкода в ConfigManager
- [x] Фаза 5: server — ServerConfig dataclass
- [ ] Фаза 6: settings.py cleanup (setup_logging) — отложена, низкий приоритет

---

## Исходные проблемы (до рефакторинга)

### 1. Четыре разных паттерна конфигурации

| Модуль | Паттерн | Файл |
|--------|---------|------|
| `embedding` | ConfigManager с deep merge | `config_manager.py` (169 строк) |
| `projection` | AnalyzerConfig singleton с properties | `config.py` (133 строки) |
| `corpus` | `__getattr__` → settings | `config.py` (15 строк) |
| `graphs` | Прямой `yaml.safe_load()`, падает без файла | `run_graph_generation.py:18` |
| `server` | Frozen dataclass → settings | `config.py` (36 строк) |

### 2. Тройное дублирование значений

```
settings.py:21        default_embedding_model = "BAAI/bge-m3"
config_manager.py:25  "default_model": _settings.default_embedding_model  # берёт из settings
embedding.yaml:9  default_model: "BAAI/bge-m3"                  # дублирует
```

### 3. YAML-файлы config/ не загружаются модулями

- `config/corpus.yaml` — **не загружается нигде**. Значения захардкожены в `downloader.py`, `builder.py`.
- `config/projection.yaml` — загружается, но значения дублируют property-дефолты.
- `config/embedding.yaml` — загружается через ConfigManager, мержится поверх дефолтов.
- `config/graphs.yaml` — загружается напрямую, без дефолтов. Падает без файла.

### 4. Захардкоженные значения разбросаны по коду

| Файл | Строка | Значение | Есть в YAML? |
|------|--------|----------|--------------|
| `embedding/builder.py` | 62 | `BATCH_SIZE_THRESHOLDS = [(3072,8),(1024,16),(768,24)]` | Нет |
| `embedding/builder.py` | 89 | `ThreadPoolExecutor(max_workers=16)` | Нет |
| `corpus/builder.py` | 254 | `max_workers=10` | В YAML, но не читается |
| `corpus/downloader.py` | 17-19 | `total=4, backoff_factor=1.5` | В YAML, но не читается |
| `corpus/downloader.py` | 125 | `timeout=(10, 30)` | В YAML, но не читается |
| `graphs/llm_processing.py` | 17 | `max_retries=5` | Нет |
| `graphs/llm_processing.py` | 19 | `backoff_factor=5` | Нет |
| `graphs/llm_processing.py` | 33 | `temperature=0.1` | Нет |
| `projection/visualization.py` | 18-29 | `HEATMAP_WIDTH=1000`, `DASHBOARD_HEIGHT=700`, etc. | Нет |
| `server/run_server.py` | 70 | `host="127.0.0.1", port=8000` | Нет |
| `server/run_server.py` | 17 | `minimum_size=1024` (gzip) | Нет |
| `server/run_server.py` | 44 | `max-age=86400` | Нет |
| `server/api/similarity.py` | 19 | `_SEARCH_JOB_TTL_SECONDS=1800` | Нет |
| `projection/loader.py` | 165,215 | `batch_size=1000`, `batch_size=5000` | Нет |

---

## Целевая архитектура

### Принцип: dataclass с дефолтами + YAML override + закомментированный reference

```
Приоритет: env vars > YAML > dataclass defaults
```

### Три слоя

1. **`settings.py`** (Pydantic BaseSettings) — глобальные пути и env vars. Уже есть, не меняется.

2. **`<module>/config.py`** — dataclass со всеми настройками модуля и дефолтами в полях. Загружает YAML и перезаписывает поверх.

3. **`config/<module>.yaml`** — закомментированный reference + активные переопределения пользователя.

### Целевой паттерн для каждого модуля

```python
# <module>/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from settings import settings


@dataclass
class CorpusConfig:
    # build
    default_type: str = "all"
    max_workers: int = 10

    # downloader
    timeout_connect: int = 10
    timeout_read: int = 30
    retry_total: int = 4
    retry_backoff_factor: float = 1.5
    retry_status_forcelist: list[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])

    # parsing
    html_include_comments: bool = False
    html_include_tables: bool = True
    pdf_extract_tables: bool = False
    pdf_preserve_layout: bool = True


def _load_yaml_overrides(config_name: str) -> dict:
    """Load YAML config if it exists, return empty dict otherwise."""
    for candidate in [
        Path(f"config/{config_name}.yaml"),
        settings.project_root / "config" / f"{config_name}.yaml",
    ]:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def load_config() -> CorpusConfig:
    overrides = _load_yaml_overrides("corpus")
    # Flatten nested YAML into flat dataclass fields
    flat = {}
    for section in overrides.values():
        if isinstance(section, dict):
            flat.update(section)
        # top-level keys pass through
    return CorpusConfig(**{
        k: v for k, v in flat.items()
        if k in CorpusConfig.__dataclass_fields__
    })
```

### Целевой формат YAML-файлов

```yaml
# config/corpus.yaml
#
# Все настройки показаны с дефолтами из кода.
# Раскомментируйте и измените только то, что нужно.

# build:
#   default_type: "all"
#   max_workers: 10

# downloader:
#   timeout_connect: 10
#   timeout_read: 30
#   retry_total: 4
#   retry_backoff_factor: 1.5
#   retry_status_forcelist: [429, 500, 502, 503, 504]

# parsing:
#   html:
#     include_comments: false
#     include_tables: true
#   pdf:
#     extract_tables: false
#     preserve_layout: true
```

---

## План по модулям

### Фаза 1: `corpus` (самый простой, хороший пилот)

**Проблема:** YAML существует но не загружается. Значения захардкожены в `downloader.py` и `builder.py`.

**Шаги:**
1. Создать `corpus/config.py` — dataclass `CorpusConfig` с полями:
   - `max_workers: int = 10`
   - `timeout_connect: int = 10`
   - `timeout_read: int = 30`
   - `retry_total: int = 4`
   - `retry_backoff_factor: float = 1.5`
   - `retry_status_forcelist: list[int]`
   - `html_include_comments: bool = False`
   - `html_include_tables: bool = True`
   - `pdf_extract_tables: bool = False`
   - `pdf_preserve_layout: bool = True`
2. Добавить `load_config()` → загружает YAML, возвращает dataclass
3. В `downloader.py` — заменить хардкод на `config.timeout_connect`, `config.retry_total`, etc.
4. В `builder.py:254` — заменить `max_workers=10` на `config.max_workers`
5. Перевести `config/corpus.yaml` в закомментированный формат
6. Пути оставить в `settings.py` (через существующий `__getattr__` или прямой импорт)

**Файлы:**
- Переписать: `src/corpus/config.py`
- Изменить: `src/corpus/downloader.py`, `src/corpus/builder.py`
- Обновить: `config/corpus.yaml`

---

### Фаза 2: `graphs` (самый сломанный)

**Проблема:** Падает без YAML. Нет дефолтов. LLM-параметры захардкожены в `llm_processing.py`.

**Шаги:**
1. Создать `graphs/config.py` — dataclass `GraphsConfig` с полями:
   - `llm_mode: str = "local"`
   - `api_key: str = ""`, `api_model: str = "gpt-4o-mini"`, `api_base_url: str = "https://api.openai.com/v1"`, `api_json_mode: bool = True`
   - `local_api_key: str = "dummy-key"`, `local_model: str = "google/gemma-4-e4b"`, `local_base_url: str = "http://127.0.0.1:1234/v1/"`, `local_json_mode: bool = False`
   - `chunk_size: int = 4000`, `chunk_overlap: int = 1000`
   - `temperature: float = 0.1`
   - `max_retries: int = 5`, `retry_backoff_factor: float = 5.0`
   - `force_overwrite: bool = False`
2. Добавить `load_config()` с YAML override
3. В `run_graph_generation.py` — убрать `_load_config()`, использовать dataclass
4. В `llm_processing.py` — убрать хардкод `temperature`, `max_retries`, `backoff_factor`, принимать из конфига
5. Перевести `config/graphs.yaml` в закомментированный формат
6. LLM-секреты (`api_key`) — читать из env vars через `settings.py`

**Файлы:**
- Создать: `src/graphs/config.py`
- Изменить: `src/graphs/run_graph_generation.py`, `src/graphs/llm_processing.py`
- Обновить: `config/graphs.yaml`

---

### Фаза 3: `projection` (унификация)

**Проблема:** AnalyzerConfig — 133 строки с property-дефолтами + глобальный singleton + функции-обёртки. Дублирует YAML.

**Шаги:**
1. Переписать `projection/config.py` — dataclass `AnalyzerConfig`:
   - `umap_configs: list[dict]` — дефолт из текущих property
   - `tsne_configs: list[dict]` — дефолт из текущих property
   - `pca_configs: list[dict] = field(default_factory=lambda: [{}])`
   - `baseline_configs: dict` — дефолт из текущего property
   - Visualization constants: `heatmap_width: int = 1000`, `heatmap_height: int = 900`, `dashboard_height: int = 700`, `distribution_height: int = 600`, `distribution_width: int = 900`, `max_text_preview_len: int = 200`, `default_sample_size: int = -1`
   - Loader: `chroma_batch_size: int = 1000`, `default_batch_size: int = 5000`
2. Добавить `load_config()` с YAML override
3. Удалить singleton `_analyzer_config`, функции `get_analyzer_config()`, `get_chroma_path()`, etc.
4. В `visualization.py` — убрать module-level константы, читать из конфига
5. В `loader.py` — убрать хардкод `batch_size=1000` и `batch_size=5000`
6. Перевести `config/projection.yaml` в закомментированный формат
7. Пути оставить в `settings.py`

**Файлы:**
- Переписать: `src/projection/config.py`
- Изменить: `src/projection/visualization.py`, `src/projection/loader.py`, `src/projection/analyzer.py`
- Обновить: `config/projection.yaml`

---

### Фаза 4: `embedding` (рефакторинг ConfigManager)

**Проблема:** ConfigManager — самый зрелый, но 169 строк, dict-based (нет типизации), дублирует settings.

**Шаги:**
1. Переписать `ConfigManager` → dataclass `EmbeddingsBuilderConfig`:
   - `default_model: str = "BAAI/bge-m3"`
   - `default_chunking: str = "paragraph"`
   - `text_type: str = "all"`
   - `batch_size: int = 32`, `cache_batch_size: int = 50`, `chroma_batch_size: int = 100`
   - `max_workers: int = 16`, `queue_maxsize: int = 10`
   - Chunking: `fixed_chunk_size: int = 512`, `fixed_chunk_overlap: int = 64`, etc.
   - Cache: `cache_validation: str = "crc32"`, `cache_max_size_mb: int = 1024`, `cache_ttl_days: int = 30`
   - Performance: `enable_metrics: bool = True`, `track_memory: bool = True`
2. Сохранить `validate()` как метод dataclass
3. Сохранить CLI override через `load_config(config_path=...)` — опциональный путь к YAML
4. В `builder.py` — убрать `BATCH_SIZE_THRESHOLDS`, `DEFAULT_BATCH_SIZE`, `max_workers=16`, `maxsize=10` — брать из конфига
5. Перевести `config/embedding.yaml` в закомментированный формат
6. Пути оставить в `settings.py`

**Файлы:**
- Переписать: `src/embedding/config_manager.py` → `src/embedding/config.py`
- Изменить: `src/embedding/builder.py`, `src/embedding/cli.py`
- Обновить: `config/embedding.yaml`
- Обновить: `tests/test_config_manager.py`

---

### Фаза 5: `server` (добавить конфигурацию)

**Проблема:** Нет YAML-конфига. Host/port/gzip/cache захардкожены.

**Шаги:**
1. Расширить `server/config.py` — добавить к `ProjectPaths`:
   - `host: str = "127.0.0.1"`, `port: int = 8000`
   - `gzip_minimum_size: int = 1024`
   - `cache_max_age: int = 86400`
   - `search_job_ttl_seconds: int = 1800`
   - `similarity_max_workers: int = 1`
2. Загружать из env vars через `settings.py` (добавить `MYTHO_UI_HOST`, `MYTHO_UI_PORT`)
3. YAML для server **не создавать** — достаточно env vars для серверных настроек
4. В `run_server.py` — убрать хардкод, брать из конфига
5. В `api/similarity.py` — убрать `_SEARCH_JOB_TTL_SECONDS`, `max_workers=1`

**Файлы:**
- Изменить: `src/server/config.py`, `src/server/run_server.py`, `src/server/api/similarity.py`
- Изменить: `src/settings.py` (добавить `ui_host`, `ui_port`)

---

### Фаза 6: `settings.py` (cleanup)

**Проблема:** Содержит `setup_logging()` (75 строк) — не относится к настройкам.

**Шаги:**
1. Вынести `setup_logging()` в отдельный `src/log_setup.py`
2. Обновить импорты во всех модулях
3. `settings.py` остаётся чистым: только `Settings` класс и `settings = Settings()`

**Файлы:**
- Изменить: `src/settings.py`
- Создать: `src/log_setup.py`
- Обновить импорты: `projection/config.py`, `graphs/run_graph_generation.py`, `embedding/cli.py`

---

## Общая утилита загрузки YAML

Чтобы не дублировать логику загрузки в каждом модуле, вынести в `settings.py` или отдельный `config_loader.py`:

```python
# src/config_loader.py
from dataclasses import fields
from pathlib import Path
from typing import TypeVar, Type

import yaml

from settings import settings

T = TypeVar("T")


def load_yaml_config(config_cls: Type[T], config_name: str, config_path: str | None = None) -> T:
    """
    Load a dataclass config with YAML overrides.

    Priority: YAML values > dataclass defaults
    Paths always come from settings.py, not YAML.
    """
    overrides = _load_yaml(config_name, config_path)
    flat = _flatten(overrides)
    valid_fields = {f.name for f in fields(config_cls)}  # type: ignore[arg-type]
    kwargs = {k: v for k, v in flat.items() if k in valid_fields}
    return config_cls(**kwargs)  # type: ignore[return-value]


def _load_yaml(config_name: str, config_path: str | None) -> dict:
    candidates = [Path(config_path)] if config_path else [
        Path(f"config/{config_name}.yaml"),
        settings.project_root / "config" / f"{config_name}.yaml",
    ]
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def _flatten(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """Flatten nested dict: {a: {b: 1}} → {a_b: 1}"""
    items: list[tuple[str, object]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
```

---

## Порядок выполнения

```
Фаза 1: corpus     — пилот, самый простой модуль
Фаза 2: graphs    — починить падение без файла
Фаза 6: settings.py cleanup — вынести logging (нужно перед фазами 3-4)
Фаза 3: projection  — унификация singleton → dataclass
Фаза 4: embedding  — рефакторинг ConfigManager → dataclass
Фаза 5: server           — добавить недостающую конфигурацию
```

Каждая фаза — отдельный коммит. Тесты запускаются после каждой фазы.

---

## Что НЕ меняется

- **`settings.py` Settings class** — глобальные пути и env vars остаются как есть
- **`.env` / `config/.env`** — механизм env vars через Pydantic не меняется
- **`config/details/`** — справочные JSON-файлы (traditions.json, download_list.json) — не конфигурация приложения
- **`config/graphs_prompts.json`** — промпты, не настройки
- **Visualization constants** (`GRID_COLOR`, `ZERO_LINE_COLOR`) — визуальные константы остаются в коде, не выносятся в конфиг (это стилистика, не настройки)

## Результат

После рефакторинга:
- Каждый модуль имеет **типизированный dataclass** с дефолтами → IDE подсказки, mypy проверки
- YAML-файлы — **закомментированный reference** с активными override'ами → пользователь видит все опции
- Приложение **работает без YAML** — дефолты в коде
- **Один паттерн** для всех модулей → предсказуемость
- Захардкоженных значений в бизнес-коде **нет** → все настройки в одном месте на модуль
