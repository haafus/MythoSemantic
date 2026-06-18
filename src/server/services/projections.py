import base64
import json
import re
import struct
from functools import lru_cache
from pathlib import Path

from model_registry import key_to_model
from server.services.models import get_model_output_dir

SAVED_HTML_METHOD_FILES = {
    "umap": "umap_2d_traditions.html",
    "residual_umap": "residual_umap_2d.html",
    "residual_normalized_umap": "residual_normalized_umap_2d.html",
    "rlace_umap": "rlace_umap_2d.html",
    "story_umap": "story_umap_2d.html",
    "motif_umap": "motif_umap_2d.html",
    "distance_heatmap": "distance_heatmap.html",
    "tradition_distribution": "tradition_distribution.html",
}

INTERACTIVE_SAVED_HTML_METHODS = {"umap", "residual_umap", "residual_normalized_umap", "rlace_umap", "story_umap", "motif_umap"}

PLOTLY_DTYPE_FORMATS = {
    "f4": "f",
    "f8": "d",
    "i1": "b",
    "u1": "B",
    "i2": "h",
    "u2": "H",
    "i4": "i",
    "u4": "I",
    "i8": "q",
    "u8": "Q",
}


def get_projection_data(model_key: str, method: str) -> dict | None:
    output_dir = get_model_output_dir(model_key)
    candidates = [
        output_dir / f"{method}_2d_coords.json",
        output_dir / f"{method}_coords.json",
    ]

    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

    if method in INTERACTIVE_SAVED_HTML_METHODS:
        saved_html_plot = get_saved_html_plot(model_key, method, output_dir=output_dir)
        saved_html_path = saved_html_plot.get("path")
        if saved_html_plot.get("exists") and saved_html_path:
            path = Path(saved_html_path)
            if path.exists():
                return _load_saved_html_projection(
                    str(path),
                    path.stat().st_mtime,
                    key_to_model(model_key),
                    method,
                )

    return None


def get_saved_html_plot(model_key: str, method: str, output_dir: Path | None = None) -> dict:
    safe_dir = model_key
    if output_dir is None:
        output_dir = get_model_output_dir(model_key)
    filename = SAVED_HTML_METHOD_FILES.get(method, f"{method}.html")
    html_path = output_dir / filename

    if not html_path.exists():
        return {
            "exists": False,
            "reason": f"Saved HTML plot not found for {method}",
        }

    return {
        "exists": True,
        "url": f"/projections/{safe_dir}/{filename}",
        "path": str(html_path),
    }


@lru_cache(maxsize=32)
def _load_saved_html_projection(path: str, _mtime: float, model_name: str, method: str) -> dict | None:
    html = Path(path).read_text(encoding="utf-8", errors="replace")
    traces = _extract_plotly_traces(html)
    if not traces:
        return None

    points = []
    for trace in traces:
        xs = _plotly_array_values(trace.get("x"))
        ys = _plotly_array_values(trace.get("y"))
        customdata = _plotly_array_values(trace.get("customdata"))
        trace_name = trace.get("name") or "Unknown"

        count = min(len(xs), len(ys))
        for index in range(count):
            custom = customdata[index] if index < len(customdata) and isinstance(customdata[index], list) else []
            point_id = _custom_value(custom, 0, "")
            tradition = _custom_value(custom, 1, trace_name) or trace_name
            chunk_index = _as_int(_custom_value(custom, 2, 0))

            points.append(
                {
                    "id": str(point_id),
                    "tradition": str(tradition or "Unknown"),
                    "chunk_index": chunk_index,
                    "text": _clean_saved_preview_text(_custom_value(custom, 3, "")),
                    "x": xs[index],
                    "y": ys[index],
                }
            )

    if not points:
        return None

    return {
        "model": model_name,
        "method": method,
        "points": points,
        "source": "saved_html",
    }


def _extract_plotly_traces(html: str) -> list | None:
    call_start = html.rfind("Plotly.newPlot(")
    if call_start == -1:
        return None

    array_start = html.find("[", call_start)
    if array_start == -1:
        return None

    traces_json = _extract_balanced_json(html, array_start, "[", "]")
    if not traces_json:
        return None

    try:
        traces = json.loads(traces_json)
    except json.JSONDecodeError:
        return None

    return traces if isinstance(traces, list) else None


def _plotly_array_values(value) -> list:
    if isinstance(value, list):
        return value

    if not isinstance(value, dict):
        return []

    if isinstance(value.get("bdata"), str):
        return _decode_plotly_typed_array(value)

    numeric_keys = sorted(
        (key for key in value if str(key).isdigit()),
        key=lambda item: int(item),
    )
    if numeric_keys:
        return [value[key] for key in numeric_keys]

    return []


def _decode_plotly_typed_array(value: dict) -> list:
    dtype = str(value.get("dtype") or "").lower()
    dtype_key = dtype.lstrip("<>=")
    fmt = PLOTLY_DTYPE_FORMATS.get(dtype_key)
    if not fmt:
        return []

    endian = ">" if dtype.startswith(">") else "<"
    try:
        raw = base64.b64decode(value["bdata"])
    except (TypeError, ValueError):
        return []

    item_size = struct.calcsize(endian + fmt)
    usable_size = len(raw) - (len(raw) % item_size)
    if usable_size <= 0:
        return []

    values = [item[0] for item in struct.iter_unpack(endian + fmt, raw[:usable_size])]
    return _reshape_plotly_array(values, value.get("shape"))


def _reshape_plotly_array(values: list, shape) -> list:
    if not shape:
        return values

    try:
        dimensions = [int(part.strip()) for part in str(shape).replace("x", ",").split(",") if part.strip()]
    except ValueError:
        return values

    if len(dimensions) < 2:
        return values

    rows, columns = dimensions[0], dimensions[1]
    if rows <= 0 or columns <= 0:
        return values

    return [values[index * columns : (index + 1) * columns] for index in range(min(rows, len(values) // columns))]


def _extract_balanced_json(text: str, start: int, open_char: str, close_char: str) -> str | None:
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _custom_value(values: list, index: int, default: float) -> float:
    return values[index] if index < len(values) else default


def _clean_saved_preview_text(value) -> str:
    text = str(value or "")
    text = re.sub(r"-\s*&lt;br\s*/?&gt;", "-", text, flags=re.IGNORECASE)
    text = re.sub(r"-\s*<br\s*/?>", "-", text, flags=re.IGNORECASE)
    text = re.sub(r"&lt;br\s*/?&gt;", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
