import logging
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_distances

from settings import settings

from .utils import reduce_dimensions

MAX_TEXT_PREVIEW_LEN = 200
RANDOM_SEED = 42
HEATMAP_WIDTH = 1000
HEATMAP_HEIGHT = 900
DISTRIBUTION_HEIGHT = 600
DISTRIBUTION_WIDTH = 900
GRID_COLOR = "rgba(190,200,210,0.45)"
ZERO_LINE_COLOR = "rgba(120,130,140,0.55)"
AXIS_LINE_COLOR = "rgba(120,130,140,0.65)"

logger = logging.getLogger(__name__)


def _sample_for_visualization(data: list[dict], sample_size: int | None, reason: str) -> list[dict]:
    if sample_size is None or len(data) <= sample_size:
        return data

    logger.info(f"Sampling {sample_size} of {len(data)} records for {reason}")
    indices = np.random.default_rng(RANDOM_SEED).choice(len(data), sample_size, replace=False)
    return [data[i] for i in indices]


def _get_color_map(data: list[dict]) -> dict[str, str]:
    color_map = {}
    for item in data:
        tradition = item.get("tradition", "unknown")
        color = item.get("color")
        if tradition not in color_map and color:
            color_map[tradition] = color

    base_colors = px.colors.qualitative.Plotly
    unique_traditions = sorted(set(item.get("tradition", "unknown") for item in data))

    for i, trad in enumerate(unique_traditions):
        if trad not in color_map:
            color_map[trad] = base_colors[i % len(base_colors)]

    return color_map


def _reduce_dimensions_safe(
    embeddings: np.ndarray, n_components: int = 2, reducer_kwargs: dict[str, Any] | None = None
) -> np.ndarray | None:
    try:
        return reduce_dimensions(embeddings, n_components=n_components, **(reducer_kwargs or {}))
    except Exception:
        logger.exception("Dimension reduction failed")
        return None


def _cartesian_axis(title: str, tickangle: int = 0, showticklabels: bool = True) -> dict[str, Any]:
    return dict(
        title=dict(text=title, font=dict(size=12)),
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        zeroline=True,
        zerolinecolor=ZERO_LINE_COLOR,
        zerolinewidth=1,
        showline=True,
        linecolor=AXIS_LINE_COLOR,
        mirror=True,
        ticks="outside",
        tickfont=dict(size=10),
        tickangle=tickangle,
        showticklabels=showticklabels,
    )


def plot_interactive_2d(
    data: list[dict],
    save_html: bool = True,
    output_dir: Path | None = None,
    model_name: str | None = None,
    reducer_kwargs: dict[str, Any] | None = None,
) -> go.Figure | None:
    if not data:
        logger.warning("No data for visualization.")
        return None

    if output_dir is None:
        output_dir = settings.analysis_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reducer_kwargs = reducer_kwargs or {}

    try:
        embeddings = np.stack([item["embedding"] for item in data])
    except Exception:
        logger.exception("Failed to stack embeddings")
        return None

    embedding_2d = _reduce_dimensions_safe(embeddings, n_components=2, reducer_kwargs=reducer_kwargs)
    if embedding_2d is None:
        return None

    fig = go.Figure()
    color_map = _get_color_map(data)

    for tradition, color in color_map.items():
        indices = [i for i, item in enumerate(data) if item.get("tradition", "unknown") == tradition]
        if not indices:
            continue

        customdata = []
        for i in indices:
            item = data[i]
            text_preview = item.get("text", "")[:MAX_TEXT_PREVIEW_LEN]
            if len(item.get("text", "")) > MAX_TEXT_PREVIEW_LEN:
                text_preview += "..."
            text_preview = "<br>".join(textwrap.wrap(text_preview, width=60))
            customdata.append([item.get("id", "unknown"), tradition, item.get("chunk_index", 0), text_preview])

        fig.add_trace(
            go.Scatter(
                x=embedding_2d[indices, 0],
                y=embedding_2d[indices, 1],
                mode="markers",
                name=tradition,
                marker=dict(size=8, opacity=0.7, color=color, line=dict(width=1, color="white")),
                customdata=customdata,
                hovertemplate="<b>%{customdata[1]}</b><br>"
                "ID: %{customdata[0]}<br>"
                "Chunk: %{customdata[2]}<br>"
                "Text: %{customdata[3]}<extra></extra>",
            )
        )

    title = "UMAP visualization by tradition"
    if model_name:
        title += f" - {model_name}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Arial, sans-serif"), x=0.5, xanchor="center"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=11),
        ),
        hovermode="closest",
        margin=dict(l=50, r=50, t=80, b=100),
        plot_bgcolor="rgba(240,240,240,0.5)",
        paper_bgcolor="white",
        xaxis=_cartesian_axis("UMAP component 1"),
        yaxis=_cartesian_axis("UMAP component 2"),
    )

    if save_html and output_dir:
        html_path = output_dir / "umap_2d_traditions.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
        logger.info(f"Saved: {html_path}")

    return fig


def plot_distance_heatmap(
    data: list[dict], output_dir: Path | None = None, model_name: str | None = None, save_html: bool = True
) -> go.Figure | None:
    if not data:
        logger.warning("No data for visualization.")
        return None

    if output_dir is None:
        output_dir = settings.analysis_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    traditions_data: dict[str, list] = {}
    for item in data:
        trad = item.get("tradition", "unknown")
        if trad not in traditions_data:
            traditions_data[trad] = []
        traditions_data[trad].append(item["embedding"])

    centroids = {}
    for trad, embeddings in traditions_data.items():
        centroids[trad] = np.mean(embeddings, axis=0)

    trad_list = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[trad] for trad in trad_list])
    distance_matrix = cosine_distances(centroid_matrix, centroid_matrix)

    fig = px.imshow(
        distance_matrix,
        x=trad_list,
        y=trad_list,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="Viridis",
        title=f"Heatmap of distances between traditions{' - ' + model_name if model_name else ''}",
        labels=dict(x="Tradition", y="Tradition", color="Cosine distance"),
    )

    fig.update_layout(
        width=HEATMAP_WIDTH,
        height=HEATMAP_HEIGHT,
        title=dict(font=dict(size=16), x=0.5, xanchor="center"),
        xaxis=dict(
            title=dict(text="Tradition", font=dict(size=12)),
            tickangle=45,
            tickfont=dict(size=11),
            side="bottom",
            showgrid=False,
        ),
        yaxis=dict(tickfont=dict(size=11), title=dict(text="Tradition", font=dict(size=12)), showgrid=False),
        margin=dict(l=100, r=50, t=80, b=150),
    )

    fig.update_traces(
        textfont=dict(size=10, color="white" if distance_matrix.max() > 0.5 else "black"),
        hovertemplate="Distance between %{x} and %{y}: %{z:.4f}<extra></extra>",
    )

    if save_html and output_dir:
        output_path = output_dir / "distance_heatmap.html"
        fig.write_html(str(output_path))
        logger.info(f"Heatmap saved: {output_path}")

    return fig


def plot_tradition_distribution(
    data: list[dict], output_dir: Path | None = None, model_name: str | None = None, save_html: bool = True
) -> go.Figure | None:
    if not data:
        logger.warning("No data for visualization.")
        return None

    if output_dir is None:
        output_dir = settings.analysis_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tradition_counts: dict[str, int] = {}
    tradition_docs: dict[str, set] = {}
    for item in data:
        trad = item.get("tradition", "unknown")
        tradition_counts[trad] = tradition_counts.get(trad, 0) + 1
        tradition_docs.setdefault(trad, set()).add(item.get("id", "unknown"))

    sorted_traditions = sorted(tradition_counts.items(), key=lambda x: -x[1])
    traditions = [t[0] for t in sorted_traditions]
    counts = [t[1] for t in sorted_traditions]
    total_chunks = sum(counts)
    percentages = [(count / total_chunks * 100) if total_chunks else 0 for count in counts]
    doc_counts = [len(tradition_docs.get(trad, set())) for trad in traditions]

    color_map = _get_color_map(data)
    colors = [color_map[t] for t in traditions]

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts,
                y=traditions,
                orientation="h",
                marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.9)", width=1)),
                customdata=np.column_stack([percentages, doc_counts]),
                text=[f"{count:,} chunks ({pct:.1f}%)" for count, pct in zip(counts, percentages, strict=False)],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>"
                "Chunks: %{x:,}<br>"
                "Share: %{customdata[0]:.2f}%<br>"
                "Source texts: %{customdata[1]}<extra></extra>",
            )
        ]
    )

    title = f"Distribution of chunks by tradition{' - ' + model_name if model_name else ''}"
    fig.update_layout(
        title=dict(text=title, font=dict(size=16), x=0.5, xanchor="center"),
        showlegend=False,
        height=max(DISTRIBUTION_HEIGHT, 28 * len(traditions) + 180),
        width=max(DISTRIBUTION_WIDTH, 1050),
        margin=dict(l=180, r=160, t=90, b=80),
        plot_bgcolor="rgba(248,249,250,0.95)",
        paper_bgcolor="white",
        xaxis=_cartesian_axis("Number of chunks"),
        yaxis=dict(
            title=dict(text="Tradition", font=dict(size=12)),
            autorange="reversed",
            showgrid=False,
            showline=True,
            linecolor=AXIS_LINE_COLOR,
            ticks="outside",
            tickfont=dict(size=11),
        ),
    )

    if save_html and output_dir:
        output_path = output_dir / "tradition_distribution.html"
        fig.write_html(str(output_path))
        logger.info(f"Distribution chart saved: {output_path}")

    return fig


def add_click_handler_to_html(html_path: str) -> None:
    try:
        with open(html_path, encoding="utf-8") as f:
            content = f.read()

        if "pointClickHandler" in content:
            return

        click_handler_js = """
        <script>
        (function() {
            function sendPointClick(pointId) {
                if (window.parent !== window) {
                    window.parent.postMessage({
                        type: 'pointClicked',
                        pointId: pointId
                    }, '*');
                }
            }

            function addClickHandler() {
                setTimeout(function() {
                    var plotDiv = document.querySelector('.plotly-graph-div') || document.getElementById('plotly-graph');
                    if (plotDiv && plotDiv.on) {
                        plotDiv.on('plotly_click', function(data) {
                            if (data.points && data.points[0] && data.points[0].customdata) {
                                var pointId = data.points[0].customdata[0];
                                if (pointId) {
                                    sendPointClick(pointId);
                                }
                            }
                        });
                    } else {
                        setTimeout(addClickHandler, 500);
                    }
                }, 1000);
            }

            addClickHandler();
        })();
        </script>
        </body>
        """

        if "</body>" in content:
            content = content.replace("</body>", click_handler_js)
        else:
            content += click_handler_js

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Added click handler to {html_path}")
    except Exception as e:
        logger.warning(f"Failed to add click handler to {html_path}: {e}")
