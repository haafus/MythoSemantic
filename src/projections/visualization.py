import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import LabelEncoder, Normalizer

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



def _get_color_map(data: list[dict]) -> dict[str, str]:
    traditions_path = settings.corpus_dir / "traditions.json"
    traditions_info: dict = {}
    if traditions_path.exists():
        try:
            with open(traditions_path, encoding="utf-8") as f:
                traditions_info = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    color_map = {}
    for name, info in traditions_info.items():
        if info.get("color"):
            color_map[name] = info["color"]

    base_colors = px.colors.qualitative.Plotly
    unique_traditions = sorted(set(item.get("tradition", "unknown") for item in data))

    for i, trad in enumerate(unique_traditions):
        if trad not in color_map:
            color_map[trad] = base_colors[i % len(base_colors)]

    return color_map


def _reduce_dimensions_safe(
    embeddings: np.ndarray, n_components: int = 2,
) -> np.ndarray | None:
    from settings import settings

    cfg = settings.projection
    try:
        return reduce_dimensions(
            embeddings, n_components=n_components,
            n_neighbors=cfg.umap_n_neighbors, min_dist=cfg.umap_min_dist,
        )
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


def _compute_tradition_residuals(data: list[dict], embeddings: np.ndarray) -> np.ndarray:
    traditions = [item.get("tradition", "unknown") for item in data]
    unique_traditions = set(traditions)
    centroids = {}
    for trad in unique_traditions:
        mask = [i for i, t in enumerate(traditions) if t == trad]
        centroids[trad] = embeddings[mask].mean(axis=0)
    residuals = np.empty_like(embeddings)
    for i, trad in enumerate(traditions):
        residuals[i] = embeddings[i] - centroids[trad]
    return residuals


def _concept_erasure(data: list[dict], embeddings: np.ndarray, max_iters: int = 35) -> np.ndarray:
    """INLP: iteratively project out tradition-predictive directions (Ravfogel et al., ACL 2020)."""
    traditions = [item.get("tradition", "unknown") for item in data]
    le = LabelEncoder()
    y = le.fit_transform(traditions)
    n_classes = len(le.classes_)
    chance = 1.0 / n_classes

    X = embeddings.astype(np.float64)
    d = X.shape[1]
    P = np.eye(d, dtype=np.float64)

    for i in range(max_iters):
        X_proj = X @ P
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
        clf.fit(X_proj, y)
        acc = clf.score(X_proj, y)
        print(f"  INLP iteration {i + 1}/{max_iters}: accuracy = {acc:.3f} (chance = {chance:.3f})", end="\r", flush=True)

        if acc < chance + 0.03:
            print()
            logger.info(f"  INLP converged after {i + 1} iterations")
            break

        W = clf.coef_
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        basis = Vt[S > 1e-10]
        rowspace_proj = basis.T @ basis
        P = P @ (np.eye(d, dtype=np.float64) - rowspace_proj)
    else:
        print()

    return (X @ P).astype(embeddings.dtype)


def _plot_umap_scatter(
    data: list[dict],
    embeddings: np.ndarray,
    title_prefix: str,
    filename: str,
    axis_prefix: str,
    output_dir: Path | None = None,
    model_name: str | None = None,
) -> None:
    if output_dir is None:
        output_dir = settings.projections_dir

    embedding_2d = _reduce_dimensions_safe(embeddings, n_components=2)
    if embedding_2d is None:
        return

    if output_dir:
        points = []
        for i, item in enumerate(data):
            text_preview = item.get("text", "")[:MAX_TEXT_PREVIEW_LEN]
            if len(item.get("text", "")) > MAX_TEXT_PREVIEW_LEN:
                text_preview += "..."
            points.append({
                "id": item["text_id"],
                "tradition": item.get("tradition", "unknown"),
                "chunk_index": item["chunk_index"],
                "text": text_preview,
                "x": round(float(embedding_2d[i, 0]), 6),
                "y": round(float(embedding_2d[i, 1]), 6),
            })
        json_path = output_dir / filename
        json_path.write_text(json.dumps({"model": model_name or "", "points": points}), encoding="utf-8")
        logger.info(f"Saved: {json_path}")


def plot_interactive_2d(
    data: list[dict],
    embeddings: np.ndarray,
    output_dir: Path | None = None,
    model_name: str | None = None,
) -> None:
    _plot_umap_scatter(
        data, embeddings,
        title_prefix="UMAP visualization by tradition",
        filename="umap_2d_traditions.json",
        axis_prefix="UMAP",
        output_dir=output_dir,
        model_name=model_name,
    )


def plot_residual_umap(
    data: list[dict],
    embeddings: np.ndarray,
    output_dir: Path | None = None,
    model_name: str | None = None,
) -> None:
    residuals = _compute_tradition_residuals(data, embeddings)
    _plot_umap_scatter(
        data, residuals,
        title_prefix="Residual UMAP (tradition centroid removed)",
        filename="residual_umap_2d.json",
        axis_prefix="Residual UMAP",
        output_dir=output_dir,
        model_name=model_name,
    )


def plot_residual_normalized_umap(
    data: list[dict],
    embeddings: np.ndarray,
    output_dir: Path | None = None,
    model_name: str | None = None,
) -> None:
    residuals = _compute_tradition_residuals(data, embeddings)
    residuals = Normalizer(norm="l2").fit_transform(residuals)
    _plot_umap_scatter(
        data, residuals,
        title_prefix="Residual Normalized UMAP (tradition centroid removed, L2-normalized)",
        filename="residual_normalized_umap_2d.json",
        axis_prefix="Residual Normalized UMAP",
        output_dir=output_dir,
        model_name=model_name,
    )


def plot_rlace_umap(
    data: list[dict],
    embeddings: np.ndarray,
    output_dir: Path | None = None,
    model_name: str | None = None,
) -> None:
    erased = _concept_erasure(data, embeddings)
    _plot_umap_scatter(
        data, erased,
        title_prefix="RLACE UMAP (tradition signal erased via INLP)",
        filename="rlace_umap_2d.json",
        axis_prefix="RLACE UMAP",
        output_dir=output_dir,
        model_name=model_name,
    )


def plot_distance_heatmap(
    data: list[dict], embeddings: np.ndarray,
    output_dir: Path | None = None, model_name: str | None = None, save_html: bool = True
) -> go.Figure | None:
    if output_dir is None:
        output_dir = settings.projections_dir

    traditions_data: dict[str, list] = {}
    for item, emb in zip(data, embeddings, strict=True):
        trad = item.get("tradition", "unknown")
        if trad not in traditions_data:
            traditions_data[trad] = []
        traditions_data[trad].append(emb)

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
    data: list[dict], embeddings: np.ndarray,
    output_dir: Path | None = None, model_name: str | None = None, save_html: bool = True
) -> go.Figure | None:
    if output_dir is None:
        output_dir = settings.projections_dir

    tradition_counts: dict[str, int] = {}
    tradition_docs: dict[str, set] = {}
    for item in data:
        trad = item.get("tradition", "unknown")
        tradition_counts[trad] = tradition_counts.get(trad, 0) + 1
        tradition_docs.setdefault(trad, set()).add(item["text_id"])

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


