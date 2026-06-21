PROJECTION_METHODS = [
    {"key": "umap",                      "label": "UMAP 2D",                   "chart_type": "scatter"},
    {"key": "residual_umap",             "label": "Residual UMAP",             "chart_type": "scatter"},
    {"key": "residual_normalized_umap",  "label": "Residual Normalized UMAP",  "chart_type": "scatter"},
    {"key": "rlace_umap",                "label": "RLACE UMAP",                "chart_type": "scatter"},
    {"key": "motif_umap",                "label": "Motif UMAP",                "chart_type": "scatter"},
    {"key": "distance_heatmap",          "label": "Distance Heatmap",          "chart_type": "heatmap"},
    {"key": "tradition_distribution",    "label": "Tradition Distribution",    "chart_type": "distribution"},
]

PROJECTION_KEYS = {m["key"] for m in PROJECTION_METHODS}
