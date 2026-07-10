# MythoSemantic

**An Interactive System for Discovering Semantic Parallels Across Mythological and Religious Traditions**

MythoSemantic is an open, interactive tool for the computational comparative analysis of mythological and religious texts. It transforms a multi-tradition corpus into a shared semantic space using neural embedding models, then exposes that space through a web interface — letting researchers browse, cluster, and semantically search across traditions without writing any code.

Comparative mythology has traditionally relied on manual cross-referencing of motifs across cultures — slow, expertise-heavy, and hard to scale. MythoSemantic doesn't replace that expertise; it gives it a faster way to find candidate parallels worth investigating.

## What it does

- **Recovers known cross-tradition connections** without supervision — e.g. Greek↔Roman, Christianity↔Islam↔Babylonian, Buddhism↔Taoism↔Japanese Buddhism cluster together in the learned embedding space, matching well-documented comparative-mythology relationships.
- **Surfaces candidate hypotheses for expert follow-up** — for example, an unexpected proximity between Australian Aboriginal and West African material that isn't covered by existing detailed comparative studies.
- **Compares five embedding backbones side by side** (`BAAI/bge-m3`, `Qwen/Qwen3-Embedding`, `intfloat/e5-large-v2`, `sentence-transformers/LaBSE`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) across seven clustering algorithms, with full quantitative evaluation (ARI, NMI, V-measure, silhouette) against tradition labels — notably showing that embedding dimensionality alone does not predict clustering quality.

## Corpus

28 mythological and religious texts spanning 22 traditions across 12 cultural areas (English translations, mainly via Project Gutenberg), segmented into ~25,000 overlapping fragments. See the paper's Data section for the full tradition-by-text breakdown and chunking methodology.

## Interface

- **Sources** — tradition-grouped corpus browser with a reading pane and per-document export.
- **Similarity** — 2D projection (PCA / t-SNE / UMAP) of any embedding model; click a point or type free text to retrieve the nearest matching fragments across traditions, ranked by cosine similarity.
- **Geography** — traditions plotted on an interactive map, linked to their source texts.
- **Clusterisation** — pick a model and clustering algorithm to get live quality metrics, a colored UMAP projection, and a tradition-cluster correspondence matrix.

*(Additional experimental views — Ages, Realms, Beings — are present but not yet stable; see the paper's Future Work.)*

## Quick start

See [RUN.md](./RUN.md) for setup and running instructions.


## License

Released under the [MIT License](./LICENSE)
