export const app = document.getElementById("app");

export const state = {
    models: [],
    selectedModel: localStorage.getItem("selectedModel") || localStorage.getItem("mythoscope.model") || "",
    corpusDocuments: [],
    corpusDocumentsBySource: {},
    selectedCorpusDoc: null,
    corpusOpenTraditions: new Set(),
    corpusOpenTraditionsInitialized: false,
    corpusCollapsedMajors: new Set(),
    traditionInfo: null,
    geographyMap: null,
    pendingPoint: null,
    lastAnalysisSearchData: null,
    analysisSearchRequestId: 0,
    searchPageRequestId: 0,
    warmedSearchModels: new Set(),
    keydownHandler: null,
};

export const SIMILARITY_METHODS = [
    ["umap", "UMAP 2D"],
    ["pca", "PCA 2D"],
    ["tsne", "t-SNE 2D"],
    ["distance_heatmap", "Distance Heatmap"],
    ["tradition_distribution", "Tradition Distribution"],
    ["methods_comparison", "Method Comparison"],
    ["umap_hyperparameters_dashboard", "UMAP Parameter Dashboard"],
    ["tsne_hyperparameters_dashboard", "t-SNE Parameter Dashboard"],
];

export const HTML_ONLY_METHODS = new Set([
    "distance_heatmap",
    "tradition_distribution",
    "methods_comparison",
    "umap_hyperparameters_dashboard",
    "tsne_hyperparameters_dashboard",
]);

export const CLUSTERING_ALGORITHMS = [
    ["birch", "BIRCH"],
    ["gmm", "Gaussian Mixture"],
    ["hdbscan", "HDBSCAN"],
    ["kmeans", "K-Means"],
    ["meanshift", "Meanshift"],
    ["optics", "Optics"],
    ["spectral", "Spectral"],
];

export const METRIC_NAMES = {
    silhouette_score: "Silhouette",
    adjusted_rand_score: "ARI",
    normalized_mutual_info: "NMI",
    v_measure: "V-measure",
    n_clusters_found: "Clusters",
    noise_ratio: "Noise Ratio",
};

export function parseHash() {
    const raw = (window.location.hash || "#/").slice(1) || "/";
    const splitAt = raw.indexOf("?");
    const path = splitAt === -1 ? raw : raw.slice(0, splitAt);
    const query = splitAt === -1 ? "" : raw.slice(splitAt + 1);
    return {
        path: path || "/",
        params: new URLSearchParams(query),
    };
}

export function normalizeRoute(path) {
    if (path === "/") return "/corpus";
    if (path === "/sources") return "/corpus";
    if (path === "/similarity") return "/embeddings_analysis";
    if (path === "/clusterisation") return "/cluster_analysis";
    return path;
}

function routeClass(path) {
    if (path === "/home") return "route-home";
    if (path === "/") return "route-corpus";
    if (path === "/corpus") return "route-corpus";
    if (path === "/geography") return "route-geography";
    if (path === "/embeddings_analysis") return "route-embeddings";
    if (path === "/cluster_analysis") return "route-cluster";
    if (path === "/searchSimilarities") return "route-search";
    return "route-corpus";
}

export function setBodyClass(path) {
    document.body.className = `has-main-navbar ${routeClass(path)}`;
}

export function setActiveNav(path) {
    const activePath = path === "/searchSimilarities" ? "/embeddings_analysis" : path;
    document.querySelectorAll(".nav-links a").forEach((link) => {
        const href = link.getAttribute("href") || "";
        const hashPath = normalizeRoute((href.split("#")[1] || "/").split("?")[0] || "/");
        link.classList.toggle("active", hashPath === activePath);
    });
}

export function cleanupRoute() {
    if (state.keydownHandler) {
        document.removeEventListener("keydown", state.keydownHandler);
        state.keydownHandler = null;
    }
    if (state.geographyMap) {
        state.geographyMap.remove();
        state.geographyMap = null;
    }
    if (window.Plotly) {
        document.querySelectorAll("#scatter-plot, .plotly-managed").forEach((plot) => {
            try {
                Plotly.purge(plot);
            } catch {
                // Plotly may already have been torn down by a route change.
            }
        });
    }
}

export async function api(path, options = {}) {
    const response = await fetch(path, {
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {}),
        },
        ...options,
    });

    if (!response.ok) {
        const text = await response.text();
        let detail = text || response.statusText;
        try {
            const payload = JSON.parse(text);
            detail = payload.detail || payload.error || payload.message || detail;
        } catch {
            // Plain text response.
        }
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) return response.json();
    return response.text();
}

export function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = String(value ?? "");
    return div.innerHTML;
}

export function normalizePreviewText(value) {
    return String(value ?? "")
        .replace(/-\s*&lt;br\s*\/?&gt;/gi, "-")
        .replace(/-\s*<br\s*\/?>/gi, "-")
        .replace(/&lt;br\s*\/?&gt;/gi, " ")
        .replace(/<br\s*\/?>/gi, " ")
        .replace(/\s+/g, " ")
        .trim();
}

export function escapeAttribute(value) {
    return escapeHtml(value).replace(/"/g, "&quot;");
}

export function escapeRegex(value) {
    return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function formatNumber(value) {
    const number = Number(value || 0);
    return Number.isFinite(number) ? number.toLocaleString("en-US") : "0";
}

function modelLabel(model) {
    return (model.name || model.key || "").replace(/_/g, "/");
}

export function persistSelectedModel(key) {
    state.selectedModel = key || "";
    localStorage.setItem("selectedModel", state.selectedModel);
    localStorage.setItem("mythoscope.model", state.selectedModel);
}

export async function ensureModels() {
    if (!state.models.length) {
        const data = await api("/api/models");
        state.models = Array.isArray(data.models) ? data.models : [];
    }

    const keys = state.models.map((model) => model.key);
    if (!state.selectedModel || !keys.includes(state.selectedModel)) {
        state.selectedModel = keys[0] || "";
    }

    return state.models;
}

export function renderModelOptions(selectedKey = state.selectedModel) {
    if (!state.models.length) return '<option value="">No available models</option>';
    return state.models.map((model) => `
        <option value="${escapeAttribute(model.key)}" ${model.key === selectedKey ? "selected" : ""}>
            ${escapeHtml(modelLabel(model))}
        </option>
    `).join("");
}

export async function ensureCorpusDocuments(source = "corpus") {
    if (!state.corpusDocumentsBySource[source]) {
        const data = await api(`/api/corpus/catalog?source=${encodeURIComponent(source)}`);
        state.corpusDocumentsBySource[source] = Array.isArray(data.documents) ? data.documents : [];
        if (source === "corpus") {
            state.corpusDocuments = state.corpusDocumentsBySource[source];
        }
    }
    return state.corpusDocumentsBySource[source];
}

export function buildCorpusApiUrl(doc) {
    const params = new URLSearchParams({
        id: doc.id || "",
        major_tradition: doc.major_tradition || "",
        tradition: doc.tradition || "",
        source: doc.source || "corpus",
    });
    return `/api/corpus/documents?${params.toString()}`;
}

export function corpusTraditionKey(major, tradition) {
    return `${major || "Other"}\u0000${tradition || "Unknown"}`;
}

export function groupDocuments(items) {
    const grouped = new Map();

    items.forEach((doc) => {
        const major = doc.major_tradition || "Other";
        const tradition = doc.tradition || "Unknown";

        if (!grouped.has(major)) grouped.set(major, new Map());
        const majorGroup = grouped.get(major);
        if (!majorGroup.has(tradition)) majorGroup.set(tradition, []);
        majorGroup.get(tradition).push(doc);
    });

    return grouped;
}
