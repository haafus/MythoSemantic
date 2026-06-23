import {
    api, app, state,
    ensureCorpusDocuments, ensureModels,
    escapeAttribute, escapeHtml,
    groupDocuments, loadTraditionInfo,
    persistSelectedModel, renderModelOptions,
} from "./core.js";
import { destroyChart, resizeChart, renderScatter, renderHeatmap, renderDistribution } from "./chart.js";
import {
    bindSearchResultClicks, chunkMetaLine, chunkTextHtml,
    fetchPointWithNeighbors, renderSearchResultItem,
    resultBookTitle, runSemanticSearch,
    searchResultMetaLine,
} from "./search-utils.js";

function getTraditionColor(name, fallback = "#555") {
    const info = state.traditionInfo || {};
    if (info[name] && info[name].color) return info[name].color;

    const cleanName = String(name || "").toLowerCase().replace(/[_\s-]+/g, "").replace(/[^a-z0-9Ѐ-ӿ]/gi, "");
    for (const key in info) {
        const cleanKey = key.toLowerCase().replace(/[_\s-]+/g, "").replace(/[^a-z0-9Ѐ-ӿ]/gi, "");
        if (cleanName === cleanKey && info[key].color) return info[key].color;
    }
    return fallback;
}

function getColorMap(traditions) {
    const colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];
    const map = {};
    traditions.forEach((tradition, index) => {
        map[tradition] = getTraditionColor(tradition, colors[index % colors.length]);
    });
    return map;
}

const CHART_RENDERERS = {
    scatter: async (el, data) => {
        const traditions = [...new Set((data.points || []).map((p) => p.tradition || "Unknown"))];
        await renderScatter(el, data, { colorMap: getColorMap(traditions), onPointClick: displayPointInfo });
    },
    heatmap: (el, data) => renderHeatmap(el, data),
    distribution: async (el, data) => {
        const names = (data.traditions || []).map((t) => t.name);
        await renderDistribution(el, data, { colorMap: getColorMap(names) });
    },
};

export async function renderEmbeddingsAnalysis() {
    document.title = "Embedding Analysis";
    app.innerHTML = `
        <main class="analysis-page container">
            <div class="controls-panel">
                <div class="form-group">
                    <label>Model:</label>
                    <select id="global-model-select"><option value="">Loading models...</option></select>
                </div>
                <span id="model-status" class="status-badge">Waiting for selection...</span>
            </div>

            <div class="main-content">
                <div class="tree-sidebar">
                    <div class="tree-header">
                        <div class="tree-title">Corpus Chunks</div>
                    </div>
                    <div id="tree-container" class="tree-container">Loading...</div>
                </div>

                <div class="plot-area">
                    <div class="card plot-container" id="plotContainer">
                        <div class="card-header">
                            <div class="form-group">
                                <label>Method:</label>
                                <select id="viz-select"><option value="">Loading...</option></select>
                            </div>
                            <div style="display: flex; gap: 8px;">
                                <button class="btn btn-outline enter-fullscreen" type="button" id="enter-fullscreen">Enter Fullscreen</button>
                                <button class="btn btn-outline exit-fullscreen" type="button" id="exit-fullscreen">Exit Fullscreen</button>
                            </div>
                        </div>
                        <div class="plot-canvas" id="plotCanvas">
                            <div class="loading-placeholder" id="loadingPlaceholder">Loading visualization...</div>
                            <div id="scatter-plot" style="width: 100%; height: 100%; display: none;"></div>
                        </div>
                    </div>
                </div>

                <div class="sidebar">
                    <div class="card">
                        <div class="card-header"><h3 class="card-title">Point Information</h3></div>
                        <div class="card-body info-content empty" id="infoContent">
                            Click any point in the chart to see information
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header"><h3 class="card-title">Semantic Search</h3></div>
                        <div class="card-body">
                            <textarea id="search-text" placeholder="Enter text to find similar fragments..."></textarea>
                            <button class="btn btn-primary search-btn" id="search-btn" type="button" disabled>Find Matches</button>
                        </div>
                    </div>
                </div>
            </div>

            <div id="readerModal">
                <div class="modal-content-reader">
                    <div class="card-header">
                        <strong id="modalTitle">Book</strong>
                        <button class="btn" id="close-reader-modal" type="button">Close</button>
                    </div>
                    <div class="modal-body-reader" id="modalBody"></div>
                </div>
            </div>

            <div id="searchModal">
                <div class="modal-content-reader search-modal-content">
                    <div class="card-header">
                        <strong id="searchModalTitle">Search Results</strong>
                        <button class="btn" id="close-search-modal" type="button">Close</button>
                    </div>
                    <div class="search-results" id="searchResults"></div>
                </div>
            </div>
        </main>
    `;

    bindEmbeddingsControls();

    try {
        await loadSimilarityMethods();
        await loadModelsIntoSelect();
        await initializeAnalysisLibrary();
        if (state.pendingPoint) {
            const pending = state.pendingPoint;
            state.pendingPoint = null;
            if (pending.model) {
                const select = document.getElementById("global-model-select");
                if (select && Array.from(select.options).some((option) => option.value === pending.model)) {
                    select.value = pending.model;
                    triggerModelChange();
                }
            }
            displayPointInfo(pending.id, pending.chunkIndex);
        }
    } catch (error) {
        updateStatus(error.message, "error");
    }
}

function bindEmbeddingsControls() {
    const modelSelect = document.getElementById("global-model-select");
    const vizSelect = document.getElementById("viz-select");
    const searchText = document.getElementById("search-text");
    const searchBtn = document.getElementById("search-btn");
    const plotContainer = document.getElementById("plotContainer");

    modelSelect.addEventListener("change", triggerModelChange);
    vizSelect.addEventListener("change", loadVisualization);
    searchText.addEventListener("input", () => {
        searchBtn.disabled = searchText.value.trim().length === 0 || !state.selectedModel;
    });
    searchBtn.addEventListener("click", performAnalysisSearch);

    document.getElementById("enter-fullscreen").addEventListener("click", toggleFullscreen);
    document.getElementById("exit-fullscreen").addEventListener("click", toggleFullscreen);
    document.getElementById("close-reader-modal").addEventListener("click", () => {
        document.getElementById("readerModal").style.display = "none";
    });
    document.getElementById("close-search-modal").addEventListener("click", closeSearchModal);

    state.keydownHandler = (event) => {
        if (event.key === "Escape" && plotContainer && plotContainer.classList.contains("fullscreen")) {
            toggleFullscreen();
        }
    };
    document.addEventListener("keydown", state.keydownHandler);
}

async function loadSimilarityMethods() {
    if (!state.similarityMethods.length) {
        state.similarityMethods = await api("/api/similarity/methods");
    }
    const vizSelect = document.getElementById("viz-select");
    if (vizSelect) {
        vizSelect.innerHTML = state.similarityMethods
            .map((m) => `<option value="${escapeAttribute(m.key)}">${escapeHtml(m.label)}</option>`)
            .join("");
    }
}

async function loadModelsIntoSelect() {
    updateStatus("Loading list...", "loading");
    await ensureModels();

    const modelSelect = document.getElementById("global-model-select");
    if (!modelSelect) return;

    modelSelect.innerHTML = renderModelOptions();
    if (!state.models.length) {
        updateStatus("Error: no models", "error");
        return;
    }

    modelSelect.value = state.selectedModel;
    triggerModelChange();
}

export function triggerModelChange() {
    const modelSelect = document.getElementById("global-model-select");
    const searchText = document.getElementById("search-text");
    const searchBtn = document.getElementById("search-btn");
    if (!modelSelect || !modelSelect.value) return;

    persistSelectedModel(modelSelect.value);
    state.analysisSearchRequestId += 1;
    if (searchBtn && searchText) {
        searchBtn.disabled = searchText.value.trim().length === 0 || !state.selectedModel;
    }
    loadVisualization();
}

function updateStatus(text, type = "loaded") {
    const status = document.getElementById("model-status");
    if (!status) return;
    status.textContent = text;
    status.className = `status-badge ${type}`;
}

function toggleFullscreen() {
    const plotContainer = document.getElementById("plotContainer");
    if (!plotContainer) return;

    plotContainer.classList.toggle("fullscreen");
    resizeChart(document.getElementById("scatter-plot"));
}

async function loadVisualization() {
    if (!state.selectedModel) return;

    const method = document.getElementById("viz-select").value;
    const scatterPlot = document.getElementById("scatter-plot");
    const loadingPlaceholder = document.getElementById("loadingPlaceholder");

    updateStatus("Loading chart data...", "loading");
    loadingPlaceholder.style.display = "block";
    loadingPlaceholder.textContent = "Loading visualization...";
    document.querySelector(".plot-hover-tooltip")?.classList.remove("visible");
    scatterPlot.style.display = "none";
    scatterPlot.style.minWidth = "";
    scatterPlot.style.minHeight = "";
    destroyChart(scatterPlot);

    try {
        const data = await api(`/api/similarity/projections/${encodeURIComponent(state.selectedModel)}/${encodeURIComponent(method)}`);

        await loadTraditionInfo();
        const chartType = (state.similarityMethods.find((m) => m.key === method) || {}).chart_type || "scatter";
        await CHART_RENDERERS[chartType](scatterPlot, data);

        loadingPlaceholder.style.display = "none";
        updateStatus("Ready", "loaded");
    } catch (error) {
        loadingPlaceholder.innerHTML = `Error: ${escapeHtml(error.message)}`;
        loadingPlaceholder.style.display = "block";
        updateStatus("Load error", "error");
    }
}

export async function displayPointInfo(pointId, chunkIndex = null) {
    if (!state.selectedModel || !pointId) return;

    const infoContent = document.getElementById("infoContent");
    if (!infoContent) return;

    infoContent.innerHTML = '<div style="text-align:center; color:#6c757d">Loading...</div>';
    infoContent.classList.remove("empty");

    try {
        const results = await fetchPointWithNeighbors(pointId, chunkIndex);
        const point = results[0];
        const neighbors = results.slice(1);
        if (!point) throw new Error("Point not found");
        let html = `
            <div class="badge">${escapeHtml(point.tradition)}</div>
            <div class="search-result-meta">${escapeHtml(chunkMetaLine(point))}</div>
            <div class="text-preview"><strong>ID:</strong> ${escapeHtml(point.id)}<br><br>${escapeHtml(point.text)}</div>
            <h4 style="margin: 16px 0 8px; font-size:14px; color:#111;">Nearest neighbors:</h4>
        `;

        if (neighbors.length > 0) {
            html += neighbors.map((neighbor) => `
                <div class="neighbor-item" data-neighbor-id="${escapeAttribute(neighbor.id)}" data-neighbor-chunk="${escapeAttribute(neighbor.chunk_index)}">
                    <span class="badge" style="background:#dee2e6; color:#212529">${escapeHtml(neighbor.tradition)}</span>
                    <div class="neighbor-meta">${escapeHtml(chunkMetaLine(neighbor))}</div>
                    <div class="neighbor-text">${escapeHtml(neighbor.text || "")}</div>
                    <div class="neighbor-stats">Similarity: ${(Number(neighbor.similarity_score || 0) * 100).toFixed(1)}%</div>
                </div>
            `).join("");
        }

        infoContent.innerHTML = html;
        infoContent.querySelectorAll(".neighbor-item").forEach((item) => {
            item.addEventListener("click", () => displayPointInfo(item.dataset.neighborId, item.dataset.neighborChunk));
        });
    } catch (error) {
        infoContent.innerHTML = `<div style="color:#d32f2f">Error: ${escapeHtml(error.message)}</div>`;
    }
}

async function initializeAnalysisLibrary() {
    await loadTraditionInfo();

    const container = document.getElementById("tree-container");
    if (!container) return;

    try {
        const documents = await ensureCorpusDocuments();
        container.innerHTML = "";
        renderAnalysisTreeFromDocuments(documents, container);
    } catch (error) {
        container.innerHTML = `<div style="color:#d32f2f; padding:10px;">${escapeHtml(error.message)}</div>`;
    }
}

function renderAnalysisTreeFromDocuments(documents, container) {
    if (!documents.length) {
        container.innerHTML = '<div style="padding:10px;">No corpus files found.</div>';
        return;
    }

    const grouped = groupDocuments(documents);
    grouped.forEach((traditions, major) => {
        const majorWrapper = createAnalysisTreeNode(major, 0, false);
        const majorChildren = document.createElement("div");
        majorChildren.style.display = "block";
        attachTreeToggle(majorWrapper.item, majorChildren);

        traditions.forEach((docs, tradition) => {
            const traditionWrapper = createAnalysisTreeNode(tradition, 1, false, getTraditionColor(tradition, docs[0]?.color || "#555"));
            const traditionChildren = document.createElement("div");
            traditionChildren.style.display = "block";
            attachTreeToggle(traditionWrapper.item, traditionChildren);

            docs.forEach((doc) => {
                const leaf = createAnalysisTreeNode(doc.title, 2, true);
                leaf.item.addEventListener("click", () => openBookReader(doc));
                traditionChildren.appendChild(leaf.wrapper);
            });

            traditionWrapper.wrapper.appendChild(traditionChildren);
            majorChildren.appendChild(traditionWrapper.wrapper);
        });

        majorWrapper.wrapper.appendChild(majorChildren);
        container.appendChild(majorWrapper.wrapper);
    });
}

function createAnalysisTreeNode(name, depth, isLeaf, color) {
    const wrapper = document.createElement("div");
    wrapper.className = `tree-node level-${depth}`;

    const item = document.createElement("div");
    item.className = `tree-item level-${depth} ${isLeaf ? "leaf" : ""}`;

    if (depth === 1) {
        const circle = document.createElement("span");
        circle.className = "color-circle";
        circle.style.backgroundColor = color || "#555";
        item.appendChild(circle);
    }

    const textSpan = document.createElement("span");
    textSpan.textContent = String(name || "").replace(/\.txt$/, "");
    item.appendChild(textSpan);

    if (!isLeaf) {
        const toggle = document.createElement("span");
        toggle.className = "folder-toggle";
        toggle.textContent = "-";
        item.appendChild(toggle);
    }

    wrapper.appendChild(item);
    return {wrapper, item};
}

function attachTreeToggle(item, childContainer) {
    item.addEventListener("click", (event) => {
        event.stopPropagation();
        const isHidden = childContainer.style.display === "none";
        childContainer.style.display = isHidden ? "block" : "none";
        const toggle = item.querySelector(".folder-toggle");
        if (toggle) toggle.textContent = isHidden ? "-" : "+";
    });
}

async function openBookReader(doc) {
    const modal = document.getElementById("readerModal");
    const modalTitle = document.getElementById("modalTitle");
    const modalBody = document.getElementById("modalBody");
    if (!modal || !modalTitle || !modalBody) return;

    modalTitle.textContent = doc.title;
    modalBody.textContent = "Loading...";
    modal.style.display = "block";

    try {
        modalBody.textContent = await api(buildCorpusApiUrl(doc));
    } catch {
        modalBody.textContent = "Load error.";
    }
}

function buildCorpusApiUrl(doc) {
    const params = new URLSearchParams({
        title: doc.title || "",
        major_tradition: doc.major_tradition || "",
        tradition: doc.tradition || "",
        source: doc.source || "corpus",
    });
    return `/api/corpus/documents?${params.toString()}`;
}

async function performAnalysisSearch() {
    const searchText = document.getElementById("search-text");
    const searchBtn = document.getElementById("search-btn");
    const text = searchText.value.trim();
    if (!text || !state.selectedModel) return;

    const requestId = state.analysisSearchRequestId + 1;
    state.analysisSearchRequestId = requestId;
    searchBtn.disabled = true;
    searchBtn.textContent = "Searching...";
    openSearchModal("Search Results");
    setSearchResults('<div class="search-loading">Searching... This may take a few seconds.</div>');

    try {
        const data = await runSemanticSearch({
            query: text,
            model: state.selectedModel,
            topK: 20,
        });
        if (requestId !== state.analysisSearchRequestId) return;
        if (!data) return;
        displayAnalysisSearchResults(data);
    } catch (error) {
        if (requestId !== state.analysisSearchRequestId) return;
        setSearchResults(`
            <div class="search-empty">
                Search error: ${escapeHtml(error.message)}
                <small>Check that model ${escapeHtml(state.selectedModel)} is available.</small>
            </div>
        `);
    } finally {
        if (requestId === state.analysisSearchRequestId) {
            searchBtn.disabled = searchText.value.trim().length === 0 || !state.selectedModel;
            searchBtn.textContent = "Find Matches";
        }
    }
}

function openSearchModal(title = "Search Results") {
    const modal = document.getElementById("searchModal");
    const modalTitle = document.getElementById("searchModalTitle");
    if (!modal || !modalTitle) return;

    modalTitle.textContent = title;
    modal.style.display = "block";
}

function closeSearchModal() {
    const modal = document.getElementById("searchModal");
    if (modal) modal.style.display = "none";
}

function setSearchResults(html) {
    const results = document.getElementById("searchResults");
    if (results) results.innerHTML = html;
}

function displayAnalysisSearchResults(data) {
    const results = Array.isArray(data.results) ? data.results : [];
    state.lastAnalysisSearchData = data;
    if (!results.length) {
        setSearchResults('<div class="search-empty">Nothing found. Try changing the query.</div>');
        return;
    }

    setSearchResults(`
        <div class="search-summary">
            <strong>Found:</strong> ${results.length} results
            <span>Model: ${escapeHtml(String(data.model || "").replace(/_/g, "/"))}</span>
        </div>
        <div class="search-result-list">
            ${results.map((result) => renderSearchResultItem(result, data)).join("")}
        </div>
    `);

    bindSearchResultClicks(document.getElementById("searchResults"), displaySearchModalPointInfo);
}

async function displaySearchModalPointInfo(pointId, chunkIndex = null) {
    if (!state.selectedModel || !pointId) return;

    openSearchModal("Chunk Details");
    setSearchResults('<div class="search-loading">Loading nearest chunks...</div>');

    try {
        const results = await fetchPointWithNeighbors(pointId, chunkIndex);
        const point = results[0];
        const neighbors = results.slice(1);
        if (!point) throw new Error("Point not found");

        setSearchResults(`
            <div class="search-detail">
                <button class="btn btn-outline" type="button" id="backToSearchResults">Back to results</button>
                <div class="search-result-topline">
                    <span class="result-tradition">${escapeHtml(point.tradition)}</span>
                    <span class="search-result-meta">${escapeHtml(searchResultMetaLine(point))}</span>
                </div>
                <div class="search-detail-text">${escapeHtml(point.text)}</div>
            </div>
            <div class="search-summary">
                <strong>Nearest neighbors</strong>
                <span>${escapeHtml(resultBookTitle(point))}</span>
            </div>
            <div class="search-result-list">
                ${neighbors.length ? neighbors.map((neighbor) => renderSearchResultItem(neighbor, {query: ""})).join("") : '<div class="search-empty">No nearest chunks found.</div>'}
            </div>
        `);

        const back = document.getElementById("backToSearchResults");
        if (back) back.addEventListener("click", () => displayAnalysisSearchResults(state.lastAnalysisSearchData || {results: []}));
        bindSearchResultClicks(document.getElementById("searchResults"), displaySearchModalPointInfo);
    } catch (error) {
        setSearchResults(`<div class="search-empty">Load error: ${escapeHtml(error.message)}</div>`);
    }
}
