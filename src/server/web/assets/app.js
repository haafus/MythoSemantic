import {
    app,
    api,
    buildCorpusApiUrl,
    cleanupRoute,
    corpusTraditionKey,
    ensureCorpusDocuments,
    ensureModels,
    escapeAttribute,
    escapeHtml,
    escapeRegex,
    formatNumber,
    groupDocuments,
    normalizePreviewText,
    normalizeRoute,
    parseHash,
    persistSelectedModel,
    renderModelOptions,
    setActiveNav,
    setBodyClass,
    state,
} from "./core.js";

import { destroyChart, resizeChart, renderScatter, renderHeatmap, renderDistribution } from "./chart.js";

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

function render() {
    cleanupRoute();
    const scatterPlot = document.getElementById("scatter-plot");
    if (scatterPlot) destroyChart(scatterPlot);

    const parsed = parseHash();
    const path = normalizeRoute(parsed.path);
    setBodyClass(path);
    setActiveNav(path);

    if (path !== parsed.path) {
        window.location.hash = `#${path}`;
        return;
    }

    if (path === "/home") return renderHome();
    if (path === "/corpus") return renderCorpus();
    if (path === "/geography") return renderGeography();
    if (path === "/embeddings_analysis") return renderEmbeddingsAnalysis();
    if (path === "/searchSimilarities") return renderSearchSimilarities(parsed.params);
    const graphType = path.slice(1);
    if (["beings", "realms", "ages"].includes(graphType)) return renderGraphPage(graphType);

    window.location.hash = "#/corpus";
}

function renderHome() {
    document.title = "MythoScope - Home";
    app.innerHTML = `
        <main class="home-page">
            <div class="header-container">
                <img src="/assets/logo.jpg" alt="MythoScope Logo" class="logo-image">

                <nav class="nav-menu">
                    <button class="nav-item active" type="button" data-tab="vision">Vision</button>
                    <span class="separator">|</span>
                    <button class="nav-item" type="button" data-tab="methodology">Methodology</button>
                    <span class="separator">|</span>
                    <button class="nav-item" type="button" data-tab="contribute">Contribute</button>
                    <span class="separator">|</span>
                    <button class="nav-item" type="button" data-tab="resources">Resources</button>
                </nav>
            </div>

            <div class="content-container">
                <div id="vision" class="tab-content active">
                    <p>The first large-scale infrastructure for comparative analysis of mythology, religion, and ancient literature &mdash; an international collaborative project integrating classical interpretive methods with artificial intelligence to investigate shared origins and deep structural patterns of human culture.</p>
                    <p>Mythoscope is an interdisciplinary research initiative and open analytical platform dedicated to the large-scale comparative study of mythology, ancient religions, and cultural texts. Integrating classical humanities methodologies with computational approaches, the project enables scholars to explore deep semantic structures, trace cultural patterns across traditions, and investigate the historical evolution of symbolic systems.</p>
                    <p><strong>Toward a Computational Framework for Comparative Mythology.</strong> The framework enables large-scale, cross-cultural, reproducible analysis, combining unsupervised (bottom-up, continuous) and supervised (top-down, discrete) methods to provide a foundation for future work in computational mythology and digital humanities.</p>
                </div>
                <div id="methodology" class="tab-content">insert your text</div>
                <div id="contribute" class="tab-content">insert your text</div>
                <div id="resources" class="tab-content">insert your text</div>
            </div>
        </main>
    `;

    app.querySelectorAll(".nav-item").forEach((button) => {
        button.addEventListener("click", () => {
            app.querySelectorAll(".tab-content").forEach((content) => content.classList.remove("active"));
            app.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
            const target = document.getElementById(button.dataset.tab);
            if (target) target.classList.add("active");
            button.classList.add("active");
        });
    });
}

async function renderCorpus() {
    document.title = "MythoScope - Sources";
    app.innerHTML = `
        <main class="corpus-page container">
            <div class="workspace">
                <aside class="panel library-panel">
                    <div class="panel-header">
                        <div class="panel-title">Literature</div>
                    </div>
                    <div class="library-tree" id="libraryTree">Loading...</div>
                </aside>

                <article class="reader">
                    <div class="reader-header">
                        <div class="reader-title" id="readerTitle">Select a book to begin reading</div>
                    </div>
                    <div class="reader-content" id="readerContent">
                        <div class="reader-placeholder">Choose a title from the literature list.</div>
                    </div>
                </article>

                <aside class="panel info-panel">
                    <div class="panel-header">
                        <div class="panel-title">Book Info</div>
                    </div>
                    <div class="book-info" id="bookInfo">
                        <div class="empty-state">Select a book to view words, sentences, description, and download options.</div>
                        <div class="actions">
                            <a class="btn btn-outline" href="/api/corpus/archive">Download Full Archive</a>
                        </div>
                    </div>
                </aside>
            </div>
        </main>
    `;

    try {
        await ensureCorpusDocuments();
        renderCorpusLibrary();
        renderBookInfo(null);
    } catch (error) {
        const library = document.getElementById("libraryTree");
        const reader = document.getElementById("readerContent");
        if (library) library.innerHTML = `<div class="error-state">${escapeHtml(error.message)}</div>`;
        if (reader) reader.innerHTML = '<div class="reader-placeholder">The literature catalog could not be loaded.</div>';
        renderBookInfo(null);
    }
}

function createCorpusDocumentButton(doc, index) {
    const active = state.selectedCorpusDoc && state.selectedCorpusDoc.id === doc.id
        && state.selectedCorpusDoc.major_tradition === doc.major_tradition
        && state.selectedCorpusDoc.tradition === doc.tradition;
    return `
        <li>
            <button class="document-button${active ? " active" : ""}" type="button" data-doc-index="${index}">
                ${escapeHtml(doc.id)}
            </button>
        </li>
    `;
}

function renderCorpusLibrary() {
    const libraryTree = document.getElementById("libraryTree");
    if (!libraryTree) return;

    const documents = state.corpusDocuments;
    if (!documents.length) {
        libraryTree.innerHTML = '<div class="empty-state">No literature found.</div>';
        return;
    }

    const docIndex = new Map(documents.map((doc, i) => [doc, i]));
    const grouped = groupDocuments(documents);
    if (!state.corpusOpenTraditionsInitialized) {
        grouped.forEach((traditions, major) => {
            traditions.forEach((_, tradition) => {
                state.corpusOpenTraditions.add(corpusTraditionKey(major, tradition));
            });
        });
        state.corpusOpenTraditionsInitialized = true;
    }

    let html = "";

    grouped.forEach((traditions, major) => {
        const isMajorCollapsed = state.corpusCollapsedMajors.has(major);
        html += `<section class="major-section${isMajorCollapsed ? " collapsed" : ""}" data-major="${escapeAttribute(major)}">
            <button class="major-title" type="button">${escapeHtml(major)}</button>
            <div class="major-body">`;

        traditions.forEach((docs, tradition) => {
            const key = corpusTraditionKey(major, tradition);
            const isOpen = state.corpusOpenTraditions.has(key);
            const color = docs[0] && docs[0].color ? docs[0].color : "#6b7280";

            html += `
                <div class="tradition-group${isOpen ? " open" : ""}" data-tradition="${escapeAttribute(tradition)}">
                    <button class="tradition-title" type="button" style="--tradition-color:${escapeAttribute(color)}">
                        <span class="tradition-dot"></span>
                        <span class="tradition-name">${escapeHtml(tradition)}</span>
                        <span class="tradition-toggle">${isOpen ? "-" : "+"}</span>
                    </button>
                    <ul class="document-list">
                        ${docs.map((doc) => createCorpusDocumentButton(doc, docIndex.get(doc))).join("")}
                    </ul>
                </div>
            `;
        });

        html += "</div></section>";
    });

    libraryTree.innerHTML = html;

    libraryTree.querySelectorAll(".major-title").forEach((button) => {
        button.addEventListener("click", () => {
            const section = button.closest(".major-section");
            section.classList.toggle("collapsed");
            const major = section.dataset.major || "Other";
            if (section.classList.contains("collapsed")) {
                state.corpusCollapsedMajors.add(major);
            } else {
                state.corpusCollapsedMajors.delete(major);
            }
        });
    });

    libraryTree.querySelectorAll(".tradition-title").forEach((button) => {
        button.addEventListener("click", () => {
            const group = button.closest(".tradition-group");
            group.classList.toggle("open");
            const section = button.closest(".major-section");
            const key = corpusTraditionKey(section?.dataset.major, group.dataset.tradition);
            if (group.classList.contains("open")) {
                state.corpusOpenTraditions.add(key);
            } else {
                state.corpusOpenTraditions.delete(key);
            }
            const toggle = group.querySelector(".tradition-toggle");
            if (toggle) toggle.textContent = group.classList.contains("open") ? "-" : "+";
        });
    });

    libraryTree.querySelectorAll(".document-button").forEach((button) => {
        button.addEventListener("click", () => {
            const doc = documents[Number(button.dataset.docIndex)];
            if (doc) openCorpusDocument(doc);
        });
    });
}

function renderBookInfo(doc, isLoading = false) {
    const bookInfo = document.getElementById("bookInfo");
    if (!bookInfo) return;

    if (!doc) {
        bookInfo.innerHTML = `
            <div class="empty-state">Select a book to view words, sentences, description, and download options.</div>
            <div class="actions">
                <a class="btn btn-outline" href="/api/corpus/archive">Download Full Archive</a>
            </div>
        `;
        return;
    }

    const url = buildCorpusApiUrl(doc);
    bookInfo.innerHTML = `
        <div class="book-title">${escapeHtml(doc.id)}</div>
        <div class="book-tradition">
            <span class="info-dot" style="--book-color:${escapeAttribute(doc.color || "#6b7280")}"></span>
            <span>${escapeHtml(doc.major_tradition || "Other")} / ${escapeHtml(doc.tradition || "Unknown")}</span>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${formatNumber(doc.word_count)}</div>
                <div class="stat-label">Words</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${formatNumber(doc.sentence_count)}</div>
                <div class="stat-label">Sentences</div>
            </div>
        </div>

        <div class="description-title">Description</div>
        <div class="description-text">${escapeHtml(doc.description || "No description available.")}</div>

        <div class="actions">
            <a class="btn btn-primary${isLoading ? " disabled" : ""}" href="${escapeAttribute(url)}" download="${escapeAttribute(doc.id || "book")}.txt">Download Book</a>
            <a class="btn btn-outline" href="/api/corpus/archive">Download Full Archive</a>
        </div>
    `;
}

async function openCorpusDocument(doc) {
    state.selectedCorpusDoc = doc;
    renderCorpusLibrary();
    renderBookInfo(doc, true);

    const readerTitle = document.getElementById("readerTitle");
    const readerContent = document.getElementById("readerContent");
    if (!readerTitle || !readerContent) return;

    readerTitle.textContent = doc.id;
    readerContent.innerHTML = '<div class="reader-placeholder">Loading book text...</div>';

    try {
        const text = await api(buildCorpusApiUrl(doc));
        readerContent.textContent = text;
        readerContent.scrollTop = 0;
        renderBookInfo(doc, false);
    } catch (error) {
        readerContent.innerHTML = `<div class="error-state">${escapeHtml(error.message)}</div>`;
        renderBookInfo(doc, false);
    }
}

async function renderGeography() {
    document.title = "MythoScope - Geography";
    app.innerHTML = `
        <main class="geography-page container">
            <div class="map-frame">
                <div id="geography-map"></div>
            </div>
        </main>
    `;

    if (typeof L === "undefined") {
        showGeographyError("Map library could not be loaded.");
        return;
    }

    try {
        const traditions = await fetchTraditions();
        initializeGeographyMap(traditions);
    } catch (error) {
        console.error(error);
        showGeographyError("Could not load geography data.");
    }
}

function isValidColor(value) {
    return typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value.trim());
}

function normalizeCoordinates(value) {
    if (!Array.isArray(value) || value.length < 2) return null;

    const lat = Number(value[0]);
    const lon = Number(value[1]);

    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;

    return [lat, lon];
}

function normalizeTraditions(raw) {
    return Object.entries(raw || {})
        .map(([name, info]) => {
            const coordinates = normalizeCoordinates(info && info.coordinates);
            if (!coordinates) return null;

            return {
                name,
                description: info.description || "",
                coordinates,
                color: isValidColor(info.color) ? info.color : "#334155",
                books: Array.isArray(info.books) ? info.books.filter(Boolean) : [],
            };
        })
        .filter(Boolean)
        .sort((a, b) => a.name.localeCompare(b.name));
}

async function loadTraditionInfo() {
    if (state.traditionInfo) return state.traditionInfo;

    try {
        const data = await api("/api/geography/traditions");
        state.traditionInfo = data.traditions || {};
    } catch {
        state.traditionInfo = {};
    }
    return state.traditionInfo;
}

async function fetchTraditions() {
    const raw = await loadTraditionInfo();
    return normalizeTraditions(raw);
}

function buildCoordinateGroups(traditions) {
    const groups = new Map();

    traditions.forEach((item) => {
        const [lat, lon] = item.coordinates;
        const key = `${lat.toFixed(4)},${lon.toFixed(4)}`;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(item);
    });

    return groups;
}

function getOffsetCoordinate(item, index, total) {
    if (total <= 1) return item.coordinates;

    const [lat, lon] = item.coordinates;
    const angle = (Math.PI * 2 * index) / total;
    const radius = 0.8;

    return [
        lat + Math.sin(angle) * radius,
        lon + Math.cos(angle) * radius,
    ];
}

function buildPopupHtml(item) {
    const books = item.books.length
        ? item.books.map((book) => `<li>${escapeHtml(book)}</li>`).join("")
        : "<li>No books listed</li>";

    return `
        <div class="popup-title">
            <span class="popup-color" style="background:${escapeAttribute(item.color)}"></span>
            <span>${escapeHtml(item.name)}</span>
        </div>
        <div class="popup-description">${escapeHtml(item.description)}</div>
        <div class="popup-books-title">Books</div>
        <ul class="popup-books">${books}</ul>
    `;
}

function createMarkerIcon(item) {
    return L.divIcon({
        className: "tradition-marker",
        html: `<button class="map-point" type="button" style="--point-color:${escapeAttribute(item.color)}" aria-label="${escapeAttribute(item.name)}"></button>`,
        iconSize: [18, 18],
        iconAnchor: [9, 9],
        popupAnchor: [0, -11],
    });
}

function renderMarkers(map, traditions) {
    const bounds = [];
    const groups = buildCoordinateGroups(traditions);

    groups.forEach((group) => {
        group.forEach((item, index) => {
            const position = getOffsetCoordinate(item, index, group.length);

            L.marker(position, {
                icon: createMarkerIcon(item),
                keyboard: true,
                title: item.name,
            })
                .addTo(map)
                .bindPopup(buildPopupHtml(item), {
                    className: "tradition-popup",
                    closeButton: true,
                    maxWidth: 340,
                });

            bounds.push(position);
        });
    });

    if (bounds.length > 0) {
        map.fitBounds(bounds, {padding: [34, 34], maxZoom: 4});
    }
}

function initializeGeographyMap(traditions) {
    const worldBounds = [
        [-90, -240],
        [90, 240],
    ];

    state.geographyMap = L.map("geography-map", {
        zoomControl: true,
        maxBounds: worldBounds,
        maxBoundsViscosity: 1.0,
    }).setView([20, 15], 2);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 7,
        minZoom: 2,
        noWrap: true,
        bounds: worldBounds,
        attribution: "&copy; OpenStreetMap contributors",
    }).addTo(state.geographyMap);

    renderMarkers(state.geographyMap, traditions);
}

function showGeographyError(message) {
    const map = document.getElementById("geography-map");
    if (map) map.innerHTML = `<div class="map-error">${escapeHtml(message)}</div>`;
}

async function renderEmbeddingsAnalysis() {
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
        scheduleSearchWarmup(modelSelect.value);
    });
    searchText.addEventListener("focus", () => scheduleSearchWarmup(modelSelect.value));
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

function triggerModelChange() {
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

function scheduleSearchWarmup(model) {
    if (!model || state.warmedSearchModels.has(model)) return;
    state.warmedSearchModels.add(model);
    setTimeout(() => {
        api("/api/similarity/search/warmup", {
            method: "POST",
            body: JSON.stringify({model}),
        }).catch(() => {
            state.warmedSearchModels.delete(model);
        });
    }, 700);
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

function getColorMap(traditions) {
    const colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];
    const map = {};
    traditions.forEach((tradition, index) => {
        map[tradition] = getTraditionColor(tradition, colors[index % colors.length]);
    });
    return map;
}

function getTraditionColor(name, fallback = "#555") {
    const info = state.traditionInfo || {};
    if (info[name] && info[name].color) return info[name].color;

    const cleanName = String(name || "").toLowerCase().replace(/[_\s-]+/g, "").replace(/[^a-z0-9\u0400-\u04FF]/gi, "");
    for (const key in info) {
        const cleanKey = key.toLowerCase().replace(/[_\s-]+/g, "").replace(/[^a-z0-9\u0400-\u04FF]/gi, "");
        if (cleanName === cleanKey && info[key].color) return info[key].color;
    }
    return fallback;
}

function normalizeBookTitle(value) {
    return String(value || "")
        .replace(/\.txt$/i, "")
        .replace(/_/g, " ")
        .trim();
}

function resultBookTitle(item) {
    return normalizeBookTitle(
        item.filename
        || (item.metadata && item.metadata.filename)
        || item.id
        || "Unknown book"
    );
}

function chunkMetaLine(item) {
    return `Book: ${resultBookTitle(item)} | Chunk #${item.chunk_index ?? 0}`;
}

function searchResultMetaLine(item) {
    return `Tradition: ${item.tradition || "Unknown"} | ${chunkMetaLine(item)}`;
}

function chunkTextHtml(item, query = "") {
    const text = item.text || "";
    if (!text) {
        return '<span class="chunk-text-empty">Chunk text is unavailable.</span>';
    }
    return highlightText(text, query);
}

async function displayPointInfo(pointId, chunkIndex = null) {
    if (!state.selectedModel || !pointId) return;

    const infoContent = document.getElementById("infoContent");
    if (!infoContent) return;

    infoContent.innerHTML = '<div style="text-align:center; color:#6c757d">Loading...</div>';
    infoContent.classList.remove("empty");

    try {
        const [point, neighborsData] = await fetchPointAndNeighbors(pointId, chunkIndex);
        const neighbors = Array.isArray(neighborsData.neighbors) ? neighborsData.neighbors : [];
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
                const leaf = createAnalysisTreeNode(doc.id, 2, true);
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

    modalTitle.textContent = doc.id;
    modalBody.textContent = "Loading...";
    modal.style.display = "block";

    try {
        modalBody.textContent = await api(buildCorpusApiUrl(doc));
    } catch {
        modalBody.textContent = "Load error.";
    }
}

function scoreClass(similarityScore) {
    const percent = Math.round(Number(similarityScore || 0) * 100);
    let cls = "score-low";
    if (percent >= 60) cls = "score-high";
    else if (percent >= 40) cls = "score-medium";
    return {percent, cls};
}

function buildPointQueries(chunkIndex) {
    const hasChunk = chunkIndex !== null && chunkIndex !== undefined && chunkIndex !== "";
    const pointQuery = hasChunk ? `?chunk_index=${encodeURIComponent(chunkIndex)}` : "";
    const neighborsQuery = new URLSearchParams({n: "5"});
    if (hasChunk) neighborsQuery.set("chunk_index", String(chunkIndex));
    return {pointQuery, neighborsQuery};
}

function fetchPointAndNeighbors(pointId, chunkIndex) {
    const {pointQuery, neighborsQuery} = buildPointQueries(chunkIndex);
    return Promise.all([
        api(`/api/similarity/points/${encodeURIComponent(state.selectedModel)}/${encodeURIComponent(pointId)}${pointQuery}`),
        api(`/api/similarity/points/${encodeURIComponent(state.selectedModel)}/${encodeURIComponent(pointId)}/neighbors?${neighborsQuery.toString()}`),
    ]);
}

function bindSearchResultClicks(container, handler) {
    container.querySelectorAll(".search-result-item").forEach((item) => {
        item.addEventListener("click", () => handler(item.dataset.pointId, item.dataset.chunkIndex));
    });
}

async function runSemanticSearch({query, model, topK = 20}) {
    return api("/api/similarity/search", {
        method: "POST",
        body: JSON.stringify({query, model, top_k: topK}),
    });
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
            <strong>Found:</strong> ${escapeHtml(data.total)} results
            <span>Model: ${escapeHtml(String(data.model || "").replace(/_/g, "/"))}</span>
        </div>
        <div class="search-result-list">
            ${results.map((result) => renderSearchResultItem(result, data)).join("")}
        </div>
    `);

    bindSearchResultClicks(document.getElementById("searchResults"), displaySearchModalPointInfo);
}

function renderSearchResultItem(result, data) {
    const {percent, cls} = scoreClass(result.similarity_score);
    return `
        <button class="search-result-item" type="button" data-point-id="${escapeAttribute(result.id)}" data-chunk-index="${escapeAttribute(result.chunk_index)}">
            <span class="search-result-topline">
                <span class="result-tradition">${escapeHtml(result.tradition)}</span>
                <span class="result-score ${cls}">${percent}% similarity</span>
            </span>
            <span class="search-result-meta">${escapeHtml(searchResultMetaLine(result))}</span>
            <span class="result-text chunk-text">${chunkTextHtml(result, data.query)}</span>
        </button>
    `;
}

async function displaySearchModalPointInfo(pointId, chunkIndex = null) {
    if (!state.selectedModel || !pointId) return;

    openSearchModal("Chunk Details");
    setSearchResults('<div class="search-loading">Loading nearest chunks...</div>');

    try {
        const [point, neighborsData] = await fetchPointAndNeighbors(pointId, chunkIndex);
        const neighbors = Array.isArray(neighborsData.neighbors) ? neighborsData.neighbors : [];

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

async function renderSearchSimilarities(params) {
    document.title = "Similarity Search - Results";
    app.innerHTML = `
        <main class="search-page container">
            <a href="#/embeddings_analysis" class="back-link">Back</a>

            <div class="search-header">
                <h2>Semantic Similarity Search</h2>
                <div class="search-form">
                    <input type="text" class="search-input" id="searchInput" placeholder="Enter text to search..." />
                    <button class="search-button" id="searchBtn" type="button">Search</button>
                </div>
                <div class="model-selector">
                    <label>Model:</label>
                    <select class="model-select" id="modelSelect"></select>
                </div>
            </div>

            <div class="results-area" id="resultsArea">
                <div class="loading">Enter text to search</div>
            </div>
        </main>
    `;

    const queryParam = params.get("q") || "";
    const modelParam = params.get("model") || "";
    const modelSelect = document.getElementById("modelSelect");
    const searchInput = document.getElementById("searchInput");
    const searchBtn = document.getElementById("searchBtn");

    try {
        await ensureModels();
        modelSelect.innerHTML = renderModelOptions(modelParam || state.selectedModel);
        if (modelParam && state.models.some((model) => model.key === modelParam)) {
            modelSelect.value = modelParam;
        } else {
            modelSelect.value = state.selectedModel;
        }
        persistSelectedModel(modelSelect.value);
    } catch {
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
    }

    modelSelect.addEventListener("change", () => {
        persistSelectedModel(modelSelect.value);
        if (searchInput.value.trim()) performSearchPageSearch();
    });
    searchBtn.addEventListener("click", performSearchPageSearch);
    searchInput.addEventListener("keypress", (event) => {
        if (event.key === "Enter") performSearchPageSearch();
    });
    searchInput.addEventListener("focus", () => scheduleSearchWarmup(modelSelect.value));
    searchInput.addEventListener("input", () => scheduleSearchWarmup(modelSelect.value));

    if (queryParam) {
        searchInput.value = queryParam;
        performSearchPageSearch();
    }
}

async function performSearchPageSearch() {
    const searchInput = document.getElementById("searchInput");
    const modelSelect = document.getElementById("modelSelect");
    const resultsArea = document.getElementById("resultsArea");
    const searchText = searchInput.value.trim();

    if (!searchText) {
        showSearchMessage("Please enter text to search");
        return;
    }

    if (!modelSelect.value) {
        showSearchMessage("Please select a model");
        return;
    }

    const requestId = state.searchPageRequestId + 1;
    state.searchPageRequestId = requestId;
    resultsArea.innerHTML = '<div class="loading">Searching... This may take a few seconds</div>';

    try {
        persistSelectedModel(modelSelect.value);
        const data = await runSemanticSearch({
            query: searchText,
            model: modelSelect.value,
            topK: 20,
        });
        if (requestId !== state.searchPageRequestId) return;
        if (!data) return;
        displaySearchResults(data);
    } catch (error) {
        if (requestId !== state.searchPageRequestId) return;
        resultsArea.innerHTML = `<div class="no-results">
            Search error: ${escapeHtml(error.message)}<br><br>
            <small>Make sure the server is running and model ${escapeHtml(modelSelect.value)} is loaded</small>
        </div>`;
    }
}

function showSearchMessage(message) {
    const resultsArea = document.getElementById("resultsArea");
    resultsArea.innerHTML = `<div class="no-results">${escapeHtml(message)}</div>`;
}

function displaySearchResults(data) {
    const resultsArea = document.getElementById("resultsArea");
    const results = Array.isArray(data.results) ? data.results : [];

    if (!results.length) {
        resultsArea.innerHTML = '<div class="no-results">Nothing found. Try changing the query.</div>';
        return;
    }

    resultsArea.innerHTML = `
        <div style="padding: 15px; background: #faf9f5; border-bottom: 1px solid #e8e6e4;">
            <strong>Found:</strong> ${escapeHtml(data.total)} results
            <span style="float: right; font-size: 0.8rem; color: #8a827c;">
                Model: ${escapeHtml(String(data.model || "").replace(/_/g, "/"))}
            </span>
        </div>
        <div class="search-result-list">
            ${results.map((result) => renderSearchResultItem(result, data)).join("")}
        </div>
    `;

    bindSearchResultClicks(resultsArea, (pointId, chunkIndex) => showPointDetails(pointId, data.model, chunkIndex));
}

function highlightText(text, query) {
    if (!query || !text) return escapeHtml(text);

    const words = query.toLowerCase().split(/\s+/).filter((word) => word.length > 2);
    let escapedText = escapeHtml(text);

    words.forEach((word) => {
        try {
            const regex = new RegExp(`(${escapeRegex(word)})`, "gi");
            escapedText = escapedText.replace(regex, "<mark>$1</mark>");
        } catch {
            // Ignore invalid regex pieces.
        }
    });

    return escapedText;
}

function showPointDetails(pointId, modelName, chunkIndex = null) {
    if (window.opener && !window.opener.closed) {
        window.opener.postMessage({
            type: "openPointDetails",
            id: pointId,
            model: modelName,
            chunkIndex,
        }, window.location.origin);

        const notification = document.createElement("div");
        notification.textContent = "Details opened in the main window";
        notification.style.cssText = "position: fixed; bottom: 20px; right: 20px; background: #2e7d32; color: white; padding: 10px 20px; border-radius: 8px; z-index: 1000; animation: fadeOut 2s forwards;";
        document.body.appendChild(notification);
        setTimeout(() => notification.remove(), 2000);
    } else {
        state.pendingPoint = {id: pointId, model: modelName, chunkIndex};
        if (modelName) persistSelectedModel(modelName);
        window.location.hash = "#/embeddings_analysis";
    }
}

const GRAPH_CATEGORY_COLORS = {
    Character: "#dcd0ff",
    Location: "#c8e6c9",
    Epoch: "#fff3cd",
    Other: "#cccccc",
};

const GRAPH_INFO_FIELDS = {
    beings: ["Name", "Category", "Description", "Roles", "Epithets", "Attributes", "Actions", "Degree", "BetweennessCentrality"],
    realms: ["Name", "Category", "Description", "Function", "Adjacent To", "Degree", "BetweennessCentrality"],
    ages: ["Name", "Category", "Description", "Keyactors", "Keyevents", "Degree", "BetweennessCentrality"],
};

async function renderGraphPage(graphType) {
    document.title = `MythoScope - ${graphType.charAt(0).toUpperCase() + graphType.slice(1)}`;
    app.innerHTML = `
        <main class="graph-page container">
            <div class="graph-sidebar">
                <div class="graph-sidebar-header">Books</div>
                <div class="graph-book-list" id="graphBookList">Loading...</div>
            </div>
            <div class="graph-area">
                <div class="graph-canvas" id="graphCanvas">
                    <div class="graph-placeholder" id="graphPlaceholder">Select a book to view its graph.</div>
                </div>
                <div class="graph-info-panel" id="graphInfoPanel" style="display:none;">
                    <button class="close-btn" id="closeGraphInfo">&times;</button>
                    <div id="graphInfoContent"></div>
                </div>
            </div>
        </main>
    `;

    try {
        const docs = await ensureCorpusDocuments();
        const books = [...new Set(docs.map((d) => d.id))].sort().map((id) => ({ id }));
        renderGraphBookList(books, graphType);
    } catch (error) {
        const list = document.getElementById("graphBookList");
        if (list) list.innerHTML = `<div class="error-state">${escapeHtml(error.message)}</div>`;
    }
}

function renderGraphBookList(books, graphType) {
    const list = document.getElementById("graphBookList");
    if (!list) return;

    if (!books.length) {
        list.innerHTML = '<div class="graph-empty">No graphs available. Run graph generation first.</div>';
        return;
    }

    list.innerHTML = books.map((book) => `
        <button class="graph-book-btn" type="button" data-book-id="${escapeAttribute(book.id)}">
            ${escapeHtml(book.id.replace(/_/g, " "))}
        </button>
    `).join("");

    list.querySelectorAll(".graph-book-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            list.querySelectorAll(".graph-book-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            loadGraphData(btn.dataset.bookId, graphType);
        });
    });
}

async function loadGraphData(bookId, graphType) {
    const placeholder = document.getElementById("graphPlaceholder");
    const canvas = document.getElementById("graphCanvas");
    if (placeholder) {
        placeholder.textContent = "Loading graph...";
        placeholder.style.display = "block";
    }

    if (state.graphCy) {
        state.graphCy.destroy();
        state.graphCy = null;
    }

    try {
        const data = await api(`/api/graphs/${encodeURIComponent(bookId)}/${encodeURIComponent(graphType)}`);
        if (placeholder) placeholder.style.display = "none";
        renderCytoscapeGraph(canvas, data, graphType);
    } catch (error) {
        if (placeholder) {
            placeholder.textContent = `Error: ${error.message}`;
            placeholder.style.display = "block";
        }
    }
}

function renderCytoscapeGraph(container, data, graphType) {
    if (typeof cytoscape === "undefined") {
        const placeholder = document.getElementById("graphPlaceholder");
        if (placeholder) {
            placeholder.textContent = "Cytoscape.js library could not be loaded.";
            placeholder.style.display = "block";
        }
        return;
    }

    const nodes = (data.nodes || []).map((node) => ({data: node}));
    const edges = (data.edges || []).map((edge) => ({data: edge}));

    state.graphCy = cytoscape({
        container,
        elements: [...nodes, ...edges],
        style: [
            {
                selector: "node",
                style: {
                    "background-color": (ele) => GRAPH_CATEGORY_COLORS[ele.data("Category")] || "#aaaaaa",
                    width: (ele) => ele.data("size") || 3,
                    height: (ele) => ele.data("size") || 3,
                    label: (ele) => ele.data("display_name") || ele.data("Name") || ele.data("id"),
                    "font-size": "2.2px",
                    "text-valign": "bottom",
                    "text-halign": "center",
                    "text-margin-y": 1,
                    "border-width": 0.2,
                    "border-color": "#555",
                },
            },
            {
                selector: "edge",
                style: {
                    width: 0.2,
                    "line-color": "#aaa",
                    "target-arrow-color": "#aaa",
                    "target-arrow-shape": "triangle",
                    "target-arrow-scale": 0.1,
                    "arrow-scale": 0.1,
                    "curve-style": "bezier",
                    label: "data(relation)",
                    "font-size": "1.5px",
                    "text-rotation": "autorotate",
                    "text-margin-y": -1,
                    "text-background-opacity": 0.7,
                    "text-background-color": "#ffffff",
                },
            },
            {
                selector: "node.hover-highlight",
                style: {"border-width": 0.3, "border-color": "#ffaa00", "overlay-opacity": 0.3, "overlay-color": "#ffaa00", "overlay-padding": "1px"},
            },
            {
                selector: "edge.hover-highlight",
                style: {width: 0.4, "line-color": "#ffaa00", "target-arrow-color": "#ffaa00"},
            },
            {
                selector: ".faded",
                style: {opacity: 0.1},
            },
        ],
        layout: {name: "cose", idealEdgeLength: 5, padding: 50, spacingFactor: 5, nodeRepulsion: 40000, gravity: 0.0005, numIter: 10000},
        wheelSensitivity: 0.5,
    });

    const cy = state.graphCy;

    let hoveredNode = null;
    cy.on("mouseover", "node", (evt) => {
        if (hoveredNode === evt.target) return;
        hoveredNode = evt.target;
        const highlightSet = hoveredNode.union(hoveredNode.neighborhood().nodes()).union(hoveredNode.connectedEdges());
        cy.elements().not(highlightSet).addClass("faded");
        highlightSet.addClass("hover-highlight");
    });
    cy.on("mouseout", "node", () => {
        hoveredNode = null;
        cy.elements().removeClass("faded hover-highlight");
    });

    const tooltipDiv = document.createElement("div");
    tooltipDiv.className = "graph-edge-tooltip";
    container.appendChild(tooltipDiv);

    let cachedRect = container.getBoundingClientRect();
    cy.on("mouseover", "edge", (evt) => {
        cachedRect = container.getBoundingClientRect();
        const edge = evt.target;
        const sName = cy.getElementById(edge.data("source")).data("display_name") || edge.data("source");
        const tName = cy.getElementById(edge.data("target")).data("display_name") || edge.data("target");
        tooltipDiv.innerHTML = `${escapeHtml(sName)} &rarr; ${escapeHtml(tName)}<br><strong>${escapeHtml(edge.data("relation") || "")}</strong>`;
        tooltipDiv.style.left = (evt.originalEvent.clientX - cachedRect.left + 10) + "px";
        tooltipDiv.style.top = (evt.originalEvent.clientY - cachedRect.top + 10) + "px";
        tooltipDiv.style.display = "block";
    });
    cy.on("mousemove", "edge", (evt) => {
        tooltipDiv.style.left = (evt.originalEvent.clientX - cachedRect.left + 10) + "px";
        tooltipDiv.style.top = (evt.originalEvent.clientY - cachedRect.top + 10) + "px";
    });
    cy.on("mouseout", "edge", () => {
        tooltipDiv.style.display = "none";
    });

    const infoPanel = document.getElementById("graphInfoPanel");
    const infoContent = document.getElementById("graphInfoContent");
    document.getElementById("closeGraphInfo").addEventListener("click", () => {
        infoPanel.style.display = "none";
    });

    cy.on("tap", "node", (evt) => {
        const d = evt.target.data();
        const fields = GRAPH_INFO_FIELDS[graphType] || GRAPH_INFO_FIELDS.beings;
        let html = `<h4>${escapeHtml(d.display_name || d.Name || d.id)}</h4><table>`;
        fields.forEach((f) => {
            if (d[f] !== undefined && d[f] !== null && d[f] !== "") {
                const value = typeof d[f] === "object" ? JSON.stringify(d[f]) : d[f];
                html += `<tr><th>${escapeHtml(f)}</th><td>${escapeHtml(value)}</td></tr>`;
            }
        });
        html += "</table>";
        infoContent.innerHTML = html;
        infoPanel.style.display = "block";
    });
}

window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.type !== "openPointDetails") return;

    state.pendingPoint = {
        id: data.id,
        model: data.model,
        chunkIndex: data.chunkIndex,
    };

    const currentPath = normalizeRoute(parseHash().path);
    if (currentPath !== "/embeddings_analysis") {
        window.location.hash = "#/embeddings_analysis";
    } else {
        if (data.model) {
            const modelSelect = document.getElementById("global-model-select");
            if (modelSelect && Array.from(modelSelect.options).some((option) => option.value === data.model)) {
                modelSelect.value = data.model;
                triggerModelChange();
            }
        }
        displayPointInfo(data.id, data.chunkIndex);
        state.pendingPoint = null;
    }
});

window.addEventListener("hashchange", render);
window.addEventListener("DOMContentLoaded", render);
