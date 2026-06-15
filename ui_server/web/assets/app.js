import {
    app,
    api,
    buildCorpusApiUrl,
    cleanupRoute,
    CLUSTERING_ALGORITHMS,
    corpusTraditionKey,
    ensureCorpusDocuments,
    ensureModels,
    escapeAttribute,
    escapeHtml,
    escapeRegex,
    formatNumber,
    groupDocuments,
    HTML_ONLY_METHODS,
    METRIC_NAMES,
    normalizePreviewText,
    normalizeRoute,
    parseHash,
    persistSelectedModel,
    renderModelOptions,
    setActiveNav,
    setBodyClass,
    SIMILARITY_METHODS,
    state,
} from "./core.js";
import {
    renderSavedPlotInto,
    resizeEmbeddedPlots,
} from "./plot-utils.js";

const NEAREST_NEIGHBOR_COUNT = 5;
const CROSS_TRADITION_NEIGHBOR_COUNT = 10;

function render() {
    cleanupRoute();

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
    if (path === "/cluster_analysis") return renderClusterAnalysis();
    if (path === "/searchSimilarities") return renderSearchSimilarities(parsed.params);
    if (["/ages", "/realms", "/beings"].includes(path)) return renderFutureDomain(path.slice(1));

    window.location.hash = "#/corpus";
}

function renderHome() {
    document.title = "MythoScope - Home";
    app.innerHTML = `
        <main class="home-page">
            <div class="header-container">
                <img src="/template/Logo.jpg" alt="MythoScope Logo" class="logo-image">

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
                        ${docs.map((doc) => createCorpusDocumentButton(doc, documents.indexOf(doc))).join("")}
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

    const urls = [
        "/corpus_chunked/traditions_info.json",
        "/corpus/traditions_info.json",
    ];

    for (const url of urls) {
        try {
            const response = await fetch(url, {cache: "no-store"});
            if (response.ok) {
                state.traditionInfo = await response.json();
                return state.traditionInfo;
            }
        } catch {
            // Try the next source.
        }
    }

    try {
        const data = await api("/api/geography/traditions");
        state.traditionInfo = data.traditions || {};
        return state.traditionInfo;
    } catch {
        state.traditionInfo = {};
        return state.traditionInfo;
    }
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
                                <select id="viz-select">
                                    ${SIMILARITY_METHODS.map(([value, label]) => `<option value="${value}">${label}</option>`).join("")}
                                </select>
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
                        <div class="card-header">
                            <h3 class="card-title">Point Information</h3>
                            <button class="btn btn-outline point-info-modal-btn" id="open-point-info-modal" type="button" hidden>Open in modal</button>
                        </div>
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

            <div id="neighborsModal">
                <div class="modal-content-reader neighbors-modal-content">
                    <div class="card-header">
                        <strong id="neighborsModalTitle">Point Information</strong>
                        <button class="btn" id="close-neighbors-modal" type="button">Close</button>
                    </div>
                    <div class="neighbors-modal-results" id="neighborsModalResults"></div>
                </div>
            </div>
        </main>
    `;

    bindEmbeddingsControls();

    try {
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
    searchText.addEventListener("focus", () => {
        scheduleSearchWarmup(state.selectedModel);
    });
    searchText.addEventListener("input", () => {
        searchBtn.disabled = searchText.value.trim().length === 0 || !state.selectedModel;
        if (searchText.value.trim()) scheduleSearchWarmup(state.selectedModel);
    });
    searchBtn.addEventListener("click", performAnalysisSearch);

    document.getElementById("enter-fullscreen").addEventListener("click", toggleFullscreen);
    document.getElementById("exit-fullscreen").addEventListener("click", toggleFullscreen);
    document.getElementById("close-reader-modal").addEventListener("click", () => {
        document.getElementById("readerModal").style.display = "none";
    });
    document.getElementById("close-search-modal").addEventListener("click", closeSearchModal);
    document.getElementById("close-neighbors-modal").addEventListener("click", closePointInfoModal);

    state.keydownHandler = (event) => {
        if (event.key === "Escape" && plotContainer && plotContainer.classList.contains("fullscreen")) {
            toggleFullscreen();
        }
    };
    document.addEventListener("keydown", state.keydownHandler);
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
    window.setTimeout(() => {
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
    const scatterPlot = document.getElementById("scatter-plot");
    if (!plotContainer) return;

    plotContainer.classList.toggle("fullscreen");
    if (window.Plotly && scatterPlot && scatterPlot.dataset.plotly === "1") {
        setTimeout(() => Plotly.relayout(scatterPlot, {autosize: true}), 100);
    }
    resizeEmbeddedPlots();
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
    scatterPlot.dataset.plotly = "";
    scatterPlot.style.minWidth = "";
    scatterPlot.style.minHeight = "";
    if (window.Plotly) Plotly.purge(scatterPlot);
    scatterPlot.innerHTML = "";

    try {
        if (!HTML_ONLY_METHODS.has(method)) {
            try {
                const data = await api(`/api/similarity/projections/${encodeURIComponent(state.selectedModel)}/${encodeURIComponent(method)}`);
                await renderProjectionPlot(data);
                loadingPlaceholder.style.display = "none";
                updateStatus("Ready", "loaded");
                return;
            } catch {
                // Fall back to the generated HTML file, matching the old template behavior for missing JSON data.
            }
        }

        const savedHtml = await api(`/api/similarity/saved-html/${encodeURIComponent(state.selectedModel)}/${encodeURIComponent(method)}`);
        if (!savedHtml.exists || !savedHtml.url) throw new Error(savedHtml.reason || "File not found. Generate the data through the CLI.");

        scatterPlot.style.display = "block";
        await renderSavedPlotInto(scatterPlot, savedHtml.url, {
            title: SIMILARITY_METHODS.find(([value]) => value === method)?.[1],
            preserveSize: HTML_ONLY_METHODS.has(method),
        });
        loadingPlaceholder.style.display = "none";
        updateStatus("Ready", "loaded");
    } catch (error) {
        loadingPlaceholder.innerHTML = `Error: ${escapeHtml(error.message)}`;
        loadingPlaceholder.style.display = "block";
        updateStatus("Load error", "error");
    }
}

async function renderProjectionPlot(data) {
    const scatterPlot = document.getElementById("scatter-plot");
    const points = Array.isArray(data.points) ? data.points : [];
    if (!window.Plotly || !points.length) throw new Error("Projection data is empty.");

    scatterPlot.style.display = "block";

    await loadTraditionInfo();
    const groupedPoints = new Map();
    points.forEach((point) => {
        const tradition = point.tradition || "Unknown";
        if (!groupedPoints.has(tradition)) groupedPoints.set(tradition, []);
        groupedPoints.get(tradition).push(point);
    });

    const traditions = [...groupedPoints.keys()];
    const colorMap = getColorMap(traditions);
    const showLegend = scatterPlot.clientWidth >= 720;

    const traces = traditions.map((tradition) => {
        const pts = groupedPoints.get(tradition) || [];
        return {
            type: "scattergl",
            x: pts.map((point) => point.x),
            y: pts.map((point) => point.y),
            mode: "markers",
            name: tradition,
            marker: {
                size: 6,
                opacity: 0.74,
                color: colorMap[tradition],
                line: {width: 0.4, color: "rgba(255,255,255,0.85)"},
            },
            customdata: pts.map((point) => [
                point.id,
                point.tradition,
                point.chunk_index,
                normalizePreviewText(point.text).substring(0, 220),
            ]),
            hoverinfo: "none",
        };
    });

    const layout = {
        margin: showLegend ? {l: 50, r: 176, t: 28, b: 48} : {l: 50, r: 28, t: 28, b: 48},
        plot_bgcolor: "#fbfcfd",
        paper_bgcolor: "#fff",
        hovermode: "closest",
        hoverlabel: {
            align: "left",
            bgcolor: "#fff",
            bordercolor: "#ced4da",
            font: {color: "#212529", size: 12},
            namelength: 24,
        },
        showlegend: showLegend,
        legend: {
            orientation: "v",
            x: 1.02,
            xanchor: "left",
            y: 1,
            yanchor: "top",
            bgcolor: "rgba(255,255,255,0.84)",
            bordercolor: "rgba(222,226,230,0.9)",
            borderwidth: 1,
            font: {size: 11, color: "#495057"},
            itemwidth: 30,
        },
        xaxis: {
            automargin: true,
            gridcolor: "#edf1f5",
            zeroline: false,
            tickfont: {size: 11, color: "#6c757d"},
        },
        yaxis: {
            automargin: true,
            gridcolor: "#edf1f5",
            zeroline: false,
            tickfont: {size: 11, color: "#6c757d"},
        },
    };

    await Plotly.newPlot(scatterPlot, traces, layout, {
        responsive: true,
        displaylogo: false,
        displayModeBar: true,
        scrollZoom: true,
    });
    scatterPlot.style.display = "block";
    scatterPlot.dataset.plotly = "1";
    bindProjectionTooltip(scatterPlot);
    scatterPlot.on("plotly_click", (event) => {
        if (event.points && event.points[0]) {
            displayPointInfo(event.points[0].customdata[0], event.points[0].customdata[2]);
        }
    });
}

function bindProjectionTooltip(scatterPlot) {
    const canvas = document.getElementById("plotCanvas");
    if (!canvas) return;
    clearProjectionTooltipHandlers(scatterPlot);

    let tooltip = canvas.querySelector(".plot-hover-tooltip");
    if (!tooltip) {
        tooltip = document.createElement("div");
        tooltip.className = "plot-hover-tooltip";
        canvas.appendChild(tooltip);
    }

    const hideTooltip = () => {
        tooltip.classList.remove("visible");
    };

    const showTooltip = (event) => {
        const point = event.points && event.points[0];
        if (!point) return hideTooltip();

        const custom = Array.isArray(point.customdata) ? point.customdata : [];
        tooltip.innerHTML = `
            <div class="plot-hover-title">${escapeHtml(custom[1] || "Unknown")}</div>
            <div class="plot-hover-meta">ID: ${escapeHtml(custom[0] || "")} | Chunk: ${escapeHtml(custom[2] ?? 0)}</div>
            <div class="plot-hover-text">${escapeHtml(normalizePreviewText(custom[3]) || "No preview available.")}</div>
        `;

        tooltip.classList.add("visible");
        positionPlotTooltip(canvas, tooltip, event.event);
    };

    const moveTooltip = (event) => {
        if (tooltip.classList.contains("visible")) {
            positionPlotTooltip(canvas, tooltip, event);
        }
    };

    scatterPlot.on("plotly_hover", showTooltip);
    scatterPlot.on("plotly_unhover", hideTooltip);
    scatterPlot.addEventListener("mouseleave", hideTooltip);
    scatterPlot.addEventListener("mousemove", moveTooltip);
    scatterPlot._projectionTooltipHandlers = {showTooltip, hideTooltip, moveTooltip};
}

function clearProjectionTooltipHandlers(scatterPlot) {
    const handlers = scatterPlot._projectionTooltipHandlers;
    if (!handlers) return;

    if (typeof scatterPlot.removeListener === "function") {
        scatterPlot.removeListener("plotly_hover", handlers.showTooltip);
        scatterPlot.removeListener("plotly_unhover", handlers.hideTooltip);
    }
    scatterPlot.removeEventListener("mouseleave", handlers.hideTooltip);
    scatterPlot.removeEventListener("mousemove", handlers.moveTooltip);
    scatterPlot._projectionTooltipHandlers = null;
}

function positionPlotTooltip(container, tooltip, event) {
    const rect = container.getBoundingClientRect();
    const clientX = event?.clientX ?? (rect.left + rect.width / 2);
    const clientY = event?.clientY ?? (rect.top + rect.height / 2);
    const gap = 14;
    const padding = 10;

    tooltip.style.left = "0px";
    tooltip.style.top = "0px";

    const width = tooltip.offsetWidth;
    const height = tooltip.offsetHeight;
    let left = clientX - rect.left + gap;
    let top = clientY - rect.top + gap;

    if (left + width + padding > rect.width) {
        left = clientX - rect.left - width - gap;
    }
    if (top + height + padding > rect.height) {
        top = clientY - rect.top - height - gap;
    }

    left = Math.max(padding, Math.min(left, rect.width - width - padding));
    top = Math.max(padding, Math.min(top, rect.height - height - padding));

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
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
        item.book_title
        || item.filename
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
    const text = item.text || item.text_preview || "";
    if (!text) {
        return '<span class="chunk-text-empty">Chunk text is unavailable.</span>';
    }
    return highlightText(text, query);
}

function renderNeighborCards(neighbors, emptyText) {
    if (!neighbors.length) {
        return `<div class="neighbor-empty">${escapeHtml(emptyText)}</div>`;
    }

    return neighbors.map((neighbor) => `
        <div class="neighbor-item" data-neighbor-id="${escapeAttribute(neighbor.id)}" data-neighbor-chunk="${escapeAttribute(neighbor.chunk_index)}">
            <span class="badge" style="background:#dee2e6; color:#212529">${escapeHtml(neighbor.tradition)}</span>
            <div class="neighbor-meta">${escapeHtml(chunkMetaLine(neighbor))}</div>
            <div class="neighbor-text">${escapeHtml(neighbor.text || neighbor.text_preview || "")}</div>
            <div class="neighbor-stats">Similarity: ${(Number(neighbor.similarity_score || 0) * 100).toFixed(1)}%</div>
        </div>
    `).join("");
}

function bindNeighborCards(container, options = {}) {
    if (!container) return;
    container.querySelectorAll(".neighbor-item").forEach((item) => {
        item.addEventListener("click", () => {
            if (options.closeModal) closePointInfoModal();
            displayPointInfo(item.dataset.neighborId, item.dataset.neighborChunk);
        });
    });
}

function openPointInfoModal(point, neighbors, crossTraditionNeighbors) {
    const modal = document.getElementById("neighborsModal");
    const modalTitle = document.getElementById("neighborsModalTitle");
    const results = document.getElementById("neighborsModalResults");
    if (!modal || !modalTitle || !results) return;

    modalTitle.textContent = "Point Information";
    results.innerHTML = `
        <div class="neighbors-modal-detail">
            <span class="badge">${escapeHtml(point.tradition)}</span>
            <div class="search-result-meta">${escapeHtml(chunkMetaLine(point))}</div>
            <div class="neighbors-modal-text"><strong>ID:</strong> ${escapeHtml(point.id)}<br><br>${escapeHtml(point.text)}</div>
        </div>
        <div class="neighbors-modal-summary">
            <strong>Nearest neighbors</strong>
            <span>${escapeHtml(NEAREST_NEIGHBOR_COUNT)} results</span>
        </div>
        <div class="neighbors-modal-list">
            ${renderNeighborCards(neighbors, "No nearest chunks found.")}
        </div>
        <div class="neighbors-modal-summary">
            <strong>${escapeHtml(CROSS_TRADITION_NEIGHBOR_COUNT)} nearest from other traditions</strong>
            <span>${escapeHtml(point.tradition || "Unknown")} excluded</span>
        </div>
        <div class="neighbors-modal-list">
            ${renderNeighborCards(crossTraditionNeighbors, "No cross-tradition neighbors found.")}
        </div>
    `;
    modal.style.display = "block";
    bindNeighborCards(results, {closeModal: true});
}

function closePointInfoModal() {
    const modal = document.getElementById("neighborsModal");
    if (modal) modal.style.display = "none";
}

async function displayPointInfo(pointId, chunkIndex = null) {
    if (!state.selectedModel || !pointId) return;

    const infoContent = document.getElementById("infoContent");
    if (!infoContent) return;

    const modalButton = document.getElementById("open-point-info-modal");
    if (modalButton) {
        modalButton.hidden = true;
        modalButton.onclick = null;
    }

    infoContent.innerHTML = '<div style="text-align:center; color:#6c757d">Loading...</div>';
    infoContent.classList.remove("empty");

    try {
        const pointQuery = chunkIndex !== null && chunkIndex !== undefined && chunkIndex !== ""
            ? `?chunk_index=${encodeURIComponent(chunkIndex)}`
            : "";
        const neighborsQuery = new URLSearchParams({n: String(NEAREST_NEIGHBOR_COUNT)});
        if (chunkIndex !== null && chunkIndex !== undefined && chunkIndex !== "") {
            neighborsQuery.set("chunk_index", String(chunkIndex));
        }
        const crossTraditionQuery = new URLSearchParams(neighborsQuery);
        crossTraditionQuery.set("n", String(CROSS_TRADITION_NEIGHBOR_COUNT));
        crossTraditionQuery.set("exclude_same_tradition", "true");

        const [point, neighborsData, crossTraditionData] = await Promise.all([
            api(`/api/similarity/points/${encodeURIComponent(state.selectedModel)}/${encodeURIComponent(pointId)}${pointQuery}`),
            api(`/api/similarity/points/${encodeURIComponent(state.selectedModel)}/${encodeURIComponent(pointId)}/neighbors?${neighborsQuery.toString()}`),
            api(`/api/similarity/points/${encodeURIComponent(state.selectedModel)}/${encodeURIComponent(pointId)}/neighbors?${crossTraditionQuery.toString()}`),
        ]);

        const neighbors = Array.isArray(neighborsData.neighbors) ? neighborsData.neighbors : [];
        const crossTraditionNeighbors = Array.isArray(crossTraditionData.neighbors) ? crossTraditionData.neighbors : [];
        let html = `
            <div class="badge">${escapeHtml(point.tradition)}</div>
            <div class="search-result-meta">${escapeHtml(chunkMetaLine(point))}</div>
            <div class="text-preview"><strong>ID:</strong> ${escapeHtml(point.id)}<br><br>${escapeHtml(point.text)}</div>
            <h4 style="margin: 16px 0 8px; font-size:14px; color:#111;">Nearest neighbors:</h4>
            ${renderNeighborCards(neighbors, "No nearest chunks found.")}
            <h4 style="margin: 18px 0 8px; font-size:14px; color:#111;">Nearest from other traditions:</h4>
            ${renderNeighborCards(crossTraditionNeighbors, "No cross-tradition neighbors found.")}
        `;

        infoContent.innerHTML = html;
        bindNeighborCards(infoContent);
        if (modalButton) {
            modalButton.hidden = false;
            modalButton.onclick = () => openPointInfoModal(point, neighbors, crossTraditionNeighbors);
        }
    } catch (error) {
        infoContent.innerHTML = `<div style="color:#d32f2f">Error: ${escapeHtml(error.message)}</div>`;
    }
}

async function initializeAnalysisLibrary() {
    await loadTraditionInfo();

    const container = document.getElementById("tree-container");
    if (!container) return;

    try {
        const documents = await ensureCorpusDocuments("chunked");
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

const SEARCH_JOB_POLL_MS = 1000;

function delay(ms) {
    return new Promise((resolve) => {
        window.setTimeout(resolve, ms);
    });
}

function renderSearchStatus(job) {
    const status = job.status === "queued" ? "Queued" : "Searching";
    const model = escapeHtml(String(job.model || "").replace(/_/g, "/"));
    return `
        <div class="search-loading">
            ${status}...
            <small>Model: ${model}</small>
        </div>
    `;
}

async function runSemanticSearch({query, model, topK = 20, onStatus, shouldContinue}) {
    const startJob = () => api("/api/similarity/search/jobs", {
        method: "POST",
        body: JSON.stringify({
            query,
            model,
            top_k: topK,
        }),
    });

    let started = await startJob();
    let restarts = 0;

    if (Array.isArray(started.results) && started.status === undefined) {
        return started;
    }

    if (!started.job_id) {
        throw new Error("Search did not start.");
    }

    if (onStatus) onStatus(started);

    while (true) {
        if (shouldContinue && !shouldContinue()) return null;
        await delay(SEARCH_JOB_POLL_MS);
        if (shouldContinue && !shouldContinue()) return null;

        let job;
        try {
            job = await api(`/api/similarity/search/jobs/${encodeURIComponent(started.job_id)}`);
        } catch (error) {
            const message = String(error.message || "");
            const jobWasLost = message.includes("Search job not found") || message.includes("Failed to fetch");
            if (jobWasLost && restarts < 2) {
                restarts += 1;
                await delay(SEARCH_JOB_POLL_MS);
                if (shouldContinue && !shouldContinue()) return null;
                started = await startJob();
                if (onStatus) onStatus({...started, status: "queued"});
                continue;
            }
            throw error;
        }

        if (job.status === "complete") {
            return {
                query: job.query,
                model: job.model,
                results: Array.isArray(job.results) ? job.results : [],
                total: Number(job.total || 0),
            };
        }
        if (job.status === "failed") {
            throw new Error(job.error || "Search failed.");
        }
        if (onStatus) onStatus(job);
    }
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
            shouldContinue: () => requestId === state.analysisSearchRequestId,
            onStatus: (job) => {
                if (requestId === state.analysisSearchRequestId) {
                    setSearchResults(renderSearchStatus(job));
                }
            },
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

    const resultsContainer = document.getElementById("searchResults");
    resultsContainer.querySelectorAll(".search-result-item").forEach((item) => {
        item.addEventListener("click", () => {
            displaySearchModalPointInfo(item.dataset.pointId, item.dataset.chunkIndex);
        });
    });
}

function renderSearchResultItem(result, data) {
    const similarityPercent = Math.round(Number(result.similarity_score || 0) * 100);
    let scoreClass = "score-low";
    if (similarityPercent >= 60) scoreClass = "score-high";
    else if (similarityPercent >= 40) scoreClass = "score-medium";

    return `
        <button class="search-result-item" type="button" data-point-id="${escapeAttribute(result.id)}" data-chunk-index="${escapeAttribute(result.chunk_index)}">
            <span class="search-result-topline">
                <span class="result-tradition">${escapeHtml(result.tradition)}</span>
                <span class="result-score ${scoreClass}">${similarityPercent}% similarity</span>
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
        const pointQuery = chunkIndex !== null && chunkIndex !== undefined && chunkIndex !== ""
            ? `?chunk_index=${encodeURIComponent(chunkIndex)}`
            : "";
        const neighborsQuery = new URLSearchParams({n: "5"});
        if (chunkIndex !== null && chunkIndex !== undefined && chunkIndex !== "") {
            neighborsQuery.set("chunk_index", String(chunkIndex));
        }

        const [point, neighborsData] = await Promise.all([
            api(`/api/similarity/points/${encodeURIComponent(state.selectedModel)}/${encodeURIComponent(pointId)}${pointQuery}`),
            api(`/api/similarity/points/${encodeURIComponent(state.selectedModel)}/${encodeURIComponent(pointId)}/neighbors?${neighborsQuery.toString()}`),
        ]);
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

        const resultsContainer = document.getElementById("searchResults");
        resultsContainer.querySelectorAll(".search-result-item").forEach((item) => {
            item.addEventListener("click", () => {
                displaySearchModalPointInfo(item.dataset.pointId, item.dataset.chunkIndex);
            });
        });
    } catch (error) {
        setSearchResults(`<div class="search-empty">Load error: ${escapeHtml(error.message)}</div>`);
    }
}

async function renderClusterAnalysis() {
    document.title = "Clustering Analysis";
    app.innerHTML = `
        <main class="cluster-page container">
            <div class="controls-panel">
                <div class="form-group">
                    <label>Model:</label>
                    <select id="global-model-select"><option value="">Loading models...</option></select>
                </div>
                <div class="form-group">
                    <label>Algorithm:</label>
                    <select id="clustering-select">
                        ${CLUSTERING_ALGORITHMS.map(([value, label]) => `<option value="${value}">${label}</option>`).join("")}
                    </select>
                </div>
                <button class="btn btn-primary" type="button" id="cluster-refresh">Refresh Data</button>
            </div>

            <div class="metrics-grid" id="metrics-grid" style="display: none;"></div>

            <div id="content-container">
                <div class="plot-grid" id="plots-grid" style="display: none;">
                    <div class="plot-card">
                        <div class="plot-header">Cluster Visualization (UMAP)</div>
                        <div class="plot-body"><div class="cluster-plot" id="clusters-plot"></div></div>
                    </div>
                    <div class="plot-card">
                        <div class="plot-header">Tradition Correspondence Matrix</div>
                        <div class="plot-body"><div class="cluster-plot" id="confusion-plot"></div></div>
                    </div>
                </div>
                <div class="loading-state" id="loading-state">
                    Select a model and algorithm to display data...
                </div>
            </div>
        </main>
    `;

    const modelSelect = document.getElementById("global-model-select");
    const clusterSelect = document.getElementById("clustering-select");
    const refresh = document.getElementById("cluster-refresh");

    try {
        await ensureModels();
        modelSelect.innerHTML = renderModelOptions();
        modelSelect.value = state.selectedModel;
        modelSelect.addEventListener("change", triggerClusterUpdate);
        clusterSelect.addEventListener("change", triggerClusterUpdate);
        refresh.addEventListener("click", triggerClusterUpdate);
        triggerClusterUpdate();
    } catch {
        modelSelect.innerHTML = '<option value="">Load error</option>';
    }
}

function getMetricScoreClass(value, name) {
    if (value === null || value === undefined) return "";
    if (name === "silhouette_score") return value >= 0.5 ? "val-good" : (value >= 0.25 ? "val-warn" : "val-bad");
    if (["adjusted_rand_score", "normalized_mutual_info", "v_measure"].includes(name)) {
        return value >= 0.7 ? "val-good" : (value >= 0.4 ? "val-warn" : "val-bad");
    }
    return "";
}

async function triggerClusterUpdate() {
    const modelSelect = document.getElementById("global-model-select");
    const clusterSelect = document.getElementById("clustering-select");
    const metricsGrid = document.getElementById("metrics-grid");
    const plotsGrid = document.getElementById("plots-grid");
    const loadingState = document.getElementById("loading-state");
    const clustersPlot = document.getElementById("clusters-plot");
    const confusionPlot = document.getElementById("confusion-plot");
    if (!modelSelect || !clusterSelect || !modelSelect.value) return;

    persistSelectedModel(modelSelect.value);
    const algorithm = clusterSelect.value;

    plotsGrid.style.display = "none";
    metricsGrid.style.display = "none";
    loadingState.style.display = "block";
    loadingState.textContent = "Loading clusters and metrics...";
    if (clustersPlot) clustersPlot.innerHTML = "";
    if (confusionPlot) confusionPlot.innerHTML = "";

    try {
        const savedPlots = await api(`/api/clustering/${encodeURIComponent(state.selectedModel)}/${encodeURIComponent(algorithm)}/plots`);
        const metrics = await api(`/api/clustering/${encodeURIComponent(state.selectedModel)}/${encodeURIComponent(algorithm)}/metrics`);

        const metricsHtml = Object.keys(METRIC_NAMES).map((key) => {
            const value = metrics[key];
            if (value === undefined || value === null) return "";
            const formatted = key === "noise_ratio"
                ? `${(Number(value) * 100).toFixed(1)}%`
                : (typeof value === "number" && key !== "n_clusters_found" ? value.toFixed(3) : value);
            const cssClass = getMetricScoreClass(value, key);
            return `
                <div class="metric-card">
                    <div class="metric-value ${cssClass}">${escapeHtml(formatted)}</div>
                    <div class="metric-label">${escapeHtml(METRIC_NAMES[key] || key)}</div>
                </div>
            `;
        }).join("");

        metricsGrid.innerHTML = metricsHtml;
        metricsGrid.style.display = metricsHtml.trim() ? "grid" : "none";

        if (!savedPlots.clusters?.exists && !savedPlots.confusion_matrix?.exists) {
            throw new Error("Clustering files were not found. Generate them through the CLI.");
        }

        loadingState.style.display = "none";
        plotsGrid.style.display = "grid";

        const renderTasks = [];
        if (savedPlots.clusters?.exists && savedPlots.clusters.url) {
            renderTasks.push(renderSavedPlotInto(clustersPlot, savedPlots.clusters.url, {title: `Embedding clustering (${algorithm})`}));
        } else if (clustersPlot) {
            clustersPlot.innerHTML = '<div class="plot-loading">Cluster plot was not generated.</div>';
        }

        if (savedPlots.confusion_matrix?.exists && savedPlots.confusion_matrix.url) {
            renderTasks.push(renderSavedPlotInto(confusionPlot, savedPlots.confusion_matrix.url, {title: "Tradition correspondence matrix"}));
        } else if (confusionPlot) {
            confusionPlot.innerHTML = '<div class="plot-loading">Correspondence matrix was not generated.</div>';
        }

        await Promise.all(renderTasks);
    } catch (error) {
        loadingState.innerHTML = escapeHtml(error.message || "Clustering files were not found. Generate them through the CLI.");
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
        if (searchInput.value.trim()) scheduleSearchWarmup(modelSelect.value);
        if (searchInput.value.trim()) performSearchPageSearch();
    });
    searchBtn.addEventListener("click", performSearchPageSearch);
    searchInput.addEventListener("focus", () => {
        scheduleSearchWarmup(modelSelect.value);
    });
    searchInput.addEventListener("input", () => {
        if (searchInput.value.trim()) scheduleSearchWarmup(modelSelect.value);
    });
    searchInput.addEventListener("keypress", (event) => {
        if (event.key === "Enter") performSearchPageSearch();
    });

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
            shouldContinue: () => requestId === state.searchPageRequestId,
            onStatus: (job) => {
                if (requestId === state.searchPageRequestId) {
                    resultsArea.innerHTML = renderSearchStatus(job);
                }
            },
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
        ${results.map((result) => {
            const similarityPercent = Math.round(Number(result.similarity_score || 0) * 100);
            let scoreClass = "score-low";
            if (similarityPercent >= 60) scoreClass = "score-high";
            else if (similarityPercent >= 40) scoreClass = "score-medium";

            return `
                <div class="result-item" data-point-id="${escapeAttribute(result.id)}" data-model="${escapeAttribute(data.model)}" data-chunk-index="${escapeAttribute(result.chunk_index)}">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <span class="result-tradition">${escapeHtml(result.tradition)}</span>
                        <span class="result-score ${scoreClass}">${similarityPercent}% similarity</span>
                    </div>
                    <div class="search-result-meta">${escapeHtml(searchResultMetaLine(result))}</div>
                    <div class="result-text chunk-text">${chunkTextHtml(result, data.query)}</div>
                </div>
            `;
        }).join("")}
    `;

    resultsArea.querySelectorAll(".result-item").forEach((item) => {
        item.addEventListener("click", () => showPointDetails(item.dataset.pointId, item.dataset.model, item.dataset.chunkIndex));
    });
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
        }, "*");

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

function renderFutureDomain(name) {
    document.title = `MythoScope - ${name}`;
    app.innerHTML = `
        <main class="placeholder-page">
            <h2>${escapeHtml(name.charAt(0).toUpperCase() + name.slice(1))}</h2>
        </main>
    `;
}

const fadeStyle = document.createElement("style");
fadeStyle.textContent = `
    @keyframes fadeOut {
        0% { opacity: 1; }
        70% { opacity: 1; }
        100% { opacity: 0; visibility: hidden; }
    }
`;
document.head.appendChild(fadeStyle);

window.addEventListener("message", (event) => {
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
window.addEventListener("resize", resizeEmbeddedPlots);
