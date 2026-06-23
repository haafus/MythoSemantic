import {
    app, state,
    ensureModels, escapeHtml,
    persistSelectedModel, renderModelOptions,
} from "./core.js";
import {
    bindSearchResultClicks, renderSearchResultItem,
    runSemanticSearch,
} from "./search-utils.js";

export async function renderSearchSimilarities(params) {
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
            <strong>Found:</strong> ${results.length} results
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
