import { api, escapeHtml, escapeAttribute, escapeRegex, state } from "./core.js";

export function scoreClass(similarityScore) {
    const percent = Math.round(Number(similarityScore || 0) * 100);
    let cls = "score-low";
    if (percent >= 60) cls = "score-high";
    else if (percent >= 40) cls = "score-medium";
    return {percent, cls};
}

export function normalizeBookTitle(value) {
    return String(value || "")
        .replace(/\.txt$/i, "")
        .trim();
}

export function resultBookTitle(item) {
    return normalizeBookTitle(item.filename || item.id || "Unknown book");
}

export function chunkMetaLine(item) {
    return `Book: ${resultBookTitle(item)} | Chunk #${item.chunk_index ?? 0}`;
}

export function searchResultMetaLine(item) {
    return `Tradition: ${item.tradition || "Unknown"} | ${chunkMetaLine(item)}`;
}

export function highlightText(text, query) {
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

export function chunkTextHtml(item, query = "") {
    const text = item.text || "";
    if (!text) {
        return '<span class="chunk-text-empty">Chunk text is unavailable.</span>';
    }
    return highlightText(text, query);
}

export function renderSearchResultItem(result, data) {
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

export function bindSearchResultClicks(container, handler) {
    container.querySelectorAll(".search-result-item").forEach((item) => {
        item.addEventListener("click", () => handler(item.dataset.pointId, item.dataset.chunkIndex));
    });
}

export function fetchPointWithNeighbors(pointId, chunkIndex, topK = 6) {
    const params = new URLSearchParams({top_k: String(topK)});
    if (chunkIndex !== null && chunkIndex !== undefined && chunkIndex !== "") {
        params.set("chunk_index", String(chunkIndex));
    }
    return api(`/api/similarity/points/${encodeURIComponent(state.selectedModel)}/${encodeURIComponent(pointId)}?${params}`);
}

export async function runSemanticSearch({query, model, topK = 20}) {
    const results = await api("/api/similarity/search", {
        method: "POST",
        body: JSON.stringify({query, model, top_k: topK}),
    });
    return {results, query, model};
}
