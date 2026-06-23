import {
    app, api, state,
    buildCorpusApiUrl, escapeAttribute, escapeHtml, formatNumber,
} from "./core.js";
import { renderLibraryTree, setActiveBook } from "./library-tree.js";

export async function renderCorpus() {
    document.title = "MythoScope - Sources";
    app.innerHTML = `
        <main class="corpus-page container">
            <div class="workspace">
                <aside class="panel library-panel">
                    <div class="panel-header">
                        <div class="panel-title">Literature</div>
                    </div>
                    <div id="libraryTree">Loading...</div>
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

    const libraryTree = document.getElementById("libraryTree");
    libraryTree.addEventListener("book-select", (e) => openCorpusDocument(e.detail.doc));

    await renderLibraryTree(libraryTree);
    if (state.selectedCorpusDoc) setActiveBook(libraryTree, state.selectedCorpusDoc);
    renderBookInfo(null);
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
        <div class="book-title">${escapeHtml(doc.title)}</div>
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
            <a class="btn btn-primary${isLoading ? " disabled" : ""}" href="${escapeAttribute(url)}" download="${escapeAttribute(doc.title || "book")}.txt">Download Book</a>
            <a class="btn btn-outline" href="/api/corpus/archive">Download Full Archive</a>
        </div>
    `;
}

async function openCorpusDocument(doc) {
    state.selectedCorpusDoc = doc;
    const libraryTree = document.getElementById("libraryTree");
    if (libraryTree) setActiveBook(libraryTree, doc);
    renderBookInfo(doc, true);

    const readerTitle = document.getElementById("readerTitle");
    const readerContent = document.getElementById("readerContent");
    if (!readerTitle || !readerContent) return;

    readerTitle.textContent = doc.title;
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
