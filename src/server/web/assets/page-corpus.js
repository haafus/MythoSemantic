import {
    app, api, state,
    buildCorpusApiUrl, corpusTraditionKey,
    ensureCorpusDocuments, escapeAttribute, escapeHtml,
    formatNumber, groupDocuments,
} from "./core.js";

export async function renderCorpus() {
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
    const active = state.selectedCorpusDoc && state.selectedCorpusDoc.title === doc.title
        && state.selectedCorpusDoc.major_tradition === doc.major_tradition
        && state.selectedCorpusDoc.tradition === doc.tradition;
    return `
        <li>
            <button class="document-button${active ? " active" : ""}" type="button" data-doc-index="${index}">
                ${escapeHtml(doc.title)}
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
    renderCorpusLibrary();
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
