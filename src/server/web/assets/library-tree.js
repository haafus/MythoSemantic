import {
    state, ensureCorpusDocuments, groupDocuments,
    corpusTraditionKey, escapeAttribute, escapeHtml,
} from "./core.js";

export async function renderLibraryTree(container) {
    container.classList.add("library-tree");
    container.innerHTML = "Loading...";

    try {
        const documents = await ensureCorpusDocuments();
        if (!documents.length) {
            container.innerHTML = '<div class="empty-state">No literature found.</div>';
            return;
        }
        renderTree(container, documents);
    } catch (error) {
        container.innerHTML = `<div class="error-state">${escapeHtml(error.message)}</div>`;
    }
}

function renderTree(container, documents) {
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
                        ${docs.map((doc) => `
                            <li>
                                <button class="document-button" type="button" data-doc-index="${docIndex.get(doc)}">
                                    ${escapeHtml(doc.title)}
                                </button>
                            </li>
                        `).join("")}
                    </ul>
                </div>
            `;
        });

        html += "</div></section>";
    });

    container.innerHTML = html;

    container.querySelectorAll(".major-title").forEach((button) => {
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

    container.querySelectorAll(".tradition-title").forEach((button) => {
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

    container.querySelectorAll(".document-button").forEach((button) => {
        button.addEventListener("click", () => {
            const doc = documents[Number(button.dataset.docIndex)];
            if (!doc) return;
            container.dispatchEvent(new CustomEvent("book-select", {
                detail: { doc },
                bubbles: true,
            }));
        });
    });
}

export function setActiveBook(container, doc) {
    container.querySelectorAll(".document-button").forEach((btn) => btn.classList.remove("active"));
    if (!doc) return;
    const documents = state.corpusDocuments;
    const index = documents.indexOf(doc);
    if (index === -1) return;
    const btn = container.querySelector(`.document-button[data-doc-index="${index}"]`);
    if (btn) btn.classList.add("active");
}
