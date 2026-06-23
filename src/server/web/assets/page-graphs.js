import { app, state, api, escapeHtml, escapeAttribute } from "./core.js";
import { renderLibraryTree, setActiveBook } from "./library-tree.js";

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

export async function renderGraphPage(graphType) {
    document.title = `MythoScope - ${graphType.charAt(0).toUpperCase() + graphType.slice(1)}`;
    app.innerHTML = `
        <main class="graph-page container">
            <div class="graph-sidebar">
                <div class="graph-sidebar-header">Books</div>
                <div id="graphBookList">Loading...</div>
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

    const bookList = document.getElementById("graphBookList");
    bookList.addEventListener("book-select", (e) => {
        setActiveBook(bookList, e.detail.doc);
        loadGraphData(e.detail.doc.title, graphType);
    });
    await renderLibraryTree(bookList);
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
