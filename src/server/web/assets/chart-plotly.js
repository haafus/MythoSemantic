// Plotly chart implementation (~3.5 MB).
// CDN: <script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>
// To activate: change the import in chart.js to "./chart-plotly.js" and add the CDN script to index.html

import { escapeHtml, normalizePreviewText } from "./core.js";

let activeChart = null;
let tooltipHandlers = null;

export function destroyChart(el) {
    if (tooltipHandlers) {
        clearTooltipHandlers(el);
    }
    if (window.Plotly) {
        try { Plotly.purge(el); } catch { /* already torn down */ }
    }
    activeChart = null;
    el.innerHTML = "";
    el.dataset.plotly = "";
}

export function resizeChart(el) {
    if (window.Plotly && el.dataset.plotly === "1") {
        setTimeout(() => Plotly.relayout(el, { autosize: true }), 100);
    }
}

export async function renderScatter(el, data, { colorMap, onPointClick }) {
    const points = Array.isArray(data.points) ? data.points : [];
    if (!window.Plotly || !points.length) throw new Error("Projection data is empty.");

    el.style.display = "block";

    const traditions = [...new Set(points.map((p) => p.tradition || "Unknown"))];
    const showLegend = el.clientWidth >= 720;

    const traces = traditions.map((tradition) => {
        const pts = points.filter((p) => (p.tradition || "Unknown") === tradition);
        return {
            type: "scattergl",
            x: pts.map((p) => p.x),
            y: pts.map((p) => p.y),
            mode: "markers",
            name: tradition,
            marker: {
                size: 6,
                opacity: 0.74,
                color: colorMap[tradition],
                line: { width: 0.4, color: "rgba(255,255,255,0.85)" },
            },
            customdata: pts.map((p) => [
                p.id,
                p.tradition,
                p.chunk_index,
                normalizePreviewText(p.text).substring(0, 220),
            ]),
            hoverinfo: "none",
        };
    });

    const layout = {
        margin: showLegend ? { l: 50, r: 176, t: 28, b: 48 } : { l: 50, r: 28, t: 28, b: 48 },
        plot_bgcolor: "#fbfcfd",
        paper_bgcolor: "#fff",
        hovermode: "closest",
        hoverlabel: {
            align: "left",
            bgcolor: "#fff",
            bordercolor: "#ced4da",
            font: { color: "#212529", size: 12 },
            namelength: 24,
        },
        showlegend: showLegend,
        legend: {
            orientation: "v",
            x: 1.02, xanchor: "left",
            y: 1, yanchor: "top",
            bgcolor: "rgba(255,255,255,0.84)",
            bordercolor: "rgba(222,226,230,0.9)",
            borderwidth: 1,
            font: { size: 11, color: "#495057" },
            itemwidth: 30,
        },
        xaxis: { automargin: true, gridcolor: "#edf1f5", zeroline: false, tickfont: { size: 11, color: "#6c757d" } },
        yaxis: { automargin: true, gridcolor: "#edf1f5", zeroline: false, tickfont: { size: 11, color: "#6c757d" } },
    };

    await Plotly.newPlot(el, traces, layout, { responsive: true, displaylogo: false, displayModeBar: true, scrollZoom: true });
    el.dataset.plotly = "1";
    bindTooltip(el);
    el.on("plotly_click", (event) => {
        if (event.points && event.points[0]) {
            onPointClick(event.points[0].customdata[0], event.points[0].customdata[2]);
        }
    });
}

export async function renderHeatmap(el, data) {
    const traditions = data.traditions || [];
    const distances = data.distances || [];
    if (!window.Plotly || !traditions.length) throw new Error("Heatmap data is empty.");

    el.style.display = "block";

    const trace = {
        type: "heatmap",
        z: distances,
        x: traditions,
        y: traditions,
        colorscale: "Viridis",
        hovertemplate: "Distance between %{x} and %{y}: %{z:.4f}<extra></extra>",
    };

    const layout = {
        margin: { l: 120, r: 60, t: 40, b: 140 },
        paper_bgcolor: "#fff",
        plot_bgcolor: "#fff",
        xaxis: { tickangle: 45, tickfont: { size: 11 }, automargin: true },
        yaxis: { tickfont: { size: 11 }, automargin: true },
    };

    await Plotly.newPlot(el, [trace], layout, { responsive: true, displaylogo: false, displayModeBar: true });
    el.dataset.plotly = "1";
}

export async function renderDistribution(el, data, { colorMap }) {
    const traditions = data.traditions || [];
    if (!window.Plotly || !traditions.length) throw new Error("Distribution data is empty.");

    el.style.display = "block";

    const names = traditions.map((t) => t.name);
    const trace = {
        type: "bar",
        x: traditions.map((t) => t.chunks),
        y: names,
        orientation: "h",
        marker: { color: names.map((n) => colorMap[n]) },
        text: traditions.map((t) => `${t.chunks.toLocaleString()} chunks (${t.percentage}%)`),
        textposition: "outside",
        cliponaxis: false,
        hovertemplate: "<b>%{y}</b><br>Chunks: %{x:,}<br>Source texts: %{customdata}<extra></extra>",
        customdata: traditions.map((t) => t.doc_count),
    };

    const layout = {
        margin: { l: 160, r: 140, t: 40, b: 60 },
        paper_bgcolor: "#fff",
        plot_bgcolor: "rgba(248,249,250,0.95)",
        showlegend: false,
        height: Math.max(500, 28 * names.length + 120),
        xaxis: { title: { text: "Number of chunks" }, automargin: true, tickfont: { size: 11 } },
        yaxis: { autorange: "reversed", tickfont: { size: 11 }, automargin: true },
    };

    await Plotly.newPlot(el, [trace], layout, { responsive: true, displaylogo: false, displayModeBar: true });
    el.dataset.plotly = "1";
}

function bindTooltip(plotEl) {
    const canvas = document.getElementById("plotCanvas");
    if (!canvas) return;
    clearTooltipHandlers(plotEl);

    let tooltip = canvas.querySelector(".plot-hover-tooltip");
    if (!tooltip) {
        tooltip = document.createElement("div");
        tooltip.className = "plot-hover-tooltip";
        canvas.appendChild(tooltip);
    }

    const hide = () => { tooltip.classList.remove("visible"); };

    const show = (event) => {
        const point = event.points && event.points[0];
        if (!point) return hide();

        const custom = Array.isArray(point.customdata) ? point.customdata : [];
        tooltip.innerHTML = `
            <div class="plot-hover-title">${escapeHtml(custom[1] || "Unknown")}</div>
            <div class="plot-hover-meta">ID: ${escapeHtml(custom[0] || "")} | Chunk: ${escapeHtml(custom[2] ?? 0)}</div>
            <div class="plot-hover-text">${escapeHtml(normalizePreviewText(custom[3]) || "No preview available.")}</div>
        `;
        tooltip.classList.add("visible");
        positionTooltip(canvas, tooltip, event.event);
    };

    const move = (event) => {
        if (tooltip.classList.contains("visible")) {
            positionTooltip(canvas, tooltip, event);
        }
    };

    plotEl.on("plotly_hover", show);
    plotEl.on("plotly_unhover", hide);
    plotEl.addEventListener("mouseleave", hide);
    plotEl.addEventListener("mousemove", move);
    tooltipHandlers = { plotEl, show, hide, move };
}

function clearTooltipHandlers(plotEl) {
    if (!tooltipHandlers) return;
    const h = tooltipHandlers;
    if (typeof plotEl.removeListener === "function") {
        plotEl.removeListener("plotly_hover", h.show);
        plotEl.removeListener("plotly_unhover", h.hide);
    }
    plotEl.removeEventListener("mouseleave", h.hide);
    plotEl.removeEventListener("mousemove", h.move);
    tooltipHandlers = null;
}

function positionTooltip(container, tooltip, event) {
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

    if (left + width + padding > rect.width) left = clientX - rect.left - width - gap;
    if (top + height + padding > rect.height) top = clientY - rect.top - height - gap;
    left = Math.max(padding, Math.min(left, rect.width - width - padding));
    top = Math.max(padding, Math.min(top, rect.height - height - padding));

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
}
