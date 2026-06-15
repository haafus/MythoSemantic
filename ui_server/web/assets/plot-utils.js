import {normalizePreviewText} from "./core.js";

export function bindResponsivePlotFrame(iframe) {
    if (!iframe || iframe.dataset.responsiveFrame === "1") return;
    iframe.dataset.responsiveFrame = "1";
    iframe.addEventListener("load", () => {
        fitPlotFrame(iframe);
        [250, 900, 1800].forEach((delay) => setTimeout(() => fitPlotFrame(iframe), delay));
    });
}

function fitPlotFrame(iframe) {
    try {
        const doc = iframe.contentDocument;
        const win = iframe.contentWindow;
        if (!doc || !doc.body) return;

        doc.documentElement.style.width = "100%";
        doc.documentElement.style.height = "100%";
        doc.body.style.width = "100%";
        doc.body.style.height = "100%";
        doc.body.style.margin = "0";
        doc.body.style.overflow = "auto";

        doc.querySelectorAll(".plotly-graph-div").forEach((plot) => {
            plot.style.width = "100%";
            if (!plot.style.height || plot.style.height === "100%") {
                plot.style.minHeight = "560px";
            }
            if (win.Plotly?.Plots?.resize) {
                win.Plotly.Plots.resize(plot);
            }
        });

        doc.querySelectorAll(".modebar-container").forEach((element) => {
            element.style.display = "";
        });
    } catch {
        // Same-origin frames are adjusted; other frames keep their original layout.
    }
}

export function resizeEmbeddedPlots() {
    document.querySelectorAll("iframe.plot-iframe, .cluster-page .plot-body iframe").forEach((iframe) => {
        fitPlotFrame(iframe);
    });
    if (window.Plotly?.Plots?.resize) {
        document.querySelectorAll(".plotly-managed").forEach((plot) => {
            Plotly.Plots.resize(plot);
        });
    }
}

export async function renderSavedPlotInto(target, url, options = {}) {
    if (!target) return;

    target.classList.add("plotly-managed");
    target.dataset.plotly = "";
    target.innerHTML = '<div class="plot-loading">Loading plot...</div>';

    try {
        if (!window.Plotly) throw new Error("Plotly is not available.");

        const response = await fetch(url, {cache: "force-cache"});
        if (!response.ok) throw new Error(`Plot file returned ${response.status}.`);

        const html = await response.text();
        const spec = extractPlotlySpec(html);
        const layout = normalizeSavedPlotLayout(spec.layout, spec.data, options);
        applySavedPlotSizing(target, layout, options);

        spec.data.forEach(trace => {
            if (Array.isArray(trace.customdata)) {
                trace.customdata.forEach(point => {
                    if (Array.isArray(point) && point[3]) {
                        point[3] = wrapText(normalizePreviewText(point[3]), 60);
                    }
                });
            }
        });

        target.innerHTML = "";
        await Plotly.newPlot(target, spec.data, layout, {
            responsive: true,
            displaylogo: false,
            displayModeBar: true,
            scrollZoom: true,
        });
        target.dataset.plotly = "1";
    } catch (error) {
        console.warn("Falling back to iframe plot:", error);
        renderPlotIframeFallback(target, url);
    }
}

function renderPlotIframeFallback(target, url) {
    target.classList.remove("plotly-managed");
    target.dataset.plotly = "";
    target.innerHTML = "";
    target.style.minWidth = "";
    target.style.minHeight = "";

    const iframe = document.createElement("iframe");
    iframe.className = "plot-iframe";
    bindResponsivePlotFrame(iframe);
    iframe.src = url;
    target.appendChild(iframe);
}

function extractPlotlySpec(html) {
    const callStart = html.lastIndexOf("Plotly.newPlot(");
    if (callStart === -1) throw new Error("Plotly.newPlot call not found.");

    const openParen = html.indexOf("(", callStart);
    let cursor = openParen + 1;

    const graphArg = readPlotlyArgument(html, cursor);
    cursor = skipPlotlySeparator(html, graphArg.next);

    const dataArg = readPlotlyArgument(html, cursor);
    cursor = skipPlotlySeparator(html, dataArg.next);

    const layoutArg = readPlotlyArgument(html, cursor);

    const data = JSON.parse(dataArg.value);
    const layout = layoutArg.value.trim().startsWith("{") ? JSON.parse(layoutArg.value) : {};

    if (!Array.isArray(data)) throw new Error("Plot data is not an array.");
    return {data, layout};
}

function readPlotlyArgument(text, start) {
    let index = skipWhitespace(text, start);
    const char = text[index];

    if (char === "[" || char === "{") {
        const closeChar = char === "[" ? "]" : "}";
        const value = extractBalancedJson(text, index, char, closeChar);
        if (!value) throw new Error("Could not read Plotly JSON argument.");
        return {value, next: index + value.length};
    }

    if (char === '"' || char === "'") {
        const quote = char;
        let escaped = false;
        for (let cursor = index + 1; cursor < text.length; cursor += 1) {
            const current = text[cursor];
            if (escaped) {
                escaped = false;
            } else if (current === "\\") {
                escaped = true;
            } else if (current === quote) {
                return {value: text.slice(index, cursor + 1), next: cursor + 1};
            }
        }
    }

    let cursor = index;
    while (cursor < text.length && text[cursor] !== "," && text[cursor] !== ")") cursor += 1;
    return {value: text.slice(index, cursor), next: cursor};
}

function extractBalancedJson(text, start, openChar, closeChar) {
    let depth = 0;
    let inString = false;
    let escaped = false;

    for (let index = start; index < text.length; index += 1) {
        const char = text[index];

        if (inString) {
            if (escaped) {
                escaped = false;
            } else if (char === "\\") {
                escaped = true;
            } else if (char === '"') {
                inString = false;
            }
            continue;
        }

        if (char === '"') inString = true;
        else if (char === openChar) depth += 1;
        else if (char === closeChar) {
            depth -= 1;
            if (depth === 0) return text.slice(start, index + 1);
        }
    }

    return "";
}

function skipWhitespace(text, start) {
    let index = start;
    while (index < text.length && /\s/.test(text[index])) index += 1;
    return index;
}

function skipPlotlySeparator(text, start) {
    let index = skipWhitespace(text, start);
    if (text[index] === ",") index += 1;
    return skipWhitespace(text, index);
}

function normalizeSavedPlotLayout(layout = {}, data = [], options = {}) {
    const hasHeatmap = data.some((trace) => ["heatmap", "histogram2d", "contour"].includes(trace.type));
    const hasLegend = data.some((trace) => trace.showlegend !== false && trace.name);
    const preserveSize = Boolean(options.preserveSize);
    const next = {
        ...layout,
        autosize: !preserveSize,
        paper_bgcolor: "#fff",
        plot_bgcolor: layout.plot_bgcolor || "#f7f9fb",
        margin: {
            ...(layout.margin || {}),
            l: 58,
            r: hasHeatmap ? 112 : (hasLegend ? 150 : 36),
            t: layout.title ? 58 : 32,
            b: 54,
        },
        hoverlabel: {
            bgcolor: "#fff",
            bordercolor: "#ced4da",
            font: {color: "#212529", size: 12},
            ...(layout.hoverlabel || {}),
        },
    };

    if (!preserveSize) {
        delete next.width;
        delete next.height;
    }

    if (typeof next.title === "string") {
        next.title = {text: next.title};
    }
    if (next.title) {
        next.title = {
            ...next.title,
            font: {size: 17, color: "#2f343b", ...(next.title.font || {})},
            x: next.title.x ?? 0.02,
            xanchor: next.title.xanchor || "left",
        };
    } else if (options.title) {
        next.title = {
            text: options.title,
            font: {size: 17, color: "#2f343b"},
            x: 0.02,
            xanchor: "left",
        };
    }

    next.legend = {
        ...(layout.legend || {}),
        orientation: "v",
        x: 1.02,
        xanchor: "left",
        y: 1,
        yanchor: "top",
        bgcolor: "rgba(255,255,255,0.84)",
        bordercolor: "rgba(222,226,230,0.9)",
        borderwidth: hasLegend ? 1 : 0,
        font: {size: 11, color: "#495057"},
    };

    collectCartesianAxisNames(next, data).forEach((axisName) => {
        const axis = next[axisName] || {};
        next[axisName] = {
            ...axis,
            automargin: true,
            showgrid: true,
            gridcolor: axis.gridcolor || "#edf1f5",
            gridwidth: axis.gridwidth || 1,
            zeroline: axis.zeroline ?? false,
            tickfont: {size: 11, color: "#6c757d", ...(axis.tickfont || {})},
        };
    });

    return next;
}

function applySavedPlotSizing(target, layout, options = {}) {
    if (!options.preserveSize) {
        target.style.minWidth = "";
        target.style.minHeight = "";
        return;
    }

    const width = Number(layout.width);
    const height = Number(layout.height);
    target.style.minWidth = Number.isFinite(width) && width > 0 ? `${width}px` : "";
    target.style.minHeight = Number.isFinite(height) && height > 0 ? `${height}px` : "";
}

function collectCartesianAxisNames(layout, data) {
    const names = new Set();

    Object.keys(layout || {}).forEach((axisName) => {
        if (/^[xy]axis\d*$/.test(axisName)) names.add(axisName);
    });

    data.forEach((trace) => {
        if (!trace || (!("x" in trace) && !("y" in trace))) return;
        if ("x" in trace) names.add(plotlyAxisLayoutName(trace.xaxis, "x"));
        if ("y" in trace) names.add(plotlyAxisLayoutName(trace.yaxis, "y"));
    });

    return [...names];
}

function plotlyAxisLayoutName(axisRef, axis) {
    const value = axisRef || axis;
    if (value === axis) return `${axis}axis`;
    return `${axis}axis${String(value).slice(axis.length)}`;
}

export function wrapText(text, maxLen) {
    const words = String(text || "").replace(/\s+/g, " ").trim().split(" ").filter(Boolean);
    const lines = [];
    let currentLine = "";

    words.forEach((word) => {
        if (word.length > maxLen) {
            if (currentLine) {
                lines.push(currentLine.trim());
                currentLine = "";
            }
            for (let index = 0; index < word.length; index += maxLen) {
                lines.push(word.slice(index, index + maxLen));
            }
        } else if (`${currentLine} ${word}`.trim().length > maxLen && currentLine) {
            lines.push(currentLine.trim());
            currentLine = word;
        } else {
            currentLine = `${currentLine} ${word}`.trim();
        }
    });

    if (currentLine) lines.push(currentLine.trim());
    return lines.join("<br>");
}
