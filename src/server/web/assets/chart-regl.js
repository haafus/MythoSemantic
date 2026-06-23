
import { escapeHtml, normalizePreviewText } from "./core.js";

let scatterInstance = null;
let pointMeta = null;
let cleanupFn = null;

export function destroyChart(el) {
    if (scatterInstance) {
        scatterInstance.destroy();
        scatterInstance = null;
    }
    if (cleanupFn) {
        cleanupFn();
        cleanupFn = null;
    }
    pointMeta = null;
    el.innerHTML = "";
    el.dataset.chart = "";
}

export function resizeChart(el) {
    if (scatterInstance) {
        setTimeout(() => {
            const canvas = el.querySelector("canvas");
            if (canvas) {
                scatterInstance.set({ width: canvas.clientWidth, height: canvas.clientHeight });
            }
        }, 100);
    }
}

// --- Utilities ---

function hexToRgba(hex, a = 1) {
    const c = hex.replace("#", "");
    return [parseInt(c.slice(0, 2), 16) / 255, parseInt(c.slice(2, 4), 16) / 255, parseInt(c.slice(4, 6), 16) / 255, a];
}

function minMax(arr) {
    let lo = Infinity, hi = -Infinity;
    for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
    return [lo, hi];
}

const VIRIDIS = [
    [68, 1, 84], [72, 40, 120], [62, 73, 137], [49, 104, 142],
    [38, 130, 142], [31, 158, 137], [53, 183, 121], [110, 206, 88],
    [181, 222, 43], [253, 231, 37],
];

function viridisColor(t) {
    t = Math.max(0, Math.min(1, t));
    const idx = t * (VIRIDIS.length - 1);
    const i = Math.min(Math.floor(idx), VIRIDIS.length - 2);
    const f = idx - i;
    const a = VIRIDIS[i], b = VIRIDIS[i + 1];
    return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(a[2] + (b[2] - a[2]) * f)})`;
}

function positionTooltip(container, tooltip, event) {
    const rect = container.getBoundingClientRect();
    const cx = event?.clientX ?? (rect.left + rect.width / 2);
    const cy = event?.clientY ?? (rect.top + rect.height / 2);
    const gap = 14, pad = 10;
    tooltip.style.left = "0px";
    tooltip.style.top = "0px";
    const w = tooltip.offsetWidth, h = tooltip.offsetHeight;
    let left = cx - rect.left + gap, top = cy - rect.top + gap;
    if (left + w + pad > rect.width) left = cx - rect.left - w - gap;
    if (top + h + pad > rect.height) top = cy - rect.top - h - gap;
    tooltip.style.left = `${Math.max(pad, Math.min(left, rect.width - w - pad))}px`;
    tooltip.style.top = `${Math.max(pad, Math.min(top, rect.height - h - pad))}px`;
}

function getTooltip() {
    const plotCanvas = document.getElementById("plotCanvas");
    if (!plotCanvas) return [null, null];
    let tip = plotCanvas.querySelector(".plot-hover-tooltip");
    if (!tip) {
        tip = document.createElement("div");
        tip.className = "plot-hover-tooltip";
        plotCanvas.appendChild(tip);
    }
    return [plotCanvas, tip];
}

function createHiDpiCanvas(el) {
    const dpr = window.devicePixelRatio || 1;
    const w = el.clientWidth, h = el.clientHeight;
    const canvas = document.createElement("canvas");
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    el.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    return { canvas, ctx, w, h };
}

function bindCanvasTooltip(canvas, hitTest) {
    const [plotCanvas, tooltip] = getTooltip();
    if (!tooltip || !plotCanvas) return () => {};
    const onMove = (e) => {
        const rect = canvas.getBoundingClientRect();
        const html = hitTest(e.clientX - rect.left, e.clientY - rect.top);
        if (html) {
            tooltip.innerHTML = html;
            tooltip.classList.add("visible");
            positionTooltip(plotCanvas, tooltip, e);
        } else {
            tooltip.classList.remove("visible");
        }
    };
    const onLeave = () => tooltip.classList.remove("visible");
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);
    return () => { canvas.removeEventListener("mousemove", onMove); canvas.removeEventListener("mouseleave", onLeave); };
}

function truncate(s, n) {
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

// --- Scatter (regl-scatterplot, WebGL) ---

let _createScatterplot = null;

export async function renderScatter(el, data, { colorMap, onPointClick }) {
    const points = Array.isArray(data.points) ? data.points : [];
    if (!points.length) throw new Error("Projection data is empty.");

    el.style.display = "block";

    if (!_createScatterplot) {
        const mod = await import("https://esm.sh/regl-scatterplot@1.16.0");
        _createScatterplot = mod.default;
    }

    const traditions = [...new Set(points.map((p) => p.tradition || "Unknown"))];
    const tradIdx = Object.create(null);
    traditions.forEach((t, i) => { tradIdx[t] = i; });

    const colors = traditions.map((t) => hexToRgba(colorMap[t] || "#888888", 0.85));

    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const [xMin, xMax] = minMax(xs);
    const [yMin, yMax] = minMax(ys);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const pad = 0.05;

    const ptData = points.map((p) => [
        ((p.x - xMin) / xRange) * (2 - 2 * pad) - (1 - pad),
        ((p.y - yMin) / yRange) * (2 - 2 * pad) - (1 - pad),
        tradIdx[p.tradition || "Unknown"],
    ]);

    pointMeta = points.map((p) => ({
        id: p.id,
        tradition: p.tradition,
        chunk_index: p.chunk_index,
        text: normalizePreviewText(p.text).substring(0, 220),
    }));

    const canvas = document.createElement("canvas");
    const showLegend = el.clientWidth >= 720;
    canvas.style.width = showLegend ? "calc(100% - 160px)" : "100%";
    canvas.style.height = "100%";
    el.appendChild(canvas);

    if (showLegend) {
        const legend = document.createElement("div");
        legend.style.cssText = "position:absolute;right:8px;top:20px;bottom:20px;width:148px;overflow-y:auto;"
            + "background:rgba(255,255,255,.84);border:1px solid rgba(222,226,230,.9);border-radius:4px;padding:8px;"
            + "font-size:11px;color:#495057;";
        legend.innerHTML = traditions.map((t) =>
            `<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">`
            + `<span style="display:inline-block;width:12px;height:12px;border-radius:2px;flex-shrink:0;background:${escapeHtml(colorMap[t] || "#888")}"></span>`
            + `<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(t)}</span></div>`
        ).join("");
        el.style.position = "relative";
        el.appendChild(legend);
    }

    scatterInstance = _createScatterplot({
        canvas,
        width: canvas.clientWidth,
        height: canvas.clientHeight,
        pointSize: 6,
        pointColor: colors,
        colorBy: "valueA",
        backgroundColor: [0.984, 0.988, 0.992, 1],
        showReticle: true,
        reticleColor: [0.4, 0.4, 0.4, 0.8],
        opacityInactiveScale: 0.33,
    });

    await scatterInstance.draw(ptData);

    const [plotCanvas, tooltip] = getTooltip();

    scatterInstance.subscribe("pointOver", (idx) => {
        if (!tooltip || !pointMeta[idx]) return;
        const m = pointMeta[idx];
        tooltip.innerHTML = `
            <div class="plot-hover-title">${escapeHtml(m.tradition || "Unknown")}</div>
            <div class="plot-hover-meta">ID: ${escapeHtml(m.id || "")} | Chunk: ${escapeHtml(m.chunk_index ?? 0)}</div>
            <div class="plot-hover-text">${escapeHtml(m.text || "No preview available.")}</div>`;
        tooltip.classList.add("visible");
        const [sx, sy] = scatterInstance.getScreenPosition(idx);
        const cr = canvas.getBoundingClientRect();
        positionTooltip(plotCanvas, tooltip, { clientX: cr.left + sx, clientY: cr.top + sy });
    });

    scatterInstance.subscribe("pointOut", () => {
        if (tooltip) tooltip.classList.remove("visible");
    });

    scatterInstance.subscribe("select", ({ points: sel }) => {
        if (sel.length === 1 && pointMeta[sel[0]]) {
            const m = pointMeta[sel[0]];
            scatterInstance.deselect({ preventEvent: true });
            onPointClick(m.id, m.chunk_index);
        }
    });

    el.dataset.chart = "1";
}

// --- Heatmap (Canvas fallback) ---

export async function renderHeatmap(el, data) {
    const traditions = data.traditions || [];
    const distances = data.distances || [];
    if (!traditions.length) throw new Error("Heatmap data is empty.");

    el.style.display = "block";

    const { canvas, ctx, w, h } = createHiDpiCanvas(el);

    const ml = 120, mr = 60, mt = 10, mb = 120;
    const pw = w - ml - mr, ph = h - mt - mb;
    const n = traditions.length;
    const cw = pw / n, ch = ph / n;

    let min = Infinity, max = -Infinity;
    for (const row of distances) for (const v of row) { if (v < min) min = v; if (v > max) max = v; }
    const range = max - min || 1;

    for (let yi = 0; yi < n; yi++) {
        for (let xi = 0; xi < n; xi++) {
            ctx.fillStyle = viridisColor((distances[yi][xi] - min) / range);
            ctx.fillRect(ml + xi * cw, mt + yi * ch, cw + 0.5, ch + 0.5);
        }
    }

    ctx.fillStyle = "#495057";
    ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i < n; i++) {
        ctx.fillText(truncate(traditions[i], 16), ml - 6, mt + i * ch + ch / 2);
    }
    for (let i = 0; i < n; i++) {
        ctx.save();
        ctx.translate(ml + i * cw + cw / 2, mt + ph + 6);
        ctx.rotate(Math.PI / 4);
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText(truncate(traditions[i], 16), 2, 0);
        ctx.restore();
    }

    cleanupFn = bindCanvasTooltip(canvas, (mx, my) => {
        const xi = Math.floor((mx - ml) / cw);
        const yi = Math.floor((my - mt) / ch);
        if (xi >= 0 && xi < n && yi >= 0 && yi < n) {
            return `Distance between <b>${escapeHtml(traditions[xi])}</b> and <b>${escapeHtml(traditions[yi])}</b>: ${distances[yi][xi].toFixed(4)}`;
        }
        return null;
    });

    el.dataset.chart = "1";
}

// --- Distribution (Canvas fallback) ---

export async function renderDistribution(el, data, { colorMap }) {
    const traditions = data.traditions || [];
    if (!traditions.length) throw new Error("Distribution data is empty.");

    el.style.display = "block";
    el.style.height = `${Math.max(500, 28 * traditions.length + 120)}px`;

    const { canvas, ctx, w, h } = createHiDpiCanvas(el);

    const ml = 160, mr = 140, mt = 30, mb = 50;
    const pw = w - ml - mr, ph = h - mt - mb;
    const n = traditions.length;
    const totalGap = Math.max(0, (n - 1) * 4);
    const barH = Math.min(24, (ph - totalGap) / n);
    const gap = n > 1 ? Math.min(4, (ph - n * barH) / (n - 1)) : 0;
    const maxVal = traditions.reduce((m, t) => Math.max(m, t.chunks), 0) || 1;

    ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";

    for (let i = 0; i < n; i++) {
        const t = traditions[i];
        const y = mt + i * (barH + gap);
        const bw = (t.chunks / maxVal) * pw;

        ctx.fillStyle = colorMap[t.name] || "#888";
        ctx.fillRect(ml, y, bw, barH);

        ctx.fillStyle = "#495057";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(truncate(t.name, 22), ml - 8, y + barH / 2);

        ctx.textAlign = "left";
        ctx.fillText(`${t.chunks.toLocaleString()} chunks (${t.percentage}%)`, ml + bw + 8, y + barH / 2);
    }

    ctx.strokeStyle = "#dee2e6";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(ml, mt + n * (barH + gap) - gap);
    ctx.lineTo(ml + pw, mt + n * (barH + gap) - gap);
    ctx.stroke();

    ctx.fillStyle = "#6c757d";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("Number of chunks", ml + pw / 2, mt + n * (barH + gap) - gap + 12);

    cleanupFn = bindCanvasTooltip(canvas, (mx, my) => {
        for (let i = 0; i < n; i++) {
            const y = mt + i * (barH + gap);
            if (my >= y && my < y + barH && mx >= ml) {
                const t = traditions[i];
                return `<b>${escapeHtml(t.name)}</b><br>Chunks: ${t.chunks.toLocaleString()}<br>Source texts: ${t.doc_count}`;
            }
        }
        return null;
    });

    el.dataset.chart = "1";
}
