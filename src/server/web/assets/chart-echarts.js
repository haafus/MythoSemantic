// ECharts chart implementation (~1 MB, 10-17x smaller than Plotly).
// CDN: <script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.6.0/echarts.min.js"></script>
// To activate: change the import in chart.js to "./chart-echarts.js" and replace the CDN script in index.html

import { escapeHtml, normalizePreviewText } from "./core.js";

let chartInstance = null;

export function destroyChart(el) {
    if (chartInstance) {
        chartInstance.dispose();
        chartInstance = null;
    }
    el.innerHTML = "";
    el.dataset.chart = "";
}

export function resizeChart(el) {
    if (chartInstance) {
        setTimeout(() => chartInstance.resize(), 100);
    }
}

function initChart(el) {
    if (chartInstance) {
        chartInstance.dispose();
    }
    chartInstance = echarts.init(el, null, { renderer: "canvas" });
    el.dataset.chart = "1";
    return chartInstance;
}

export async function renderScatter(el, data, { colorMap, onPointClick }) {
    const points = Array.isArray(data.points) ? data.points : [];
    if (!window.echarts || !points.length) throw new Error("Projection data is empty.");

    el.style.display = "block";

    const traditions = [...new Set(points.map((p) => p.tradition || "Unknown"))];
    const showLegend = el.clientWidth >= 720;
    const chart = initChart(el);

    const series = traditions.map((tradition) => {
        const pts = points.filter((p) => (p.tradition || "Unknown") === tradition);
        return {
            type: "scatter",
            name: tradition,
            data: pts.map((p) => ({
                value: [p.x, p.y],
                _id: p.id,
                _tradition: p.tradition,
                _chunk: p.chunk_index,
                _text: normalizePreviewText(p.text).substring(0, 220),
            })),
            itemStyle: {
                color: colorMap[tradition],
                opacity: 0.74,
                borderColor: "rgba(255,255,255,0.85)",
                borderWidth: 0.4,
            },
            symbolSize: 6,
            large: pts.length > 5000,
            largeThreshold: 5000,
            emphasis: {
                focus: "series",
                blurScope: "coordinateSystem",
            },
        };
    });

    chart.setOption({
        tooltip: {
            trigger: "item",
            confine: true,
            backgroundColor: "#fff",
            borderColor: "#ced4da",
            borderWidth: 1,
            textStyle: { color: "#212529", fontSize: 12 },
            extraCssText: "box-shadow:0 10px 28px rgba(15,23,42,.16);max-width:360px;line-height:1.45;overflow-wrap:anywhere;",
            formatter: (params) => {
                const d = params.data;
                return `<div style="font-weight:700;color:#111827;margin-bottom:4px">${escapeHtml(d._tradition || "Unknown")}</div>`
                    + `<div style="color:#6c757d;margin-bottom:7px">ID: ${escapeHtml(d._id || "")} | Chunk: ${escapeHtml(d._chunk ?? 0)}</div>`
                    + `<div style="color:#343a40">${escapeHtml(d._text || "No preview available.")}</div>`;
            },
        },
        legend: showLegend ? {
            type: "scroll",
            orient: "vertical",
            right: 10,
            top: 20,
            bottom: 20,
            backgroundColor: "rgba(255,255,255,0.84)",
            borderColor: "rgba(222,226,230,0.9)",
            borderWidth: 1,
            textStyle: { fontSize: 11, color: "#495057" },
            pageTextStyle: { color: "#495057" },
        } : { show: false },
        grid: {
            left: 50,
            right: showLegend ? 176 : 28,
            top: 28,
            bottom: 48,
        },
        xAxis: {
            type: "value",
            splitLine: { lineStyle: { color: "#edf1f5" } },
            axisLine: { show: false },
            axisTick: { show: false },
            axisLabel: { fontSize: 11, color: "#6c757d" },
        },
        yAxis: {
            type: "value",
            splitLine: { lineStyle: { color: "#edf1f5" } },
            axisLine: { show: false },
            axisTick: { show: false },
            axisLabel: { fontSize: 11, color: "#6c757d" },
        },
        dataZoom: [
            { type: "inside", xAxisIndex: 0, filterMode: "none" },
            { type: "inside", yAxisIndex: 0, filterMode: "none" },
        ],
        animation: true,
        animationDuration: 600,
        series,
    });

    chart.on("click", (params) => {
        if (params.componentType === "series" && params.data) {
            onPointClick(params.data._id, params.data._chunk);
        }
    });
}

const VIRIDIS = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"];

export async function renderHeatmap(el, data) {
    const traditions = data.traditions || [];
    const distances = data.distances || [];
    if (!window.echarts || !traditions.length) throw new Error("Heatmap data is empty.");

    el.style.display = "block";

    const chart = initChart(el);

    const heatmapData = [];
    let min = Infinity, max = -Infinity;
    for (let y = 0; y < distances.length; y++) {
        for (let x = 0; x < distances[y].length; x++) {
            const v = distances[y][x];
            heatmapData.push([x, y, v]);
            if (v < min) min = v;
            if (v > max) max = v;
        }
    }

    chart.setOption({
        tooltip: {
            trigger: "item",
            confine: true,
            backgroundColor: "#fff",
            borderColor: "#ced4da",
            borderWidth: 1,
            textStyle: { color: "#212529", fontSize: 12 },
            formatter: (params) => {
                const [xi, yi, val] = params.data;
                return `Distance between <b>${escapeHtml(traditions[xi])}</b> and <b>${escapeHtml(traditions[yi])}</b>: ${val.toFixed(4)}`;
            },
        },
        grid: { left: 120, right: 80, top: 40, bottom: 140 },
        xAxis: {
            type: "category",
            data: traditions,
            axisLabel: { rotate: 45, fontSize: 11, interval: 0 },
            splitArea: { show: true },
        },
        yAxis: {
            type: "category",
            data: traditions,
            axisLabel: { fontSize: 11, interval: 0 },
            splitArea: { show: true },
        },
        visualMap: {
            min,
            max,
            calculable: true,
            orient: "vertical",
            right: 10,
            bottom: 40,
            inRange: { color: VIRIDIS },
        },
        series: [{
            type: "heatmap",
            data: heatmapData,
            emphasis: {
                itemStyle: { borderColor: "#333", borderWidth: 1 },
            },
        }],
        animation: true,
    });
}

export async function renderDistribution(el, data, { colorMap }) {
    const traditions = data.traditions || [];
    if (!window.echarts || !traditions.length) throw new Error("Distribution data is empty.");

    el.style.display = "block";
    el.style.height = `${Math.max(500, 28 * traditions.length + 120)}px`;

    const chart = initChart(el);

    const names = traditions.map((t) => t.name);

    chart.setOption({
        tooltip: {
            trigger: "item",
            confine: true,
            backgroundColor: "#fff",
            borderColor: "#ced4da",
            borderWidth: 1,
            textStyle: { color: "#212529", fontSize: 12 },
            formatter: (params) => {
                const t = traditions[params.dataIndex];
                return `<b>${escapeHtml(t.name)}</b><br>Chunks: ${t.chunks.toLocaleString()}<br>Source texts: ${t.doc_count}`;
            },
        },
        grid: { left: 160, right: 140, top: 40, bottom: 60 },
        xAxis: {
            type: "value",
            name: "Number of chunks",
            nameLocation: "middle",
            nameGap: 30,
            axisLabel: { fontSize: 11 },
        },
        yAxis: {
            type: "category",
            data: names,
            inverse: true,
            axisLabel: { fontSize: 11 },
        },
        series: [{
            type: "bar",
            data: traditions.map((t) => ({
                value: t.chunks,
                itemStyle: { color: colorMap[t.name] },
            })),
            label: {
                show: true,
                position: "right",
                formatter: (params) => {
                    const t = traditions[params.dataIndex];
                    return `${t.chunks.toLocaleString()} chunks (${t.percentage}%)`;
                },
                fontSize: 11,
                color: "#495057",
            },
            barMaxWidth: 28,
        }],
        animation: true,
        animationDuration: 800,
    });
}
