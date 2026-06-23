// Active chart backend. Change the import to switch:
//   "./chart-plotly.js"  — Plotly (~3.5 MB CDN)
//   "./chart-echarts.js" — ECharts (~1 MB CDN)
//   "./chart-regl.js"    — regl-scatterplot (~30 KB, no CDN needed)
export { destroyChart, resizeChart, renderScatter, renderHeatmap, renderDistribution } from "./chart-plotly.js";
