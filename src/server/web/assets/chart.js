// Active chart backend. Change the import below and the CDN script in index.html:
//
//   "./chart-plotly.js"  — Plotly (~3.5 MB)
//     <script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>
//
//   "./chart-echarts.js" — ECharts (~1 MB)
//     <script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.6.0/echarts.min.js"></script>
//
//   "./chart-regl.js"    — regl-scatterplot (~30 KB, WebGL)
//     No CDN needed — loads as ES module from https://esm.sh/regl-scatterplot@1.16.0
//     Remove the Plotly/ECharts <script> from index.html.
//
export { destroyChart, resizeChart, renderScatter, renderHeatmap, renderDistribution } from "./chart-plotly.js";
