import {
    cleanupRoute, normalizeRoute, parseHash,
    persistSelectedModel, setActiveNav, setBodyClass, state,
} from "./core.js";
import { destroyChart } from "./chart.js";
import { renderHome } from "./page-home.js";
import { renderCorpus } from "./page-corpus.js";
import { renderGeography } from "./page-geography.js";
import { renderEmbeddingsAnalysis, displayPointInfo, triggerModelChange } from "./page-embeddings.js";
import { renderGraphPage } from "./page-graphs.js";

function render() {
    cleanupRoute();
    const scatterPlot = document.getElementById("scatter-plot");
    if (scatterPlot) destroyChart(scatterPlot);

    const parsed = parseHash();
    const path = normalizeRoute(parsed.path);
    setBodyClass(path);
    setActiveNav(path);

    if (path !== parsed.path) {
        window.location.hash = `#${path}`;
        return;
    }

    if (path === "/home") return renderHome();
    if (path === "/corpus") return renderCorpus();
    if (path === "/geography") return renderGeography();
    if (path === "/embeddings_analysis") return renderEmbeddingsAnalysis();
    const graphType = path.slice(1);
    if (["beings", "realms", "ages"].includes(graphType)) return renderGraphPage(graphType);

    window.location.hash = "#/corpus";
}

window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.type !== "openPointDetails") return;

    state.pendingPoint = {
        id: data.id,
        model: data.model,
        chunkIndex: data.chunkIndex,
    };

    const currentPath = normalizeRoute(parseHash().path);
    if (currentPath !== "/embeddings_analysis") {
        window.location.hash = "#/embeddings_analysis";
    } else {
        if (data.model) {
            const modelSelect = document.getElementById("global-model-select");
            if (modelSelect && Array.from(modelSelect.options).some((option) => option.value === data.model)) {
                modelSelect.value = data.model;
                triggerModelChange();
            }
        }
        displayPointInfo(data.id, data.chunkIndex);
        state.pendingPoint = null;
    }
});

window.addEventListener("hashchange", render);
window.addEventListener("DOMContentLoaded", render);
