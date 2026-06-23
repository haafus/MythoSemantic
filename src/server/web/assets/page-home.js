import { app } from "./core.js";

export function renderHome() {
    document.title = "MythoScope - Home";
    app.innerHTML = `
        <main class="home-page">
            <div class="header-container">
                <img src="/assets/logo.jpg" alt="MythoScope Logo" class="logo-image">

                <nav class="nav-menu">
                    <button class="nav-item active" type="button" data-tab="vision">Vision</button>
                    <span class="separator">|</span>
                    <button class="nav-item" type="button" data-tab="methodology">Methodology</button>
                    <span class="separator">|</span>
                    <button class="nav-item" type="button" data-tab="contribute">Contribute</button>
                    <span class="separator">|</span>
                    <button class="nav-item" type="button" data-tab="resources">Resources</button>
                </nav>
            </div>

            <div class="content-container">
                <div id="vision" class="tab-content active">
                    <p>The first large-scale infrastructure for comparative analysis of mythology, religion, and ancient literature &mdash; an international collaborative project integrating classical interpretive methods with artificial intelligence to investigate shared origins and deep structural patterns of human culture.</p>
                    <p>Mythoscope is an interdisciplinary research initiative and open analytical platform dedicated to the large-scale comparative study of mythology, ancient religions, and cultural texts. Integrating classical humanities methodologies with computational approaches, the project enables scholars to explore deep semantic structures, trace cultural patterns across traditions, and investigate the historical evolution of symbolic systems.</p>
                    <p><strong>Toward a Computational Framework for Comparative Mythology.</strong> The framework enables large-scale, cross-cultural, reproducible analysis, combining unsupervised (bottom-up, continuous) and supervised (top-down, discrete) methods to provide a foundation for future work in computational mythology and digital humanities.</p>
                </div>
                <div id="methodology" class="tab-content">insert your text</div>
                <div id="contribute" class="tab-content">insert your text</div>
                <div id="resources" class="tab-content">insert your text</div>
            </div>
        </main>
    `;

    app.querySelectorAll(".nav-item").forEach((button) => {
        button.addEventListener("click", () => {
            app.querySelectorAll(".tab-content").forEach((content) => content.classList.remove("active"));
            app.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
            const target = document.getElementById(button.dataset.tab);
            if (target) target.classList.add("active");
            button.classList.add("active");
        });
    });
}
