import { app, state, api, escapeHtml, escapeAttribute, loadTraditionInfo } from "./core.js";

export async function renderGeography() {
    document.title = "MythoScope - Geography";
    app.innerHTML = `
        <main class="geography-page container">
            <div class="map-frame">
                <div id="geography-map"></div>
            </div>
        </main>
    `;

    if (typeof L === "undefined") {
        showGeographyError("Map library could not be loaded.");
        return;
    }

    try {
        const traditions = await fetchTraditions();
        initializeGeographyMap(traditions);
    } catch (error) {
        console.error(error);
        showGeographyError("Could not load geography data.");
    }
}

function isValidColor(value) {
    return typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value.trim());
}

function normalizeCoordinates(value) {
    if (!Array.isArray(value) || value.length < 2) return null;

    const lat = Number(value[0]);
    const lon = Number(value[1]);

    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;

    return [lat, lon];
}

function normalizeTraditions(raw) {
    return Object.entries(raw || {})
        .map(([name, info]) => {
            const coordinates = normalizeCoordinates(info && info.coordinates);
            if (!coordinates) return null;

            return {
                name,
                description: info.description || "",
                coordinates,
                color: isValidColor(info.color) ? info.color : "#334155",
                books: Array.isArray(info.books) ? info.books.filter(Boolean) : [],
            };
        })
        .filter(Boolean)
        .sort((a, b) => a.name.localeCompare(b.name));
}

async function fetchTraditions() {
    const raw = await loadTraditionInfo();
    return normalizeTraditions(raw);
}

function buildCoordinateGroups(traditions) {
    const groups = new Map();

    traditions.forEach((item) => {
        const [lat, lon] = item.coordinates;
        const key = `${lat.toFixed(4)},${lon.toFixed(4)}`;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(item);
    });

    return groups;
}

function getOffsetCoordinate(item, index, total) {
    if (total <= 1) return item.coordinates;

    const [lat, lon] = item.coordinates;
    const angle = (Math.PI * 2 * index) / total;
    const radius = 0.8;

    return [
        lat + Math.sin(angle) * radius,
        lon + Math.cos(angle) * radius,
    ];
}

function buildPopupHtml(item) {
    const books = item.books.length
        ? item.books.map((book) => `<li>${escapeHtml(book)}</li>`).join("")
        : "<li>No books listed</li>";

    return `
        <div class="popup-title">
            <span class="popup-color" style="background:${escapeAttribute(item.color)}"></span>
            <span>${escapeHtml(item.name)}</span>
        </div>
        <div class="popup-description">${escapeHtml(item.description)}</div>
        <div class="popup-books-title">Books</div>
        <ul class="popup-books">${books}</ul>
    `;
}

function createMarkerIcon(item) {
    return L.divIcon({
        className: "tradition-marker",
        html: `<button class="map-point" type="button" style="--point-color:${escapeAttribute(item.color)}" aria-label="${escapeAttribute(item.name)}"></button>`,
        iconSize: [18, 18],
        iconAnchor: [9, 9],
        popupAnchor: [0, -11],
    });
}

function renderMarkers(map, traditions) {
    const bounds = [];
    const groups = buildCoordinateGroups(traditions);

    groups.forEach((group) => {
        group.forEach((item, index) => {
            const position = getOffsetCoordinate(item, index, group.length);

            L.marker(position, {
                icon: createMarkerIcon(item),
                keyboard: true,
                title: item.name,
            })
                .addTo(map)
                .bindPopup(buildPopupHtml(item), {
                    className: "tradition-popup",
                    closeButton: true,
                    maxWidth: 340,
                });

            bounds.push(position);
        });
    });

    if (bounds.length > 0) {
        map.fitBounds(bounds, {padding: [34, 34], maxZoom: 4});
    }
}

function initializeGeographyMap(traditions) {
    const worldBounds = [
        [-90, -240],
        [90, 240],
    ];

    state.geographyMap = L.map("geography-map", {
        zoomControl: true,
        maxBounds: worldBounds,
        maxBoundsViscosity: 1.0,
    }).setView([20, 15], 2);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 7,
        minZoom: 2,
        noWrap: true,
        bounds: worldBounds,
        attribution: "&copy; OpenStreetMap contributors",
    }).addTo(state.geographyMap);

    renderMarkers(state.geographyMap, traditions);
}

function showGeographyError(message) {
    const map = document.getElementById("geography-map");
    if (map) map.innerHTML = `<div class="map-error">${escapeHtml(message)}</div>`;
}
