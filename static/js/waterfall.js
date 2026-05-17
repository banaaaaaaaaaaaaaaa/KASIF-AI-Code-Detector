const WATERFALL_CONFIDENCE_THRESHOLD = 0.70;
const WATERFALL_UI_HELP = {
    panelInfo:
        "This section shows the most important features that influenced the prediction. The larger the absolute SHAP value, the stronger that feature affected the result.",
    moreDetailsUrl: "/waterfall-details"
};

function waterfallEscapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function waterfallToUnitConfidence(value) {
    const num = Number(value || 0);
    if (!Number.isFinite(num)) return 0;
    return num > 1 ? num / 100 : num;
}

function formatWaterfallSignedShapValue(value) {
    const num = Number(value || 0);
    return `${num >= 0 ? "+" : ""}${num.toFixed(4)}`;
}

function renderWaterfall(features, confidence = 1, uiHelp = {}) {
    const waterfallContainer = document.getElementById("waterfallContainer");
    if (!waterfallContainer) return;

    renderWaterfallTopbar(uiHelp);

    const normalizedConfidence = waterfallToUnitConfidence(confidence);
    const isLowConfidence = normalizedConfidence < WATERFALL_CONFIDENCE_THRESHOLD;

    if (!Array.isArray(features) || features.length === 0) {
        waterfallContainer.innerHTML =
            '<div class="empty-state-box waterfall-empty">No SHAP waterfall available.</div>';
        return;
    }

    const sortedFeatures = features
        .slice()
        .sort((a, b) => Math.abs(Number(b.shap_value || 0)) - Math.abs(Number(a.shap_value || 0)));

    const topFeatures = sortedFeatures.slice(0, 10);

    const maxAbs = Math.max(
    ...topFeatures.map((item) => Math.abs(Number(item.shap_value || 0))),
    0
);

    waterfallContainer.innerHTML = topFeatures.map((item) => {
        const shapValue = Number(item.shap_value || 0);
        const width = maxAbs > 0 ? (Math.abs(shapValue) / maxAbs) * 100 : 0;
        const featureName =
            item.display_name || item.feature || item.feature_name || item.name || "-";
        const explanation =
            item.explanation || "Explanation will be added later.";

        let rowClass = "";
        if (isLowConfidence) {
            rowClass = shapValue >= 0 ? "lowconf-positive" : "lowconf-negative";
        } else {
            rowClass = shapValue >= 0 ? "ai" : "human";
        }

        return `
            <div class="waterfall-row ${rowClass}">
                <div class="waterfall-label-wrap">
                    <span
                        class="waterfall-label"
                        aria-label="${waterfallEscapeHtml(explanation)}"
                    >
                        ${waterfallEscapeHtml(featureName)}
                    </span>

                    <span
                        class="feature-help-icon"
                        data-tooltip="${waterfallEscapeHtml(explanation)}"
                        aria-label="${waterfallEscapeHtml(explanation)}"
                        tabindex="0"
                    >?</span>
                </div>

                <div class="waterfall-bar-track">
                    <div class="waterfall-bar-fill" style="width:${width.toFixed(1)}%"></div>
                </div>

                <div
                    class="waterfall-value"
                    aria-label="${waterfallEscapeHtml(explanation)}"
                >
                    ${formatWaterfallSignedShapValue(shapValue)}
                </div>
            </div>
        `;
    }).join("");
}

window.renderWaterfall = renderWaterfall;

function renderWaterfallTopbar(uiHelp = {}) {
    const waterfallTopbar = document.getElementById("waterfallTopbar");
    if (!waterfallTopbar) return;

    const panelInfo =
        uiHelp.panelInfo ||
        WATERFALL_UI_HELP.panelInfo;

    const moreDetailsUrl =
        uiHelp.moreDetailsUrl ||
        WATERFALL_UI_HELP.moreDetailsUrl;

    const waterfallPanelInfoIcon = document.getElementById("waterfallPanelInfoIcon");

    if (waterfallPanelInfoIcon) {
        waterfallPanelInfoIcon.setAttribute("data-tooltip", panelInfo);
        waterfallPanelInfoIcon.setAttribute("aria-label", panelInfo);
    }

    waterfallTopbar.innerHTML = `
        <div class="section-topbar">
            <a
                class="more-details-btn"
                href="${waterfallEscapeHtml(moreDetailsUrl)}"
            >More details</a>
        </div>
    `;
}

window.renderWaterfall = renderWaterfall;