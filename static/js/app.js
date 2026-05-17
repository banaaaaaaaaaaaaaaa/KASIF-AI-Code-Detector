const categoryInputs = document.querySelectorAll('input[name="category"]');
const codeInput = document.getElementById("codeInput");
const codeFileInput = document.getElementById("codeFileInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const codeOverlay = document.getElementById("codeOverlay");

const predictionValue = document.getElementById("predictionValue");
const messageBox = document.getElementById("messageBox");
const groupSummaryContainer = document.getElementById("groupSummaryContainer");
const unmappedFeaturesNote = document.getElementById("unmappedFeaturesNote");
const aiProbBar = document.getElementById("aiProbBar");
const humanProbBar = document.getElementById("humanProbBar");
const aiProbValue = document.getElementById("aiProbValue");
const humanProbValue = document.getElementById("humanProbValue");
const groupSummaryTopbar = document.getElementById("groupSummaryTopbar");
let lastAnalysisData = null;
const KASIF_SESSION_KEYS = [
    "kasif_last_analysis",
    "kasif_top_features",
    "kasif_grouped_influences",
    "kasif_prediction_label",
    "kasif_prediction_confidence",
    "kasif_saved_code",
    "kasif_saved_category",
    "kasif_return_from_details"
];

function clearKasifSession() {
    KASIF_SESSION_KEYS.forEach((key) => sessionStorage.removeItem(key));
}

function clearOldLocalStorage() {
    KASIF_SESSION_KEYS.forEach((key) => localStorage.removeItem(key));
}

function getNavigationType() {
    const nav = performance.getEntriesByType("navigation")[0];
    return nav ? nav.type : "navigate";
}

function saveAnalysisForDetails() {
    if (!lastAnalysisData) return;

    const displayLabel = buildUiDecisionFromResult(lastAnalysisData).label;

    const { ai, human } = normalizeProbabilities(
        lastAnalysisData.probabilities,
        lastAnalysisData.label,
        lastAnalysisData.confidence
    );

    const displayConfidence = Math.max(ai, human) / 100;

    const savedAnalysis = {
        data: lastAnalysisData,
        displayLabel,
        displayConfidence
    };

    sessionStorage.setItem("kasif_last_analysis", JSON.stringify(savedAnalysis));
    sessionStorage.setItem("kasif_top_features", JSON.stringify((lastAnalysisData.top_features || []).slice(0, 30)));
    sessionStorage.setItem("kasif_grouped_influences", JSON.stringify(lastAnalysisData.grouped_influences || []));
    sessionStorage.setItem("kasif_prediction_label", displayLabel || "");
    sessionStorage.setItem("kasif_prediction_confidence", String(displayConfidence || 0));
    sessionStorage.setItem("kasif_saved_code", codeInput.value || "");
    sessionStorage.setItem("kasif_saved_category", getSelectedCategory());
    sessionStorage.setItem("kasif_return_from_details", "1");
}

const CONTEXT_HELP = {
     "assignment": {
         title: "Assignment: CodeChef and CodeBench",
        image: "/static/images/context_samples/codebench_sample.png",
        level: "Beginner to Intermediate",
        use: "This context is for homework-style coding tasks and practice assignments.Use this when the code comes from homework, take-home tasks, or regular programming practice."
  
    },
    "labs-aybu": {
        title: "Lab: AYBU Labs",
        level: "Beginner to Intermediate",
        image: "/static/images/context_samples/aybu_labs_sample.png",
        use: "This context is for guided lab work and practical classroom exercises.Use this when the code comes from lab sessions, supervised practice, or structured exercises."
    },
    "exams-aybu": {
        title: "Exam: AYBU Exams",
        image: "/static/images/context_samples/aybu_exams_sample.png",
        level: "Intermediate to Advanced",
        use: "This context is for official AYBU exam-style solutions, usually written under time pressure. Use this when the code comes from AYBU exams, quizzes, or timed academic assessments."
    },

};

const CONFIDENCE_THRESHOLD = 0.70;



let currentLineFeatureMap = {};
let currentPredictedLabel = null;

function toUnitConfidence(value) {
    const num = Number(value || 0);
    if (!Number.isFinite(num)) return 0;
    return num > 1 ? num / 100 : num;
}

function getDisplayPrediction(probabilities, label, confidence) {
    const { ai, human } = normalizeProbabilities(probabilities, label, confidence);
    const maxProbPercent = Math.max(ai, human);

    if (maxProbPercent < CONFIDENCE_THRESHOLD * 100) {
        return "Mixed";
    }

    return human >= ai ? "Human" : "AI-generated";
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function getSelectedCategory() {
    const checked = document.querySelector('input[name="category"]:checked');
    return checked ? checked.value : "assignment";
}

function setMessage(text, type) {
    if (!messageBox) return;

    if (!text) {
        messageBox.textContent = "";
        messageBox.className = "message-box hidden";
        return;
    }

    messageBox.textContent = text;
    messageBox.className = `message-box is-${type}`;
}

function resetPredictionStyles() {
    if (predictionValue) {
        predictionValue.classList.remove("is-ai", "is-human", "is-mixed");
    }
}

function renderSummary(label) {
    if (!predictionValue) return;

    predictionValue.textContent = label || "-";
    resetPredictionStyles();

    if (label === "AI-generated") {
        predictionValue.classList.add("is-ai");
    } else if (label === "Mixed") {
        predictionValue.classList.add("is-mixed");
    } else if (label && label !== "N/A" && label !== "-") {
        predictionValue.classList.add("is-human");
    }
}

function normalizeProbabilities(probabilities, label, confidence) {
    let ai = null;
    let human = null;

    if (probabilities && typeof probabilities === "object" && !Array.isArray(probabilities)) {
        ai = Number(
            probabilities["AI-generated"] ??
            probabilities["AI"] ??
            probabilities.ai ??
            probabilities.AI ??
            probabilities.ai_probability ??
            probabilities.ai_generated ??
            probabilities[1]
        );

        human = Number(
            probabilities["Human-written"] ??
            probabilities["Human"] ??
            probabilities.human ??
            probabilities.Human ??
            probabilities.human_probability ??
            probabilities[0]
        );
    }

    if (Array.isArray(probabilities) && probabilities.length >= 2) {
        human = Number(probabilities[0]);
        ai = Number(probabilities[1]);
    }

    if (!Number.isFinite(ai) || !Number.isFinite(human)) {
        const conf = Number(confidence || 0);

        if (label === "AI-generated") {
            ai = conf;
            human = 100 - conf;
        } else {
            human = conf;
            ai = 100 - conf;
        }
    }

    if (ai <= 1 && human <= 1) {
        ai *= 100;
        human *= 100;
    }

    ai = Math.max(0, Math.min(ai, 100));
    human = Math.max(0, Math.min(human, 100));

    return { ai, human };
}

function renderProbabilities(probabilities, label, confidence) {
    if (!aiProbBar || !humanProbBar || !aiProbValue || !humanProbValue) return;

    const { ai, human } = normalizeProbabilities(probabilities, label, confidence);

    aiProbBar.style.width = `${ai.toFixed(1)}%`;
    humanProbBar.style.width = `${human.toFixed(1)}%`;

    aiProbValue.textContent = `${ai.toFixed(1)}%`;
    humanProbValue.textContent = `${human.toFixed(1)}%`;
}

function renderGroupSummary(groups, uiHelp = {}) {
    if (!groupSummaryContainer) return;

    const panelInfo =
        uiHelp.group_panel_info ||
        "This section shows how much each feature group contributed among the displayed important features.";

    const moreDetailsUrl =
        uiHelp.group_more_details_url || "#";

    const groupPanelInfoIcon = document.getElementById("groupPanelInfoIcon");

    if (groupPanelInfoIcon) {
        groupPanelInfoIcon.setAttribute("data-tooltip", panelInfo);
        groupPanelInfoIcon.setAttribute("aria-label", panelInfo);
    }

    if (groupSummaryTopbar) {
        groupSummaryTopbar.innerHTML = `
            <div class="group-summary-topbar">
                <a
                    class="group-more-details-btn"
                    href="${escapeHtml(moreDetailsUrl)}"
                >More details</a>
            </div>
        `;
    }

    if (!Array.isArray(groups) || groups.length === 0) {
        groupSummaryContainer.innerHTML = `
            <div class="empty-state-box">No group summary available.</div>
        `;
        return;
    }

    const total = groups.reduce((sum, item) => {
        const value = Number(item.group_support_score || 0);
        return sum + Math.max(value, 0);
    }, 0);

    const cardsHtml = groups.map((item) => {
        const score = Number(item.group_support_score || 0);
        const percent = total > 0 ? (score / total) * 100 : 0;
        const safePercent = Math.max(0, Math.min(percent, 100));

        const groupExplanation =
            item.explanation || "Explanation will be added later.";

        return `
            <div
                class="group-card"
                aria-label="${escapeHtml(groupExplanation)}"
            >
                <div class="group-card-header">
                    <div class="group-name-wrap">
                        <div class="group-name">${escapeHtml(item.group || "-")}</div>

                        <span
                            class="group-info-icon small"
                            data-tooltip="${escapeHtml(groupExplanation)}"
                            aria-label="${escapeHtml(groupExplanation)}"
                            tabindex="0"
                        >ⓘ</span>
                    </div>

                    <div class="group-percent">${safePercent.toFixed(1)}%</div>
                </div>

                <div class="group-bar-track">
                    <div
                        class="group-bar-fill"
                        style="width:${safePercent.toFixed(1)}%"
                    ></div>
                </div>

                <div class="group-meta">
                    <span>${Number(item.count || 0)} features</span>
              
                </div>
            </div>
        `;
    }).join("");

    groupSummaryContainer.innerHTML = `
        <div class="group-summary-grid">
            ${cardsHtml}
        </div>
    `;
}

function renderUnmappedFeatures(unmappedFeatures) {
    if (!unmappedFeaturesNote) return;

    if (Array.isArray(unmappedFeatures) && unmappedFeatures.length > 0) {
        const shown = unmappedFeatures.slice(0, 10).join(", ");

        unmappedFeaturesNote.textContent =
            "Some global features could not be mapped exactly to lines: " + shown;

        unmappedFeaturesNote.classList.remove("hidden");
    } else {
        unmappedFeaturesNote.textContent = "";
        unmappedFeaturesNote.classList.add("hidden");
    }
}

function getLineClass(lineNumber, lineFeatureMap) {
    const items = lineFeatureMap?.[lineNumber] || [];
    if (!items.length) return "";

    const total = items.reduce((sum, item) => {
        return sum + Number(item.shap_value || 0);
    }, 0);

    if (total > 0) return "is-ai";
    if (total < 0) return "is-human";

    return "";
}

function formatSignedValue(value) {
    const num = Number(value || 0);
    return `${num >= 0 ? "+" : ""}${num.toFixed(4)}`;
}

function getInlineReasonsHtml(lineNumber) {
    const items = currentLineFeatureMap[lineNumber] || [];
    if (!items.length) return "";

    const topItems = items
        .slice()
        .sort((a, b) => {
            return Math.abs(Number(b.shap_value || 0)) - Math.abs(Number(a.shap_value || 0));
        })
        .slice(0, 2);

    return `
        <div class="editor-line-badges">
            ${topItems.map((item) => {
                const shapNum = Number(item.shap_value || 0);

                const shapClass =
                    shapNum > 0 ? "is-positive" :
                    shapNum < 0 ? "is-negative" :
                    "is-neutral";

                return `
                    <span class="editor-reason-badge compact">
                        <span class="editor-reason-feature">
                            ${escapeHtml(item.feature || "feature")}
                        </span>

                       
                    </span>
                `;
            }).join("")}
        </div>
    `;
}

function renderEditor() {
    if (!codeInput || !codeOverlay) return;

    const code = codeInput.value || "";

    if (!code) {
        codeOverlay.innerHTML = '<div class="editor-empty">Paste Python code here...</div>';
        return;
    }

    const lines = code.split("\n");

    const html = lines.map((line, index) => {
        const lineNumber = index + 1;
        const safeLine = line.length ? escapeHtml(line) : "&nbsp;";
        const lineClass = getLineClass(lineNumber, currentLineFeatureMap);
        const reasonsHtml = getInlineReasonsHtml(lineNumber);

        return `
            <div class="editor-line ${lineClass}">
                <div class="editor-line-number">${lineNumber}</div>

                <div class="editor-line-main">
                    <div class="editor-line-row">
                        <div class="editor-line-code">${safeLine}</div>
                        ${reasonsHtml}
                    </div>
                </div>
            </div>
        `;
    }).join("");

    codeOverlay.innerHTML = html;
    syncEditorScroll();
}

function syncEditorScroll() {
    if (!codeInput || !codeOverlay) return;

    codeOverlay.scrollTop = codeInput.scrollTop;
    codeOverlay.scrollLeft = codeInput.scrollLeft;
}

function clearHighlightsOnly() {
    currentLineFeatureMap = {};
    currentPredictedLabel = null;

    renderEditor();
    renderUnmappedFeatures([]);
}

function attachContextHoverHelp() {
    const contextInputs = document.querySelectorAll('input[name="category"]');

    contextInputs.forEach((input) => {
        const info = CONTEXT_HELP[input.value];
        if (!info) return;

        const label = input.closest("label");
        if (!label) return;

        label.querySelectorAll(".context-tooltip-box").forEach((el) => el.remove());

        label.classList.add("context-option-pill");

        const tooltip = document.createElement("div");
        tooltip.className = "context-tooltip-box";

        tooltip.innerHTML = `
            <div class="context-tooltip-title">
                ${escapeHtml(info.title)}
            </div>

            <img
                class="context-code-image"
                src="${escapeHtml(info.image)}"
                alt="${escapeHtml(info.title)} code sample"
            />

            <div class="context-tooltip-line">
                <strong>Level:</strong> ${escapeHtml(info.level)}
            </div>

            <div class="context-tooltip-line">
                <strong>Use:</strong> ${escapeHtml(info.use)}
            </div>
        `;

        label.appendChild(tooltip);
    });
}

async function runAnalysis() {
    const code = codeInput.value.trim();
    const assessmentType = getSelectedCategory();

    if (!code) {
        setMessage("Please enter code first.", "error");
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Processing...";

    setMessage("", "");
    renderGroupSummary([]);
    renderWaterfall([]);
    renderUnmappedFeatures([]);

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                code: code,
                assessment_type: assessmentType,
            }),
        });

        const data = await response.json();
        lastAnalysisData = data;

        if (!response.ok || !data.success) {
            setMessage(data.error || "Something went wrong.", "error");
            return;
        }

        const displayLabel = buildUiDecisionFromResult(data).label;

        const { ai, human } = normalizeProbabilities(
            data.probabilities,
            data.label,
            data.confidence
        );

        const displayConfidence = Math.max(ai, human) / 100;

        updatePredictionPanel(data);
        renderSummary(displayLabel);
        renderProbabilities(data.probabilities, data.label, data.confidence);

        renderGroupSummary(
            data.grouped_influences || [],
            data.ui_help || {}
        );

        renderWaterfall(
            data.top_features || [],
            displayConfidence,
            data.ui_help?.waterfall || {}
        );

        currentLineFeatureMap = {};

        Object.entries(data.line_to_features || {}).forEach(([lineNumber, featureList]) => {
            currentLineFeatureMap[Number(lineNumber)] = featureList;
        });

        currentPredictedLabel = displayLabel || null;

        renderEditor();
        renderUnmappedFeatures(data.unmapped_features || []);

        setMessage("Analysis completed successfully.", "success");
    } catch (error) {
        setMessage("Request failed: " + error.message, "error");
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Run Analysis";
    }
}

categoryInputs.forEach((input) => {
    input.addEventListener("change", () => {
        clearHighlightsOnly();
    });
});

codeFileInput.addEventListener("change", async (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;

    try {
        const text = await file.text();

        codeInput.value = text;

     
        clearHighlightsOnly();
        renderEditor();

        setMessage(`Loaded file: ${file.name}`, "success");
    } catch (error) {
        setMessage("Could not read the selected file.", "error");
    }
});


function readStoredJson(key, fallback = null) {
    try {
        const raw = sessionStorage.getItem(key);
        return raw ? JSON.parse(raw) : fallback;
    } catch {
        return fallback;
    }
}
function getSelectedCategoryLabel() {
    const checked = document.querySelector('input[name="category"]:checked');
    if (!checked) return "-";

    const label = checked.closest("label");
    const labelText = label?.querySelector("span")?.textContent?.trim();
    return labelText || checked.value || "-";
}

document.getElementById("downloadReportBtn")?.addEventListener("click", async () => {
    const savedAnalysis = readStoredJson("kasif_last_analysis", {});
    const data = lastAnalysisData || savedAnalysis?.data || {};

    if (!data || Object.keys(data).length === 0) {
        alert("Run an analysis first, then download the PDF report.");
        return;
    }

    const uiDecision = buildUiDecisionFromResult(data);
    const { ai, human } = normalizeProbabilities(
        data.probabilities,
        data.label,
        data.confidence
    );

    const payload = {
        prediction: uiDecision.label || document.getElementById("predictionValue")?.textContent || "N/A",
        ai_probability: `${ai.toFixed(1)}%`,
        human_probability: `${human.toFixed(1)}%`,
        context: getSelectedCategoryLabel(),
        decision_reason: getPredictionTooltipText(uiDecision),
        caution:
            uiDecision.caution ||
            "This result alone is not enough. Please also review the explanation and reasons before making a final decision.",
        top_features:
            data.top_features ||
            readStoredJson("kasif_top_features", []),
        grouped_influences:
            data.grouped_influences ||
            readStoredJson("kasif_grouped_influences", []),
    };

    try {
        const response = await fetch("/download-report", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            let message = "Could not generate the PDF report.";
            try {
                const errorData = await response.json();
                message = errorData.error || message;
            } catch {}
            alert(message);
            return;
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);

        const link = document.createElement("a");
        link.href = url;
        link.download = "kasif_prediction_explanation_report.pdf";
        document.body.appendChild(link);
        link.click();

        link.remove();
        window.URL.revokeObjectURL(url);
    } catch (error) {
        alert("Download failed: " + error.message);
    }
});

codeInput.addEventListener("input", () => {
    lastAnalysisData = null;
    clearKasifSession();
    clearHighlightsOnly();
});

codeInput.addEventListener("scroll", syncEditorScroll);
analyzeBtn.addEventListener("click", runAnalysis);
document.addEventListener("click", (event) => {
    const detailsLink = event.target.closest(
        ".group-more-details-btn, .more-details-btn, a[href='/group-details'], a[href='/waterfall-details']"
    );

    if (detailsLink) {
        saveAnalysisForDetails();
    }
}, true);

function setupFloatingTooltips() {
    let tooltip = document.getElementById("floatingTooltip");

    if (!tooltip) {
        tooltip = document.createElement("div");
        tooltip.id = "floatingTooltip";
        tooltip.className = "floating-tooltip";
        document.body.appendChild(tooltip);
    }

    function showTooltip(target) {
        const text = target.getAttribute("data-tooltip");
        if (!text) return;

        tooltip.innerHTML = `
            <div class="floating-tooltip-title"></div>
            <div>${escapeHtml(text)}</div>
        `;

        tooltip.classList.remove("tooltip-above", "tooltip-below");
        tooltip.classList.add("is-visible");

        const rect = target.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        const margin = 12;

        let left = rect.left + rect.width / 2 - tooltipRect.width / 2;

        left = Math.max(
            margin,
            Math.min(left, window.innerWidth - tooltipRect.width - margin)
        );

        let top = rect.top - tooltipRect.height - margin;
        let placement = "above";

        if (top < margin) {
            top = rect.bottom + margin;
            placement = "below";
        }

        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;

        const arrowLeft = rect.left + rect.width / 2 - left;

        tooltip.style.setProperty("--tooltip-arrow-left", `${arrowLeft}px`);
        tooltip.classList.add(placement === "above" ? "tooltip-above" : "tooltip-below");
    }

    function hideTooltip() {
        tooltip.classList.remove("is-visible", "tooltip-above", "tooltip-below");
    }

    document.addEventListener("mouseover", (event) => {
        const target = event.target.closest("[data-tooltip]");
        if (target) showTooltip(target);
    });

    document.addEventListener("mouseout", (event) => {
        const target = event.target.closest("[data-tooltip]");
        if (target) hideTooltip();
    });

    document.addEventListener("focusin", (event) => {
        const target = event.target.closest("[data-tooltip]");
        if (target) showTooltip(target);
    });

    document.addEventListener("focusout", hideTooltip);
}
function restoreSavedInput() {
    if (!codeInput) return;

    const savedCode = sessionStorage.getItem("kasif_saved_code");
    const savedCategory = sessionStorage.getItem("kasif_saved_category");

    if (savedCode !== null) {
        codeInput.value = savedCode;
    }

    if (savedCategory) {
        const savedInput = document.querySelector(
            `input[name="category"][value="${savedCategory}"]`
        );

        if (savedInput) {
            savedInput.checked = true;
        }
    }
}

function restoreSavedAnalysis() {
    const saved = sessionStorage.getItem("kasif_last_analysis");
    if (!saved) return;

    let parsed;

    try {
        parsed = JSON.parse(saved);
    } catch {
        return;
    }

    const data = parsed.data || {};
    lastAnalysisData = data;
    
    const displayLabel = buildUiDecisionFromResult(data).label || parsed.displayLabel || "-";
    const displayConfidence = Number(parsed.displayConfidence || 0);

    updatePredictionPanel(data);
    renderSummary(displayLabel);
    renderProbabilities(data.probabilities, data.label, data.confidence);

    renderGroupSummary(
        data.grouped_influences || [],
        data.ui_help || {}
    );

    renderWaterfall(
        data.top_features || [],
        displayConfidence,
        data.ui_help?.waterfall || {}
    );

    currentLineFeatureMap = {};

    Object.entries(data.line_to_features || {}).forEach(([lineNumber, featureList]) => {
        currentLineFeatureMap[Number(lineNumber)] = featureList;
    });

    currentPredictedLabel = displayLabel || null;

    renderEditor();
    renderUnmappedFeatures(data.unmapped_features || []);

    setMessage("Restored previous analysis result.", "success");
}

function asPercent(value) {
    const num = Number(value || 0);
    return num <= 1 ? (num * 100) : num;
}

function setPredictionBadgeClass(el, tone) {
    el.classList.remove("is-ai", "is-human", "is-mixed", "is-neutral");

    if (tone === "ai") el.classList.add("is-ai");
    else if (tone === "human") el.classList.add("is-human");
    else if (tone === "mixed") el.classList.add("is-mixed");
    else el.classList.add("is-neutral");
}

function getPredictionTooltipText(uiDecision) {
    const label = uiDecision.label || "";

   if (label === "AI-generated") {
    return "The code is classified as AI-generated because the AI probability is above 70%. It shows more AI-like coding patterns.";
    }

    if (label === "Human" || label === "Human-written") {
        return "The code is classified as Human-written because the Human probability is above 70%. ";
    }

    if (label === "Mixed") {
        return "The code is classified as Mixed because both probabilities are below 70%.";
    }

    return "Hover over the prediction after running analysis to see what it means.";
}

function buildUiDecisionFromResult(result) {
    const existingDecision = result.ui_decision || {};

    const { ai, human } = normalizeProbabilities(
        result.probabilities,
        result.label,
        result.confidence
    );

    let label = "Mixed";
    let tone = "mixed";

    if (ai >= CONFIDENCE_THRESHOLD * 100) {
        label = "AI-generated";
        tone = "ai";
    } else if (human >= CONFIDENCE_THRESHOLD * 100) {
        label = "Human";
        tone = "human";
    }

    return {
        ...existingDecision,
        label,
        tone,
        ai_probability: ai / 100,
        human_probability: human / 100,
        caution:
            existingDecision.caution ||
            "This result alone is not enough. Please also review the explanation and reasons before making a final decision.",
        rule:
            existingDecision.rule ||
            "AI-generated means the AI probability is greater than 70%. Human means the Human probability is greater than 70%. Mixed means both AI and Human probabilities are less than 70%."
    };
}

function updatePredictionPanel(result) {
    if (!result) return;

    const uiDecision = buildUiDecisionFromResult(result);

    const predictionValue = document.getElementById("predictionValue");
    const predictionTooltip = document.getElementById("predictionTooltip");

    const aiProbValue = document.getElementById("aiProbValue");
    const humanProbValue = document.getElementById("humanProbValue");

    const aiProbBar = document.getElementById("aiProbBar");
    const humanProbBar = document.getElementById("humanProbBar");

    const resultCautionText = document.getElementById("resultCautionText");
    const resultRuleText = document.getElementById("resultRuleText");

    const aiPercent = Math.max(0, Math.min(asPercent(uiDecision.ai_probability ?? 0), 100));
    const humanPercent = Math.max(0, Math.min(asPercent(uiDecision.human_probability ?? 0), 100));

    if (predictionValue) {
        predictionValue.textContent = uiDecision.label || "-";
        setPredictionBadgeClass(predictionValue, uiDecision.tone || "neutral");
    }

    if (predictionTooltip) {
        predictionTooltip.textContent = getPredictionTooltipText(uiDecision);
    }

    if (aiProbValue) aiProbValue.textContent = `${aiPercent.toFixed(1)}%`;
    if (humanProbValue) humanProbValue.textContent = `${humanPercent.toFixed(1)}%`;

    if (aiProbBar) aiProbBar.style.width = `${aiPercent.toFixed(1)}%`;
    if (humanProbBar) humanProbBar.style.width = `${humanPercent.toFixed(1)}%`;

    if (resultCautionText) {
        resultCautionText.textContent =
            uiDecision.caution ||
            "This result alone is not enough. Please also review the explanation and reasons before making a final decision.";
    }

    if (resultRuleText) {
        resultRuleText.textContent =
            uiDecision.rule ||
            "AI-generated means the AI probability is greater than 70%. Human means the Human probability is greater than 70%. Mixed means both AI and Human probabilities are less than 70%.";
    }
}
function initializePageState() {
    clearOldLocalStorage();

    const navType = getNavigationType();
    const shouldRestoreFromDetails =
        sessionStorage.getItem("kasif_return_from_details") === "1";

    if (navType === "reload") {
        clearKasifSession();

        if (codeInput) codeInput.value = "";

        const defaultCategory = document.querySelector(
            'input[name="category"][value="assignment"]'
        );

        if (defaultCategory) {
            defaultCategory.checked = true;
        }

        lastAnalysisData = null;
        currentLineFeatureMap = {};
        currentPredictedLabel = null;

        renderEditor();
        renderGroupSummary([]);
        renderWaterfall([]);
        renderUnmappedFeatures([]);
        setMessage("", "");

        return;
    }

    if (shouldRestoreFromDetails) {
        restoreSavedInput();
        restoreSavedAnalysis();
        sessionStorage.removeItem("kasif_return_from_details");
        return;
    }

    clearKasifSession();

    if (codeInput) codeInput.value = "";
    renderEditor();
}

setupFloatingTooltips();
attachContextHoverHelp();
initializePageState();