from __future__ import annotations

import base64
import io
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

POS_COLOR = "#ff0d57"
NEG_COLOR = "#1e88ff"
LOW_CONF_NEG_COLOR = "#16a34a"
TOTAL_COLOR = "#d6004c"
CONNECTOR_COLOR = "#b8b8b8"
TEXT_COLOR = "#222222"
BG_COLOR = "white"


@dataclass
class WaterfallFeature:
    name: str
    shap_value: float


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_name(item: dict) -> str:
    for key in ("display_name", "feature", "feature_name", "name", "label"):
        if key in item and item[key]:
            return str(item[key])
    return "feature"

def _normalize_top_features(
    top_features: Sequence[dict],
    max_features: int = 6,
    low_confidence_mode: bool = False,
) -> List[WaterfallFeature]:
    features: List[WaterfallFeature] = []

    for item in top_features:
        if not isinstance(item, dict):
            continue
        name = _extract_name(item)
        shap_value = _safe_float(item.get("shap_value", 0.0))
        features.append(WaterfallFeature(name=name, shap_value=shap_value))

    features.sort(key=lambda x: abs(x.shap_value), reverse=True)
    features = features[:max_features]

    if low_confidence_mode:
        return features

    negatives = [f for f in features if f.shap_value < 0]
    positives = [f for f in features if f.shap_value >= 0]

    negatives.sort(key=lambda x: x.shap_value)
    positives.sort(key=lambda x: x.shap_value)

    return negatives + positives

def _wrap_label(text: str, width: int = 18) -> str:
    text = str(text).replace("_", " ")
    return "\n".join(textwrap.wrap(text, width=width)) if text else ""


def generate_waterfall_figure(
    top_features: Sequence[dict],
    predicted_class: Optional[str] = None,
    base_value: float = 0.0,
    max_features: int = 6,
    title: str = "Explainability (SHAP Waterfall Chart)",
    prediction_confidence: Optional[float] = None,
    low_conf_threshold: float = 0.7,
):
    is_low_conf = (
        prediction_confidence is not None
        and float(prediction_confidence) < low_conf_threshold
    )

    if is_low_conf:
        title = f"{title} - Low Confidence Mode (< {low_conf_threshold:.1f})"

    features = _normalize_top_features(
        top_features,
        max_features=max_features,
        low_confidence_mode=is_low_conf,
    )

    if not features:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        ax.text(
            0.5,
            0.5,
            "No SHAP waterfall data available.",
            ha="center",
            va="center",
            fontsize=13,
            color="#666666",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return fig

    shap_values = [f.shap_value for f in features]
    names = [_wrap_label(f.name, width=18) for f in features]

    cumulative = [base_value]
    for val in shap_values:
        cumulative.append(cumulative[-1] + val)

    final_value = cumulative[-1]

    if predicted_class is None:
        predicted_class = "AI" if final_value >= 0 else "Human"

    n = len(features)
    x_positions = np.arange(n)
    total_x = n + 1

    fig, ax = plt.subplots(figsize=(max(9, n * 1.4 + 2.5), 4.8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    bar_width = 0.72

    for i, (name, val) in enumerate(zip(names, shap_values)):
        start = cumulative[i]
        end = cumulative[i + 1]

        bottom = min(start, end)
        height = abs(val)

        if is_low_conf:
            color = POS_COLOR if val >= 0 else LOW_CONF_NEG_COLOR
        else:
            color = POS_COLOR if val >= 0 else NEG_COLOR

        ax.bar(
            x_positions[i],
            height,
            bottom=bottom,
            width=bar_width,
            color=color,
            edgecolor=color,
            linewidth=1.0,
            zorder=3,
        )

        ax.text(
            x_positions[i],
            max(start, end) + (0.02 * max(0.35, abs(final_value), 1)),
            f"{val:+.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT_COLOR,
            fontweight="bold",
        )

        ax.text(
            x_positions[i],
            bottom + height / 2,
            name,
            ha="center",
            va="center",
            fontsize=9,
            color="white" if abs(val) > 0.08 else TEXT_COLOR,
            fontweight="medium",
            zorder=4,
        )

        if i < n - 1:
            ax.plot(
                [x_positions[i] + bar_width / 2, x_positions[i + 1] - bar_width / 2],
                [end, end],
                color=CONNECTOR_COLOR,
                linewidth=1.4,
                zorder=2,
            )

    total_bottom = min(base_value, final_value)
    total_height = abs(final_value - base_value)
    if total_height == 0:
        total_height = 0.0001

    ax.bar(
        total_x,
        total_height,
        bottom=total_bottom,
        width=0.85,
        color=TOTAL_COLOR if final_value >= 0 else NEG_COLOR,
        edgecolor=TOTAL_COLOR if final_value >= 0 else NEG_COLOR,
        linewidth=1.2,
        zorder=3,
        alpha=0.96,
    )

    ax.text(
        total_x,
        max(base_value, final_value) + (0.02 * max(0.35, abs(final_value), 1)),
        f"{predicted_class}\n{final_value:+.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color=TEXT_COLOR,
        fontweight="bold",
    )

    ax.axhline(base_value, color="#9e9e9e", linewidth=1.0, zorder=1)

    xticks = list(x_positions) + [total_x]
    xlabels = [""] * n + [f"Total {predicted_class}\nScore"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=10, fontweight="medium")

    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", color=TEXT_COLOR, pad=12)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d0d0d0")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", length=0)
    ax.grid(False)

    y_min = min(min(cumulative), base_value, final_value)
    y_max = max(max(cumulative), base_value, final_value)
    padding = max(0.12, (y_max - y_min) * 0.22)
    ax.set_ylim(y_min - padding, y_max + padding)

    plt.tight_layout()
    return fig


def figure_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def generate_waterfall_base64(
    top_features: Sequence[dict],
    predicted_class: Optional[str] = None,
    base_value: float = 0.0,
    max_features: int = 6,
    title: str = "Explainability (SHAP Waterfall Chart)",
    prediction_confidence: Optional[float] = None,
    low_conf_threshold: float = 0.7,
) -> str:
    fig = generate_waterfall_figure(
        top_features=top_features,
        predicted_class=predicted_class,
        base_value=base_value,
        max_features=max_features,
        title=title,
        prediction_confidence=prediction_confidence,
        low_conf_threshold=low_conf_threshold,
    )
    return figure_to_base64(fig)

def generate_waterfall_html(
    top_features: Sequence[dict],
    predicted_class: Optional[str] = None,
    base_value: float = 0.0,
    max_features: int = 6,
    title: str = "Explainability (SHAP Waterfall Chart)",
    prediction_confidence: Optional[float] = None,
    low_conf_threshold: float = 0.7,
) -> str:
    image_b64 = generate_waterfall_base64(
        top_features=top_features,
        predicted_class=predicted_class,
        base_value=base_value,
        max_features=max_features,
        title=title,
        prediction_confidence=prediction_confidence,
        low_conf_threshold=low_conf_threshold,
    )
    return (
        f'<img src="data:image/png;base64,{image_b64}" '
        f'alt="SHAP Waterfall Chart" class="waterfall-image" />'
    )
def _supports_prediction(item: dict, predicted_label: str) -> bool:
    shap_value = _safe_float(item.get("shap_value", 0.0), 0.0)
    if predicted_label == "AI-generated":
        return shap_value > 0
    return shap_value < 0


def _sort_by_abs_shap(features: Sequence[dict]) -> List[dict]:
    return sorted(
        [f for f in features if isinstance(f, dict)],
        key=lambda x: abs(_safe_float(x.get("shap_value", 0.0), 0.0)),
        reverse=True,
    )


def _normalized_confidence(confidence) -> float:
    conf = _safe_float(confidence, 0.0)
    if conf > 1:
        conf = conf / 100.0
    return conf


def _pick_mixed_by_sign(features: Sequence[dict], limit: int = 10) -> List[dict]:
    positives = _sort_by_abs_shap(
        [f for f in features if _safe_float(f.get("shap_value", 0.0), 0.0) > 0]
    )
    negatives = _sort_by_abs_shap(
        [f for f in features if _safe_float(f.get("shap_value", 0.0), 0.0) < 0]
    )

    half = limit // 2
    selected = positives[:half] + negatives[:half]

    used_ids = {id(x) for x in selected}
    remaining = [
        f for f in _sort_by_abs_shap(features)
        if id(f) not in used_ids
    ]

    selected.extend(remaining[: max(0, limit - len(selected))])
    return _sort_by_abs_shap(selected)[:limit]


def select_waterfall_features(
    top_features: Sequence[dict],
    predicted_label: str,
    confidence,
    limit: int = 10,
    threshold: float = 0.70,
) -> List[dict]:
    all_features = [f for f in top_features if isinstance(f, dict)]
    conf = _normalized_confidence(confidence)

    if conf >= threshold:
        selected = [
            f for f in all_features
            if _supports_prediction(f, predicted_label)
        ]
        selected = _sort_by_abs_shap(selected)

        if not selected:
            selected = _sort_by_abs_shap(all_features)

        return selected[:limit]

    return _pick_mixed_by_sign(all_features, limit=limit)