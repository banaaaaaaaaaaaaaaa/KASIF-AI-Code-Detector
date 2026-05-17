from pathlib import Path
import sys
from collections import defaultdict

from flask import Flask, jsonify, render_template, request, send_file
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.pipeline.predict import predict_code
from src.explainability.line_highlighter import explain_code_with_line_highlights
from src.explainability.shap_waterfall import select_waterfall_features
from src.explainability.explanation_catalog import (
    attach_feature_explanations,
    attach_group_explanations,
    get_group_panel_info,
    get_group_more_details_url,
)

app = Flask(__name__)

CONFIDENCE_THRESHOLD = 0.70
DISPLAY_FEATURE_LIMIT = 30


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/group-details")
def group_details():
    return render_template("group_details.html")


@app.route("/waterfall-details")
def waterfall_details():
    return render_template("waterfall_details.html")
PDF_GROUP_EXPLANATIONS = {
    "LEXICAL": "Lexical features describe the surface style of the code, such as naming, line length, keywords, literals, and spacing.",
    "SYNTACTIC": "Syntactic features describe the code structure, such as loops, conditions, nesting, and AST patterns.",
    "STRUCTURAL": "Structural features describe how the code is organized, such as functions, classes, assignments, and statements.",
    "COMPLEXITY": "Complexity features describe how complicated the logic is, especially branches, conditions, and nested code.",
    "COMPLEXITY_MAINTAINABILITY": "These features describe code complexity and how easy the code may be to read and maintain.",
    "HALSTEAD": "Halstead features measure operators and operands to describe code complexity and coding style.",
    "EMBEDDING": "Embedding features are learned automatically by the model. They capture hidden patterns, but they are not directly human-readable.",
    "OTHER": "Other useful features that do not belong to one main group.",
    "UNMAPPED": "This feature was not mapped to a specific group.",
}


def _pdf_text(value, default="-"):
    text = str(value if value is not None else default).strip()
    return xml_escape(text if text else default)


def _pdf_percent(value):
    try:
        if isinstance(value, str) and "%" in value:
            return value
        num = float(value or 0)
        if num <= 1:
            num *= 100
        return f"{num:.1f}%"
    except Exception:
        return str(value or "0%")


def _pdf_shap(value):
    try:
        num = float(value or 0)
        return f"{num:+.4f}"
    except Exception:
        return str(value or "0")


def _feature_name(item):
    for key in ("display_name", "feature", "feature_name", "name", "label"):
        if item.get(key):
            return str(item.get(key))
    return "-"


def _feature_direction(item):
    if item.get("direction"):
        return str(item.get("direction"))
    try:
        shap_value = float(item.get("shap_value", 0) or 0)
        if shap_value > 0:
            return "Supports AI-generated"
        if shap_value < 0:
            return "Supports Human"
        return "Neutral"
    except Exception:
        return "-"


def _paragraph(text, style):
    return Paragraph(_pdf_text(text), style)


def _make_table(rows, col_widths, styles, repeat_rows=1):
    return Table(rows, colWidths=col_widths, repeatRows=repeat_rows, hAlign="LEFT", style=styles)


def _table_style(header_color="#0f2b3d"):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("FONTSIZE", (0, 1), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])



REPORT_COLORS = {
    "navy": colors.HexColor("#10283D"),
    "blue": colors.HexColor("#10283D"),
    "teal": colors.HexColor("#10283D"),

    "red": colors.HexColor("#DC2626"),
    "green": colors.HexColor("#0F8A4B"),
    "amber": colors.HexColor("#B45309"),

    "ink": colors.HexColor("#10283D"),
    "muted": colors.HexColor("#5B6B82"),
    "line": colors.HexColor("#D8E0E8"),

    "soft_blue": colors.HexColor("#EEF5FF"),
    "soft_red": colors.HexColor("#FEEDEE"),
    "soft_green": colors.HexColor("#EDF9F2"),
    "soft_amber": colors.HexColor("#FFF6E8"),
    "soft_gray": colors.HexColor("#F7F9FB"),
    "table_header": colors.HexColor("#E7EDF3"),
    "table_header_text": colors.HexColor("#10283D"),
    "white": colors.white,
}


def _prediction_tone(label: str) -> str:
    text = str(label or "").strip().lower()
    if "mixed" in text:
        return "mixed"
    if "ai" in text:
        return "ai"
    return "human"


def _tone_colors(label: str):
    tone = _prediction_tone(label)
    if tone == "ai":
        return REPORT_COLORS["soft_red"], REPORT_COLORS["red"]
    if tone == "mixed":
        return REPORT_COLORS["soft_amber"], REPORT_COLORS["amber"]
    return REPORT_COLORS["soft_green"], REPORT_COLORS["green"]


def _draw_fallback_report_logo(canvas, x, y, size=30):
    """Draw a simple fallback logo if the project logo image is not available."""
    canvas.saveState()
    canvas.setFillColor(REPORT_COLORS["soft_blue"])
    canvas.roundRect(x, y, size, size, 8, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.circle(x + size * 0.38, y + size * 0.62, size * 0.18, fill=1, stroke=0)
    canvas.setStrokeColor(colors.white)
    canvas.setLineWidth(2.8)
    canvas.line(x + size * 0.52, y + size * 0.48, x + size * 0.73, y + size * 0.27)
    canvas.setFont("Helvetica-Bold", size * 0.34)
    canvas.drawString(x + size * 0.16, y + size * 0.15, "K")
    canvas.restoreState()


def _draw_report_logo(canvas, x, y, size=30):
    """
    Prefer the real project logo in static/images/logo.png.
    If it is missing or cannot be read, draw the fallback logo.
    """
    logo_candidates = [
        APP_DIR / "static" / "images" / "logo.png",
        APP_DIR / "static" / "images" / "logo.jpg",
        APP_DIR / "static" / "images" / "logo.jpeg",
    ]

    for logo_path in logo_candidates:
        if logo_path.exists():
            try:
                canvas.saveState()
                canvas.drawImage(
                    str(logo_path),
                    x,
                    y,
                    width=size,
                    height=size,
                    preserveAspectRatio=True,
                    mask="auto",
                    anchor="c",
                )
                canvas.restoreState()
                return
            except Exception:
                canvas.restoreState()

    _draw_fallback_report_logo(canvas, x, y, size=size)


def _draw_report_chrome(canvas, doc):
    """Draw the PDF header/footer using the same clean colors as the web app."""
    canvas.saveState()

    page_w, page_h = doc.pagesize
    left = doc.leftMargin
    right = page_w - doc.rightMargin

    header_y = page_h - 0.78 * inch

    # Bigger logo
    _draw_report_logo(canvas, left, header_y - 6, size=58)

    # Bigger title and clearer subtitle
    canvas.setFillColor(REPORT_COLORS["ink"])
    canvas.setFont("Helvetica-Bold", 24)
    canvas.drawString(left + 72, header_y + 24, "KASIF Report")

    canvas.setFillColor(REPORT_COLORS["muted"])
    canvas.setFont("Helvetica", 13)
    canvas.drawString(left + 72, header_y + 7, "AI Code Detection Summary")

    # Divider line under header
    canvas.setStrokeColor(REPORT_COLORS["line"])
    canvas.setLineWidth(1.1)
    canvas.line(left, header_y - 18, right, header_y - 18)

    # Footer
    footer_y = 0.42 * inch
    canvas.setStrokeColor(REPORT_COLORS["line"])
    canvas.setLineWidth(0.9)
    canvas.line(left, footer_y + 10, right, footer_y + 10)

    canvas.setFillColor(REPORT_COLORS["muted"])
    canvas.setFont("Helvetica", 9.5)
    canvas.drawString(left, footer_y - 2, "KASIF • Explainable AI Code Detector")
    canvas.drawRightString(right, footer_y - 2, f"Page {canvas.getPageNumber()}")

    canvas.restoreState()


def _section_title(text, styles):
    return Paragraph(f'<font color="#102A43"><b>{_pdf_text(text)}</b></font>', styles["Heading2"])


def _info_box(title, body, styles, bg="#FFFFFF", border="#D8E0E8", title_color="#10283D"):
    content = [[Paragraph(
        f'<font color="{title_color}"><b>{_pdf_text(title)}:</b></font> <font color="#10283D">{_pdf_text(body)}</font>',
        styles["SmallBody"],
    )]]
    table = Table(content, colWidths=[7.1 * inch])
    table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
    ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ("TOPPADDING", (0, 0), (-1, -1), 8),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
]))
    return table


def _summary_cards(prediction, ai_probability, human_probability, context, styles):
    # Clean white cards: no colored backgrounds and no colored probability numbers.
    cards = [[
        Paragraph(
            f'<font color="#5B6B82" size="9">Context</font><br/>'
            f'<font color="#10283D" size="14"><b>{_pdf_text(context)}</b></font>',
            styles["SmallBody"],
        ),
        Paragraph(
            f'<font color="#5B6B82" size="9">Prediction</font><br/>'
            f'<font color="#10283D" size="14"><b>{_pdf_text(prediction)}</b></font>',
            styles["SmallBody"],
        ),
        Paragraph(
            f'<font color="#5B6B82" size="9">AI Probability</font><br/>'
            f'<font color="#10283D" size="18"><b>{_pdf_text(ai_probability)}</b></font>',
            styles["SmallBody"],
        ),
        Paragraph(
            f'<font color="#5B6B82" size="9">Human Probability</font><br/>'
            f'<font color="#10283D" size="18"><b>{_pdf_text(human_probability)}</b></font>',
            styles["SmallBody"],
        ),
    ]]

    tbl = Table(
        cards,
        colWidths=[1.90 * inch, 1.70 * inch, 1.70 * inch, 1.70 * inch],
    )

    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.8, REPORT_COLORS["line"]),
        ("INNERGRID", (0, 0), (-1, -1), 0.8, REPORT_COLORS["line"]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 13),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 13),
    ]))

    return tbl


def _styled_table(rows, col_widths, header_bg=None, header_fg=None, zebra=True):
    table = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")

    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg or REPORT_COLORS["table_header"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), header_fg or REPORT_COLORS["table_header_text"]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8.4),
        ("GRID", (0, 0), (-1, -1), 0.55, REPORT_COLORS["line"]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ])

    if zebra:
        for r in range(1, len(rows)):
            bg = colors.white if r % 2 else REPORT_COLORS["soft_gray"]
            style.add("BACKGROUND", (0, r), (-1, r), bg)

    table.setStyle(style)
    return table


@app.route("/download-report", methods=["POST"])
def download_report():
    """Create a redesigned PDF report from the current browser analysis result."""
    data = request.get_json(silent=True) or {}

    reports_dir = APP_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    pdf_path = reports_dir / "latest_analysis_report.pdf"

    prediction = data.get("prediction", "N/A")
    ai_probability = _pdf_percent(data.get("ai_probability", "0%"))
    human_probability = _pdf_percent(data.get("human_probability", "0%"))
    context = data.get("context", "-")
    decision_reason = data.get("decision_reason") or "The prediction was selected based on the displayed AI and Human probabilities."
    caution = data.get("caution") or "This result alone is not enough. Please also review the explanation and reasons before making a final decision."
    top_features = data.get("top_features") or []
    grouped_influences = data.get("grouped_influences") or []

    styles = getSampleStyleSheet()

    styles["Title"].fontName = "Helvetica-Bold"

    styles["Heading2"].fontName = "Helvetica-Bold"
    styles["Heading2"].fontSize = 17
    styles["Heading2"].leading = 21
    styles["Heading2"].spaceAfter = 10
    styles["Heading2"].textColor = REPORT_COLORS["navy"]

    styles["BodyText"].textColor = REPORT_COLORS["ink"]
    styles["BodyText"].fontSize = 10.2
    styles["BodyText"].leading = 14

    styles.add(ParagraphStyle(
        name="SmallBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=13.5,
        alignment=TA_LEFT,
        textColor=REPORT_COLORS["ink"],
    ))

    styles.add(ParagraphStyle(
        name="TinyBody",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11.2,
        alignment=TA_LEFT,
        textColor=REPORT_COLORS["ink"],
    ))

    styles.add(ParagraphStyle(
        name="Meta",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=12,
        textColor=REPORT_COLORS["muted"],
        spaceAfter=9,
    ))

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=0.42 * inch,
        leftMargin=0.42 * inch,
        topMargin=1.45 * inch,
        bottomMargin=0.65 * inch,
        title="KASIF AI Code Detection Report",
    )
    doc.report_prediction = prediction

    story = []
    story.append(Paragraph(f"Generated on: {_pdf_text(datetime.now().strftime('%Y-%m-%d %H:%M'))}", styles["Meta"]))
    story.append(_section_title("Prediction Summary", styles))
    story.append(_summary_cards(prediction, ai_probability, human_probability, context, styles))
    story.append(Spacer(1, 12))

    story.append(_info_box("Decision reason", decision_reason, styles))
    story.append(Spacer(1, 6))
    story.append(_info_box("Caution", caution, styles))
    story.append(Spacer(1, 12))

    story.append(_section_title("How to Read the Explainability Values", styles))
    explain_text = (
        "Positive SHAP values support the AI-generated class. "
        "Negative SHAP values support the Human class. "
        "A larger absolute SHAP value means the feature had a stronger influence on the prediction."
    )
    story.append(_info_box("Explanation", explain_text, styles))
    story.append(Spacer(1, 12))

    story.append(_section_title("Group Influence Percentages", styles))
    if grouped_influences:
        total_score = sum(max(float(g.get("group_support_score", 0) or 0), 0) for g in grouped_influences) or 1
        group_rows = [[
            Paragraph('<font color="#10283D"><b>Group</b></font>', styles["TinyBody"]),
            Paragraph('<font color="#10283D"><b>Influence</b></font>', styles["TinyBody"]),
            Paragraph('<font color="#10283D"><b>Features</b></font>', styles["TinyBody"]),
            Paragraph('<font color="#10283D"><b>Meaning</b></font>', styles["TinyBody"]),
        ]]
        for group in grouped_influences:
            group_name = str(group.get("group") or "OTHER").upper()
            score = max(float(group.get("group_support_score", 0) or 0), 0)
            percent = (score / total_score) * 100
            explanation = group.get("explanation") or PDF_GROUP_EXPLANATIONS.get(group_name, "No group explanation available.")
            group_rows.append([
                Paragraph(_pdf_text(group_name), styles["TinyBody"]),
                Paragraph(f'<b>{percent:.1f}%</b>', styles["TinyBody"]),
                Paragraph(_pdf_text(str(group.get("count", 0))), styles["TinyBody"]),
                Paragraph(_pdf_text(explanation), styles["TinyBody"]),
            ])
        story.append(_styled_table(
            group_rows,
            [1.30 * inch, 0.95 * inch, 0.85 * inch, 4.10 * inch],
            header_bg=REPORT_COLORS["table_header"],
        ))
    else:
        story.append(Paragraph("No group influence data was available.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(_section_title("Top Features Explaining the Decision", styles))
    story.append(Paragraph(
        "These are the most influential features behind the prediction, ranked by SHAP magnitude.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 4))
    if top_features:
        feature_rows = [[
            Paragraph('<font color="#10283D"><b>#</b></font>', styles["TinyBody"]),
            Paragraph('<font color="#10283D"><b>Feature</b></font>', styles["TinyBody"]),
            Paragraph('<font color="#10283D"><b>Group</b></font>', styles["TinyBody"]),
            Paragraph('<font color="#10283D"><b>SHAP</b></font>', styles["TinyBody"]),
            Paragraph('<font color="#10283D"><b>Direction</b></font>', styles["TinyBody"]),
            Paragraph('<font color="#10283D"><b>Explanation</b></font>', styles["TinyBody"]),
        ]]
        sorted_features = sorted(
            top_features,
            key=lambda x: abs(float(x.get("shap_value", 0) or 0)),
            reverse=True,
        )[:30]
        for idx, item in enumerate(sorted_features, start=1):
            direction = _feature_direction(item)
            shap_val = _pdf_shap(item.get("shap_value"))
            feature_rows.append([
                Paragraph(str(idx), styles["TinyBody"]),
                Paragraph(_pdf_text(_feature_name(item)), styles["TinyBody"]),
                Paragraph(_pdf_text(item.get("group") or "-"), styles["TinyBody"]),
                Paragraph(f'<b>{_pdf_text(shap_val)}</b>', styles["TinyBody"]),
                Paragraph(f'<b>{_pdf_text(direction)}</b>', styles["TinyBody"]),
                Paragraph(_pdf_text(item.get("explanation") or "No explanation available."), styles["TinyBody"]),
            ])
        story.append(_styled_table(
            feature_rows,
            [0.34 * inch, 1.35 * inch, 0.92 * inch, 0.72 * inch, 1.12 * inch, 2.75 * inch],
            header_bg=REPORT_COLORS["table_header"],
        ))
    else:
        story.append(Paragraph("No SHAP feature data was available.", styles["BodyText"]))

    doc.build(story, onFirstPage=_draw_report_chrome, onLaterPages=_draw_report_chrome)

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name="kasif_prediction_explanation_report.pdf",
        mimetype="application/pdf",
    )


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalized_confidence(confidence):
    conf = _safe_float(confidence, 0.0)
    if conf > 1:
        conf = conf / 100.0
    return conf

def _normalize_probability_value(value):
    prob = _safe_float(value, 0.0)
    if prob > 1:
        prob = prob / 100.0
    return max(0.0, min(prob, 1.0))


def _extract_probability_pair(probabilities, predicted_label=None, confidence=None):
    """
    Return (ai_probability, human_probability) as unit values between 0 and 1.

    Supports both common formats:
    - dict: {"AI-generated": 0.85, "Human": 0.15}
    - list/tuple: [human_probability, ai_probability]
    """
    ai_prob = None
    human_prob = None

    if isinstance(probabilities, dict):
        ai_keys = (
            "AI-generated", "AI", "ai",
            "ai_probability", "ai_generated",
            "AI Probability", "AI probability",
            "1", 1,
        )
        human_keys = (
            "Human-written", "Human", "human",
            "human_probability", "human_written",
            "Human Probability", "Human probability",
            "0", 0,
        )

        for key in ai_keys:
            if key in probabilities:
                ai_prob = _normalize_probability_value(probabilities.get(key))
                break

        for key in human_keys:
            if key in probabilities:
                human_prob = _normalize_probability_value(probabilities.get(key))
                break

    elif isinstance(probabilities, (list, tuple)) and len(probabilities) >= 2:
        # The frontend already treats probability arrays as [Human, AI].
        human_prob = _normalize_probability_value(probabilities[0])
        ai_prob = _normalize_probability_value(probabilities[1])

    if ai_prob is None or human_prob is None:
        conf = _normalized_confidence(confidence or 0.0)
        label_text = str(predicted_label or "").strip().lower()

        if conf > 0:
            if "ai" in label_text:
                ai_prob = conf if ai_prob is None else ai_prob
                human_prob = (1.0 - conf) if human_prob is None else human_prob
            elif "human" in label_text:
                human_prob = conf if human_prob is None else human_prob
                ai_prob = (1.0 - conf) if ai_prob is None else ai_prob

    return (
        0.0 if ai_prob is None else max(0.0, min(float(ai_prob), 1.0)),
        0.0 if human_prob is None else max(0.0, min(float(human_prob), 1.0)),
    )


def _get_prob(probabilities, *keys):
    # Backward-compatible helper. Prefer _extract_probability_pair for final decisions.
    if not isinstance(probabilities, dict):
        return 0.0

    for key in keys:
        if key in probabilities:
            return _normalize_probability_value(probabilities.get(key, 0.0))
    return 0.0


def build_ui_decision(probabilities, threshold=0.70, predicted_label=None, confidence=None):
    ai_prob, human_prob = _extract_probability_pair(
        probabilities=probabilities,
        predicted_label=predicted_label,
        confidence=confidence,
    )

    if ai_prob >= threshold:
        label = "AI-generated"
        tone = "ai"
        reason = "AI-generated means the AI probability is greater than 70%."
    elif human_prob >= threshold:
        label = "Human"
        tone = "human"
        reason = "Human means the Human probability is greater than 70%."
    else:
        label = "Mixed"
        tone = "mixed"
        reason = "Mixed means both AI and Human probabilities are less than 70%."

    caution = (
        "This result alone is not enough. Please also review the explanation "
        "and reasons before making a final decision."
    )

    rule = (
        "If AI probability is above 70%, the result is AI-generated. "
        "If Human probability is above 70%, the result is Human. "
        "If neither exceeds 70%, the result is Mixed."
    )

    return {
        "label": label,
        "tone": tone,
        "reason": reason,
        "caution": caution,
        "rule": rule,
        "ai_probability": ai_prob,
        "human_probability": human_prob,
        "threshold": threshold,
    }


def _supports_prediction(item, predicted_label):
    shap_value = _safe_float(item.get("shap_value", 0.0), 0.0)
    if predicted_label == "AI-generated":
        return shap_value > 0
    return shap_value < 0


def _sort_by_abs_shap(features):
    return sorted(
        features,
        key=lambda x: abs(_safe_float(x.get("shap_value", 0.0), 0.0)),
        reverse=True,
    )


def _is_line_eligible_feature(item):
    if not isinstance(item, dict):
        return False

    if item.get("mapped_feature_name"):
        return True

    feature_type = str(item.get("feature_type", "")).strip().lower()
    return (
        ("software engineering" in feature_type)
        or ("software_engineering_features" in feature_type)
        or ("manual" in feature_type)
    )
def _pick_mixed_by_sign(features, limit=30):
    positives = _sort_by_abs_shap([
        f for f in features
        if _safe_float(f.get("shap_value", 0.0), 0.0) > 0
    ])

    negatives = _sort_by_abs_shap([
        f for f in features
        if _safe_float(f.get("shap_value", 0.0), 0.0) < 0
    ])

    half = limit // 2
    selected = positives[:half] + negatives[:half]

    used_ids = {id(x) for x in selected}
    remaining = [
        f for f in _sort_by_abs_shap(features)
        if id(f) not in used_ids
    ]

    selected.extend(remaining[:max(0, limit - len(selected))])
    return _sort_by_abs_shap(selected)[:limit]


def select_line_highlight_features(
    top_features,
    predicted_label,
    confidence,
    limit=10,
    threshold=0.70,
):
    line_features = [f for f in top_features if _is_line_eligible_feature(f)]
    conf = _normalized_confidence(confidence)

    if conf >= threshold:
        selected = [
            f for f in line_features
            if _supports_prediction(f, predicted_label)
        ]
        selected = _sort_by_abs_shap(selected)

        if not selected:
            selected = _sort_by_abs_shap(line_features)

        return selected[:limit]

    return _pick_mixed_by_sign(line_features, limit=limit)


def build_grouped_influences_from_features(features):
    grouped = defaultdict(
        lambda: {"group": "Other", "group_support_score": 0.0, "count": 0}
    )

    for item in features:
        group_name = str(item.get("group") or "Other")
        shap_value = abs(_safe_float(item.get("shap_value", 0.0), 0.0))

        grouped[group_name]["group"] = group_name
        grouped[group_name]["group_support_score"] += shap_value
        grouped[group_name]["count"] += 1

    return sorted(
        grouped.values(),
        key=lambda x: x["group_support_score"],
        reverse=True,
    )



@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(silent=True) or {}

        code_input = (data.get("code") or "").strip()
        assessment_type = (data.get("assessment_type") or "").strip()

        if not code_input:
            return jsonify({
                "success": False,
                "error": "Please enter code first."
            }), 400

        allowed_types = {
            "assignment",
            "labs-aybu",
            "exams-aybu",
        }

        if assessment_type not in allowed_types:
            return jsonify({
                "success": False,
                "error": "Invalid assessment type."
            }), 400

        result = predict_code(code_input, assessment_type)

        label = result.get("label", "N/A")
        confidence = result.get("predicted_class_confidence", 0.0)
        raw_top_features = result.get("top_features", [])
        probabilities = result.get("probabilities", {})
        ui_decision = build_ui_decision(
            probabilities=probabilities,
            threshold=CONFIDENCE_THRESHOLD,
            predicted_label=label,
            confidence=confidence,
        )

        display_top_features = select_waterfall_features(
            top_features=raw_top_features,
            predicted_label=label,
            confidence=confidence,
            limit=DISPLAY_FEATURE_LIMIT,
            threshold=CONFIDENCE_THRESHOLD,
        )

        display_top_features = attach_feature_explanations(display_top_features)

        line_highlight_features = select_line_highlight_features(
            top_features=raw_top_features,
            predicted_label=label,
            confidence=confidence,
            limit=DISPLAY_FEATURE_LIMIT,
            threshold=CONFIDENCE_THRESHOLD,
        )

        grouped_influences = build_grouped_influences_from_features(display_top_features)
        grouped_influences = attach_group_explanations(grouped_influences)

        highlight_result = explain_code_with_line_highlights(
            code=code_input,
            top_features=line_highlight_features,
            predicted_label=label,
        )

        return jsonify({
            "success": True,
            "label": label,
            "display_label": ui_decision["label"],
            "ui_decision": ui_decision,
            "confidence": confidence,
            "probabilities": probabilities,
            "top_features": display_top_features,
            "grouped_influences": grouped_influences,
            
            "line_to_features": highlight_result.get("line_to_features", {}),
            "unmapped_features": highlight_result.get("unmapped_features", []),
            "ui_help": {
                "group_panel_info": get_group_panel_info(),
                "group_more_details_url": get_group_more_details_url(),
            },
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True) 