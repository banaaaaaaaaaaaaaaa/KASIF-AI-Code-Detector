from __future__ import annotations

import ast
import html
import io
import re
import tokenize
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set


def format_shap(value: float) -> str:
    return f"{value:+.4f}"


def get_feature_line_matchers() -> Dict[str, Any]:
    return {
        "def_Density": lambda s: re.search(r"^\s*def\s+", s) is not None,
        "for_Density": lambda s: re.search(r"^\s*for\s+", s) is not None,
        "if_Density": lambda s: re.search(r"^\s*if\s+", s) is not None,
        "while_Density": lambda s: re.search(r"^\s*while\s+", s) is not None,
        "elif_Density": lambda s: re.search(r"^\s*elif\s+", s) is not None,
        "else_Density": lambda s: re.search(r"^\s*else\s*:", s) is not None,
        "break_Density": lambda s: re.search(r"\bbreak\b", s) is not None,
        "continue_Density": lambda s: re.search(r"\bcontinue\b", s) is not None,
        "return_Density": lambda s: re.search(r"\breturn\b", s) is not None,
        "yield_Density": lambda s: re.search(r"\byield\b", s) is not None,
        "try_Density": lambda s: re.search(r"^\s*try\s*:", s) is not None,
        "except_Density": lambda s: re.search(r"^\s*except\b", s) is not None,
        "import_Density": lambda s: re.search(r"^\s*import\b", s) is not None,
        "from_Density": lambda s: re.search(r"^\s*from\b", s) is not None,
        "class_Density": lambda s: re.search(r"^\s*class\s+", s) is not None,
        "lambda_Density": lambda s: re.search(r"\blambda\b", s) is not None,
        "numInputStmtsDensity": lambda s: re.search(r"\binput\s*\(", s) is not None,
        "numFunctionsDensity": lambda s: re.search(r"^\s*def\s+", s) is not None,
    }


def feature_matches_line(feature_name: str, line: str) -> bool:
    matcher = get_feature_line_matchers().get(feature_name)
    return bool(matcher(line)) if matcher else False


def _exact_feature_lines(feature_name: str, code_lines: List[str]) -> Set[int]:
    return {
        i
        for i, line in enumerate(code_lines, start=1)
        if feature_matches_line(feature_name, line)
    }


def _friendly_feature_reason(feature_name: str) -> str:
    reason_map = {
        "def_Density": "this line defines a function",
        "for_Density": "this line starts a for loop",
        "if_Density": "this line starts an if condition",
        "while_Density": "this line starts a while loop",
        "elif_Density": "this line starts an elif branch",
        "else_Density": "this line starts an else branch",
        "break_Density": "this line uses break",
        "continue_Density": "this line uses continue",
        "return_Density": "this line returns a value",
        "yield_Density": "this line uses yield",
        "try_Density": "this line starts a try block",
        "except_Density": "this line starts an except block",
        "import_Density": "this line imports a module",
        "from_Density": "this line imports from a module",
        "class_Density": "this line defines a class",
        "lambda_Density": "this line uses a lambda expression",
        "numInputStmtsDensity": "this line reads input",
        "numFunctionsDensity": "this line contributes to the number of functions",
    }
    return reason_map.get(feature_name, f"this line matched {feature_name}")


def _support_label(shap_value: float) -> str:
    if shap_value > 0:
        return "AI"
    if shap_value < 0:
        return "Human"
    return "Neutral"


def _line_support_class(features: List[Dict[str, Any]]) -> str:
    total = sum(float(f.get("shap_value", 0.0) or 0.0) for f in features)
    if total > 0:
        return "support-ai"
    if total < 0:
        return "support-human"
    return "support-neutral"


def _render_reason_text(feat: Dict[str, Any]) -> str:
    feature_name = str(feat.get("feature_name", "Unknown Feature"))
    source_feature_name = str(feat.get("source_feature_name") or feature_name)
    shap_value = float(feat.get("shap_value", 0.0) or 0.0)
    corr_r = feat.get("corr_r")
    support = _support_label(shap_value)
    why_text = _friendly_feature_reason(feature_name)

    if source_feature_name != feature_name:
        mapping_text = f"{html.escape(source_feature_name)} → {html.escape(feature_name)}"
    else:
        mapping_text = html.escape(feature_name)

    corr_text = ""
    if corr_r is not None and corr_r == corr_r:
        corr_text = f" Correlation r={float(corr_r):.3f}."

    return (
        f"Supports <b>{support}</b> because {html.escape(why_text)} "
        f"(matched <code>{mapping_text}</code>, SHAP {format_shap(shap_value)}).{corr_text}"
    )


def build_line_feature_map(mapped_features: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    line_feature_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for feat in mapped_features:
        lines = feat.get("matched_lines") or feat.get("line_numbers") or []

        if isinstance(lines, int):
            lines = [lines]

        for line_no in sorted(set(lines)):
            line_feature_map[line_no].append(
                {
                    "feature_name": feat.get("feature_name", "Unknown Feature"),
                    "source_feature_name": feat.get("source_feature_name"),
                    "feature_type": feat.get("group") or feat.get("feature_type", "Unknown"),
                    "shap_value": feat.get("shap_value", 0.0),
                    "corr_r": feat.get("corr_r"),
                }
            )

    for line_no in line_feature_map:
        line_feature_map[line_no].sort(
            key=lambda x: abs(float(x["shap_value"])),
            reverse=True,
        )

    return dict(line_feature_map)


def render_code_with_feature_labels(
    code_text: str,
    mapped_features: List[Dict[str, Any]],
    predicted_class: Optional[str],
) -> str:
    code_lines = code_text.splitlines()
    line_feature_map = build_line_feature_map(mapped_features)
    predicted_text = str(predicted_class or "").strip()

    rows_html: List[str] = []

    for i, line in enumerate(code_lines, start=1):
        features = line_feature_map.get(i, [])
        is_highlighted = len(features) > 0

        badges_html = ""
        reasons_html = ""
        line_class = "line-code"

        if is_highlighted:
            line_class = f"line-code {_line_support_class(features)}"

        for feat in features:
            shap_value = float(feat["shap_value"])
            shap_class = "shap-ai" if shap_value > 0 else "shap-human" if shap_value < 0 else "shap-neutral"
            badges_html += f'''
                <div class="feature-badge">
                    <span class="feature-name">{html.escape(str(feat["feature_name"]))}</span>
                    <span class="feature-type">{html.escape(str(feat["feature_type"]))}</span>
                    <span class="feature-shap {shap_class}">{format_shap(shap_value)}</span>
                </div>
            '''
            reasons_html += f'<div class="line-reason">{_render_reason_text(feat)}</div>'

        safe_line = html.escape(line) if line.strip() else "&nbsp;"

        rows_html.append(
            f'''
            <div class="code-row">
                <div class="line-number">{i}</div>
                <div class="{line_class}">{safe_line}</div>
                <div class="line-meta">{reasons_html or badges_html}</div>
            </div>
            '''
        )

    return f'''
    <div class="code-support-wrapper">
        <h2 class="code-support-title">Code Lines Supporting the Prediction</h2>
        <p class="prediction-note">
            Line colors are based on each line's own SHAP direction. Current prediction:
            <b>{html.escape(predicted_text)}</b>
        </p>

        <div class="code-table">
            {"".join(rows_html)}
        </div>
    </div>
    '''


def explain_code_with_line_highlights(
    code: str,
    top_features: List[Dict[str, Any]],
    predicted_label: Optional[str] = None,
    include_embeddings: bool = False,
    strict_line_mode: bool = False,
) -> Dict[str, Any]:
    code_lines = code.splitlines()
    line_count = len(code_lines)

    if line_count == 0:
        return {
            "predicted_label": predicted_label,
            "html": render_code_with_feature_labels("", [], predicted_label),
            "line_scores": [],
            "feature_to_lines": {},
            "line_to_features": {},
            "unmapped_features": [],
        }

    ast_tree = _safe_parse(code)
    ast_index = _build_ast_index(ast_tree)
    token_index = _build_token_index(code_lines)
    stats = _compute_line_stats(code_lines)

    line_scores: Dict[int, float] = defaultdict(float)
    feature_to_lines: Dict[str, List[int]] = {}
    line_to_features: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    unmapped_features: List[str] = []
    mapped_features: List[Dict[str, Any]] = []

    for item in top_features:
        feature_name = str(
            item.get("mapped_feature_name")
            or item.get("feature")
            or item.get("feature_name")
            or ""
        ).strip()
        source_feature_name = str(
            item.get("source_feature_name")
            or item.get("source_feature")
            or feature_name
        ).strip()
        feature_type = str(item.get("feature_type", "")).strip().lower()
        shap_value = float(item.get("shap_value", 0.0) or 0.0)
        corr_r = item.get("corr_r")

        if not feature_name:
            continue
        if ("embedding" in feature_type) and not include_embeddings and not item.get("mapped_feature_name"):
            continue
        if not _is_manual_feature(item):
            continue

        exact_lines = sorted(_exact_feature_lines(feature_name, code_lines))

        if strict_line_mode:
            matched_lines = exact_lines
        else:
            matched_lines = exact_lines or sorted(
                _map_feature_to_lines(
                    feature_name=feature_name,
                    code_lines=code_lines,
                    ast_index=ast_index,
                    token_index=token_index,
                    stats=stats,
                )
            )

        if not matched_lines:
            unmapped_features.append(feature_name)
            continue

        weight = abs(shap_value)
        per_line_weight = weight / max(len(matched_lines), 1)
        feature_to_lines[feature_name] = matched_lines

        mapped_features.append(
            {
                "feature_name": feature_name,
                "source_feature_name": source_feature_name,
                "feature_type": item.get("feature_type") or "Unknown",
                "group": item.get("group"),
                "shap_value": shap_value,
                "corr_r": corr_r,
                "matched_lines": matched_lines,
            }
        )

        for line_no in matched_lines:
            line_scores[line_no] += per_line_weight
            line_to_features[line_no].append(
                {
                    "feature": feature_name,
                    "source_feature_name": source_feature_name,
                    "group": item.get("group"),
                    "shap_value": shap_value,
                    "corr_r": corr_r,
                    "direction": item.get("direction"),
                    "weight_added": per_line_weight,
                    "reason_text": _render_reason_text(
                        {
                            "feature_name": feature_name,
                            "source_feature_name": source_feature_name,
                            "shap_value": shap_value,
                            "corr_r": corr_r,
                        }
                    ),
                }
            )

    line_scores_list = [
        {
            "line_number": i,
            "code": code_lines[i - 1],
            "score": float(line_scores.get(i, 0.0)),
            "matched_features": sorted([entry["feature"] for entry in line_to_features.get(i, [])]),
            "reasons": [entry["reason_text"] for entry in line_to_features.get(i, [])],
        }
        for i in range(1, line_count + 1)
    ]

    rendered_html = render_code_with_feature_labels(
        code_text=code,
        mapped_features=mapped_features,
        predicted_class=predicted_label,
    )

    return {
        "predicted_label": predicted_label,
        "html": rendered_html,
        "line_scores": line_scores_list,
        "feature_to_lines": feature_to_lines,
        "line_to_features": dict(line_to_features),
        "unmapped_features": unmapped_features,
    }


def _safe_parse(code: str):
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _is_manual_feature(item: Dict[str, Any]) -> bool:
    if item.get("mapped_feature_name"):
        return True

    feature_type = str(item.get("feature_type", "")).strip().lower()
    return (
        ("software engineering" in feature_type)
        or ("software_engineering_features" in feature_type)
        or ("manual" in feature_type)
    )


def _build_ast_index(tree) -> Dict[str, Set[int]]:
    index: Dict[str, Set[int]] = defaultdict(set)
    if tree is None:
        return index

    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            continue

        if isinstance(node, ast.FunctionDef):
            index["FUNCTION_LINES"].add(lineno)
        if isinstance(node, ast.ClassDef):
            index["CLASS_LINES"].add(lineno)
        if isinstance(node, ast.Call):
            index["CALL_LINES"].add(lineno)
        if isinstance(node, ast.Assign):
            index["ASSIGN_LINES"].add(lineno)
        if isinstance(node, ast.Return):
            index["RETURN_LINES"].add(lineno)
        if isinstance(node, ast.If):
            index["IF_LINES"].add(lineno)
        if isinstance(node, ast.For):
            index["FOR_LINES"].add(lineno)
        if isinstance(node, ast.While):
            index["WHILE_LINES"].add(lineno)
        if isinstance(node, ast.Try):
            index["TRY_LINES"].add(lineno)
        if isinstance(node, ast.With):
            index["WITH_LINES"].add(lineno)
        if isinstance(node, ast.Import):
            index["IMPORT_LINES"].add(lineno)
        if isinstance(node, ast.ImportFrom):
            index["IMPORTFROM_LINES"].add(lineno)
        if isinstance(node, ast.Lambda):
            index["LAMBDA_LINES"].add(lineno)
        if isinstance(node, ast.ListComp):
            index["LISTCOMP_LINES"].add(lineno)

    return index


def _build_token_index(code_lines: List[str]) -> Dict[str, Set[int]]:
    index: Dict[str, Set[int]] = defaultdict(set)
    joined = "\n".join(code_lines)

    try:
        for tok in tokenize.generate_tokens(io.StringIO(joined).readline):
            tok_type = tok.type
            tok_str = tok.string
            tok_line = tok.start[0]
            if tok_type == tokenize.NAME:
                index[f"NAME::{tok_str}"].add(tok_line)
            elif tok_type == tokenize.STRING:
                index["STRING_LINES"].add(tok_line)
            elif tok_type == tokenize.NUMBER:
                index["NUMBER_LINES"].add(tok_line)
    except tokenize.TokenError:
        pass

    return index


def _compute_line_stats(code_lines: List[str]) -> Dict[str, Any]:
    lengths = {i + 1: len(line) for i, line in enumerate(code_lines)}
    nonempty_lines = [i + 1 for i, line in enumerate(code_lines) if line.strip()]
    empty_lines = [i + 1 for i, line in enumerate(code_lines) if not line.strip()]

    return {
        "lengths": lengths,
        "nonempty_lines": nonempty_lines,
        "empty_lines": empty_lines,
        "longest_lines": _top_n_by_value(lengths, 8),
    }


def _map_feature_to_lines(
    feature_name: str,
    code_lines: List[str],
    ast_index: Dict[str, Set[int]],
    token_index: Dict[str, Set[int]],
    stats: Dict[str, Any],
) -> Set[int]:
    name = feature_name.strip()
    lines: Set[int] = set()

    density_keyword_map = {
        "if_Density": "if", "elif_Density": "elif", "else_Density": "else",
        "for_Density": "for", "while_Density": "while", "try_Density": "try",
        "except_Density": "except", "return_Density": "return", "def_Density": "def",
        "lambda_Density": "lambda", "import_Density": "import", "from_Density": "from",
        "with_Density": "with", "break_Density": "break", "continue_Density": "continue",
        "pass_Density": "pass", "and_Density": "and", "or_Density": "or",
        "not_Density": "not", "in_Density": "in", "is_Density": "is",
        "as_Density": "as", "None_Density": "None", "True_Density": "True",
        "False_Density": "False",
    }

    ast_alias_map = {
        "nttf_Call": "CALL_LINES", "ntad_Call": "CALL_LINES",
        "nttf_Assign": "ASSIGN_LINES", "ntad_Assign": "ASSIGN_LINES",
        "nttf_Return": "RETURN_LINES", "ntad_Return": "RETURN_LINES",
        "nttf_If": "IF_LINES", "ntad_If": "IF_LINES",
        "nttf_For": "FOR_LINES", "ntad_For": "FOR_LINES",
        "nttf_While": "WHILE_LINES", "ntad_While": "WHILE_LINES",
        "nttf_Try": "TRY_LINES", "ntad_Try": "TRY_LINES",
        "nttf_With": "WITH_LINES", "ntad_With": "WITH_LINES",
        "nttf_FunctionDef": "FUNCTION_LINES", "ntad_FunctionDef": "FUNCTION_LINES",
        "nttf_Import": "IMPORT_LINES", "ntad_Import": "IMPORT_LINES",
        "nttf_ImportFrom": "IMPORTFROM_LINES", "ntad_ImportFrom": "IMPORTFROM_LINES",
        "nttf_Lambda": "LAMBDA_LINES", "ntad_Lambda": "LAMBDA_LINES",
        "nttf_ListComp": "LISTCOMP_LINES", "ntad_ListComp": "LISTCOMP_LINES",
    }

    if name in density_keyword_map:
        return set(token_index.get(f"NAME::{density_keyword_map[name]}", set()))
    if name in ast_alias_map:
        return set(ast_index.get(ast_alias_map[name], set()))
    if name in {"avgLineLength", "stdDevLineLength"}:
        return set(stats["longest_lines"])
    if name == "emptyLinesDensity":
        return set(stats["empty_lines"])
    if name == "whiteSpaceRatio":
        return {i for i, line in enumerate(code_lines, start=1) if line.startswith((" ", "\t"))}
    if name in {"avgFunctionLength", "avgParams", "stdDevNumParams", "numFunctionsDensity"}:
        return set(ast_index.get("FUNCTION_LINES", set()))
    if name == "numFunctionCallsDensity":
        return set(ast_index.get("CALL_LINES", set()))
    if name == "numClassesDensity":
        return set(ast_index.get("CLASS_LINES", set()))
    if name in {"numAssignmentStmtDensity", "numVariablesDensity"}:
        return set(ast_index.get("ASSIGN_LINES", set()))
    if name in {"cyclomaticComplexity", "maintainabilityIndex", "branchingFactor", "nestingDepth", "maxDecisionTokens"}:
        for key in ("IF_LINES", "FOR_LINES", "WHILE_LINES", "TRY_LINES"):
            lines |= set(ast_index.get(key, set()))
        return lines
 
    if name == "avgIdentifierLength":
        regex = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{6,}\b")
        return {i for i, line in enumerate(code_lines, start=1) if len(regex.findall(line)) >= 2}
    if name == "numKeywordsDensity":
        keywords = {
            "if", "elif", "else", "for", "while", "try", "except", "return",
            "def", "lambda", "import", "from", "with", "break", "continue",
            "pass", "and", "or", "not", "in", "is", "as", "None", "True", "False",
        }
        return {i for i, line in enumerate(code_lines, start=1) if sum(1 for t in re.findall(r"\b\w+\b", line) if t in keywords) >= 2}
    if name == "numLiteralsDensity":
        return set(token_index.get("STRING_LINES", set())) | set(token_index.get("NUMBER_LINES", set()))
    if name == "numInputStmtsDensity":
        regex = re.compile(r"\binput\s*\(")
        return {i for i, line in enumerate(code_lines, start=1) if regex.search(line)}

    return lines


def _top_n_by_value(mapping: Dict[int, int], n: int) -> List[int]:
    return [k for k, _ in sorted(mapping.items(), key=lambda x: x[1], reverse=True)[:n]]
