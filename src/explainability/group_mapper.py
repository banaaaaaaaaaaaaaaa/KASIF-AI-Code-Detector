from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


_MAPPING_CACHE: Dict[str, Dict[str, str]] = {}
_MAPPING_DF_CACHE: Dict[str, pd.DataFrame] = {}


LEXICAL_SET = {
    "avgIdentifierLength", "avgLineLength", "avgFunctionLength",
    "numKeywordsDensity", "numLiteralsDensity", "stdDevLineLength",
    "whiteSpaceRatio", "emptyLinesDensity", "False_Density",
    "None_Density", "True_Density", "and_Density", "as_Density",
    "break_Density", "continue_Density", "def_Density", "elif_Density",
    "else_Density", "except_Density", "for_Density", "from_Density",
    "if_Density", "import_Density", "in_Density", "is_Density",
    "lambda_Density", "nonlocal_Density", "not_Density", "or_Density",
    "pass_Density", "return_Density", "try_Density", "while_Density",
    "with_Density",
}

SYNTACTIC_CORE_SET = {
    "branchingFactor", "maxDepthASTNode", "nestingDepth", "maxDecisionTokens",
    "nttf_Assign", "nttf_Attribute", "nttf_AugAssign", "nttf_BinOp",
    "nttf_BoolOp", "nttf_Call", "nttf_Compare", "nttf_Dict",
    "nttf_ExceptHandler", "nttf_Expr", "nttf_For", "nttf_FormattedValue",
    "nttf_FunctionDef", "nttf_GeneratorExp", "nttf_If", "nttf_IfExp",
    "nttf_Import", "nttf_ImportFrom", "nttf_JoinedStr", "nttf_Lambda",
    "nttf_List", "nttf_ListComp", "nttf_Match", "nttf_MatchSequence",
    "nttf_MatchValue", "nttf_Module", "nttf_Name", "nttf_Return",
    "nttf_Slice", "nttf_Starred", "nttf_Subscript", "nttf_Try",
    "nttf_Tuple", "nttf_UnaryOp", "nttf_While", "nttf_arguments",
    "nttf_comprehension", "nttf_keyword", "nttf_match_case",
    "ntad_Assign", "ntad_Attribute", "ntad_AugAssign", "ntad_BinOp",
    "ntad_BoolOp", "ntad_Call", "ntad_Compare", "ntad_Dict",
    "ntad_ExceptHandler", "ntad_Expr", "ntad_For", "ntad_FormattedValue",
    "ntad_FunctionDef", "ntad_GeneratorExp", "ntad_If", "ntad_IfExp",
    "ntad_Import", "ntad_ImportFrom", "ntad_JoinedStr", "ntad_Lambda",
    "ntad_List", "ntad_ListComp", "ntad_Match", "ntad_MatchSequence",
    "ntad_MatchValue", "ntad_Module", "ntad_Name", "ntad_Return",
    "ntad_Slice", "ntad_Starred", "ntad_Subscript", "ntad_Try",
    "ntad_Tuple", "ntad_UnaryOp", "ntad_While", "ntad_arguments",
    "ntad_comprehension", "ntad_keyword", "ntad_match_case",
    "ntad_With", "ntad_withitem", "nttf_With", "nttf_withitem",
}

STRUCTURAL_SET = {
    "numFunctionsDensity", "numFunctionCallsDensity", "numClassesDensity",
    "numAssignmentStmtDensity", "numStatementsDensity", "numVariablesDensity",
    "numInputStmtsDensity", "sloc", "avgParams", "stdDevNumParams",
}

COMPLEXITY_MAINTAINABILITY_SET = {
    "cyclomaticComplexity",
    "maintainabilityIndex",
}

HALSTEAD_SET = {
    "numberOfDistinctOperands", "numberOfDistinctOperators",
    "totalNumberOfOperands", "totalNumberOfOperators",
}


def _get_explainability_dir() -> Path:
    return Path(__file__).resolve().parent


def _normalize_label(predicted_label: str) -> str:
    label = str(predicted_label or "").strip().lower()

    if label in {"ai", "ai-generated", "generated", "machine", "1"}:
        return "ai"

    if label in {"human", "human-written", "0"}:
        return "human"

    raise ValueError(f"Unsupported predicted label: {predicted_label}")


def _get_mapping_path(predicted_label: str) -> Path:
    normalized = _normalize_label(predicted_label)

    if normalized == "ai":
        return _get_explainability_dir() / "embedding_assigned_by_group_ai.csv"

    return _get_explainability_dir() / "embedding_assigned_by_group_human.csv"


def _clean_optional_value(value: Any) -> Optional[Any]:
    if value is None:
        return None

    text = str(value).strip()

    if text.upper() in {"", "NAN", "NONE", "NULL", "UNMAPPED"}:
        return None

    return value


def _clean_group(value: Any) -> Optional[str]:
    cleaned = _clean_optional_value(value)

    if cleaned is None:
        return None

    group = str(cleaned).strip().upper()

    if group in {"", "NAN", "NONE", "NULL", "UNMAPPED"}:
        return None

    return group


def _clean_number(value: Any) -> Optional[float]:
    cleaned = _clean_optional_value(value)

    if cleaned is None:
        return None

    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def _clean_int(value: Any) -> Optional[int]:
    cleaned = _clean_optional_value(value)

    if cleaned is None:
        return None

    try:
        return int(float(cleaned))
    except (TypeError, ValueError):
        return None


def _load_embedding_assignment_df(predicted_label: str) -> pd.DataFrame:
    """
    Load the embedding assignment CSV as a DataFrame.

    This keeps all columns, not only:
        embedding -> assigned_group

    Useful columns include:
        embedding
        assigned_group
        group_corr_sum
        n_pairs
        max_corr
        group_rank
        group_support

    Some columns may not exist depending on the CSV version.
    """
    csv_path = _get_mapping_path(predicted_label)
    cache_key = str(csv_path)

    if cache_key in _MAPPING_DF_CACHE:
        return _MAPPING_DF_CACHE[cache_key]

    if not csv_path.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]

    required_columns = {"embedding", "assigned_group"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns {missing} in {csv_path.name}")

    df["embedding"] = df["embedding"].astype(str).str.strip().str.lower()
    df["assigned_group"] = df["assigned_group"].astype(str).str.strip().str.upper()

    _MAPPING_DF_CACHE[cache_key] = df

    return df


def _load_mapping_table(predicted_label: str) -> Dict[str, str]:
    """
    Load only the simple mapping:
        embedding -> assigned_group

    This is kept for compatibility with map_embedding_to_group().
    """
    csv_path = _get_mapping_path(predicted_label)
    cache_key = str(csv_path)

    if cache_key in _MAPPING_CACHE:
        return _MAPPING_CACHE[cache_key]

    df = _load_embedding_assignment_df(predicted_label)

    mapping_dict = dict(zip(df["embedding"], df["assigned_group"]))
    _MAPPING_CACHE[cache_key] = mapping_dict

    return mapping_dict


def is_embedding_feature(feature_name: str) -> bool:
    return str(feature_name or "").strip().lower().startswith("emb_")


def map_embedding_to_group_info(
    feature_name: str,
    predicted_label: str,
) -> Dict[str, Optional[Any]]:
    """
    Return embedding group assignment plus evidence from the CSV.

    Example output:
    {
        "feature": "emb_593",
        "feature_type": "embedding",
        "group": "SYNTACTIC",
        "group_corr_sum": 2.1065,
        "n_pairs": 4,
        "max_corr": 0.5506,
        "group_rank": 3,
        "group_support": 1.0,
    }
    """
    df = _load_embedding_assignment_df(predicted_label)

    if df.empty:
        return {
            "feature": feature_name,
            "feature_type": "embedding",
            "group": None,
            "group_corr_sum": None,
            "n_pairs": None,
            "max_corr": None,
            "group_rank": None,
            "group_support": None,
        }

    key = str(feature_name or "").strip().lower()
    row = df[df["embedding"] == key]

    if row.empty:
        return {
            "feature": feature_name,
            "feature_type": "embedding",
            "group": None,
            "group_corr_sum": None,
            "n_pairs": None,
            "max_corr": None,
            "group_rank": None,
            "group_support": None,
        }

    item = row.iloc[0]

    return {
        "feature": feature_name,
        "feature_type": "embedding",
        "group": _clean_group(item.get("assigned_group")),
        "group_corr_sum": _clean_number(item.get("group_corr_sum")),
        "n_pairs": _clean_int(item.get("n_pairs")),
        "max_corr": _clean_number(item.get("max_corr")),
        "group_rank": _clean_int(item.get("group_rank")),
        "group_support": _clean_number(item.get("group_support")),
    }


def map_embedding_to_group(
    embedding_name: str,
    predicted_label: str,
) -> Optional[str]:
    """
    Backward-compatible helper.

    Returns only the group name for an embedding feature.
    """
    embedding_info = map_embedding_to_group_info(embedding_name, predicted_label)
    return embedding_info.get("group")


def map_manual_feature_to_group(feature_name: str) -> Optional[str]:
    feature_name = str(feature_name or "").strip()

    if feature_name in LEXICAL_SET:
        return "LEXICAL"

    if feature_name in SYNTACTIC_CORE_SET:
        return "SYNTACTIC"

    if feature_name in STRUCTURAL_SET:
        return "STRUCTURAL"

    if feature_name in COMPLEXITY_MAINTAINABILITY_SET:
        return "COMPLEXITY_MAINTAINABILITY"

    if feature_name in HALSTEAD_SET:
        return "HALSTEAD"

    return None


def map_any_feature_to_group(
    feature_name: str,
    predicted_label: str,
) -> Dict[str, Optional[Any]]:
    """
    Map any feature to its feature type and group.

    Manual feature output:
    {
        "feature": "sloc",
        "feature_type": "manual",
        "group": "STRUCTURAL",
        "group_corr_sum": None,
        "n_pairs": None,
        "max_corr": None,
        "group_rank": None,
        "group_support": None,
    }

    Embedding feature output:
    {
        "feature": "emb_593",
        "feature_type": "embedding",
        "group": "SYNTACTIC",
        "group_corr_sum": 2.1065,
        "n_pairs": 4,
        "max_corr": 0.5506,
        "group_rank": 3,
        "group_support": 1.0,
    }
    """
    feature_name = str(feature_name or "").strip()

    if is_embedding_feature(feature_name):
        return map_embedding_to_group_info(feature_name, predicted_label)

    return {
        "feature": feature_name,
        "feature_type": "manual",
        "group": map_manual_feature_to_group(feature_name),
        "group_corr_sum": None,
        "n_pairs": None,
        "max_corr": None,
        "group_rank": None,
        "group_support": None,
    }