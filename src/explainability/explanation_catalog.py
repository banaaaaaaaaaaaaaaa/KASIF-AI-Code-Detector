from __future__ import annotations

from typing import Dict, List

GROUP_PANEL_INFO = (
    "This section shows which feature groups had the biggest impact on the prediction. "
    "Higher percentages mean stronger influence."
)

GROUP_MORE_DETAILS_URL = "/group-details"

GROUP_EXPLANATIONS = {
    "LEXICAL": "Lexical features look at the surface style of the code, like naming, line length, keywords, and literals.",
    "SYNTACTIC": "Syntactic features look at the code structure, such as loops, conditions, nesting, and AST patterns.",
    "STRUCTURAL": "Structural features look at how the code is organized, such as functions, classes, assignments, and statements.",
    "COMPLEXITY": "Complexity features show how complicated the logic is, especially in conditions, branches, and nested code.",
    "COMPLEXITY_MAINTAINABILITY": "These features describe how complex the code is and how easy it may be to read and maintain.",
    "HALSTEAD": "Halstead features measure the use of operators and operands to describe code complexity and coding style.",
    "EMBEDDING": "Embedding features are learned automatically by the model. They capture hidden patterns, but they are not directly human-readable.",
    "OTHER": "Other useful features that do not belong to one main group.",
}

FEATURE_EXPLANATIONS = {
    # Lexical
    "avgIdentifierLength": "The average length of names like variables, functions, and classes.",
    "avgLineLength": "The average number of characters in each line of code.",
    "avgFunctionLength": "The average number of lines inside each function.",
    "keywordsDensity": "How often programming keywords appear in the code.",
    "numKeywordsDensity": "How many different keywords are used across the code.",
    "numLiteralsDensity": "How often literal values like numbers and strings appear.",
    "stdDevLineLength": "How much line lengths change from one line to another.",

    # Syntactic / AST
    "ASTNodeTypesTF": "How often different AST node types appear in the code structure.",
    "ASTNodeTypeAvgDep": "The average depth of AST node types inside the code tree.",
    "branchingFactor": "How many branches each part of the code structure tends to create.",
    "maxDepthASTNode": "The deepest level reached in the AST structure.",
    "nestingDepth": "How deeply loops, conditions, or blocks are nested inside each other.",
    "maxDecisionTokens": "How complex the largest condition is, based on how many tokens it contains.",

    # Structural
    "numFunctionsDensity": "How often functions are defined in the code.",
    "numFunctionCallsDensity": "How often functions or methods are called.",
    "numClassesDensity": "How often classes are defined in the code.",
    "numAssignmentStmtDensity": "How often assignment statements are used.",
    "numStatementsDensity": "How many executable statements appear in the code.",
    "numVariablesDensity": "How many variables are assigned values.",
    "numInputStmtsDensity": "How often input-related statements are used.",
    "sloc": "The total number of source lines in the code, excluding blank lines and comments.",
    "avgParams": "The average number of parameters used in functions.",
    "stdDevNumParams": "How much the number of function parameters changes between functions.",

    # Complexity / Maintainability
    "cyclomaticComplexity": "A measure of how many different logic paths the code has.",
    "maintainabilityIndex": "A score that estimates how easy the code is to read and maintain.",

    # Halstead
    "numberOfDistinctOperands": "The number of different operands used, such as variables and constants.",
    "numberOfDistinctOperators": "The number of different operators used, such as +, -, =, and others.",
    "totalNumberOfOperands": "The total number of operand uses in the code.",
    "totalNumberOfOperators": "The total number of operator uses in the code.",

    # Common density/manual features
    "def_Density": "How often function definitions appear.",
    "for_Density": "How often for loops appear.",
    "if_Density": "How often if statements appear.",
    "while_Density": "How often while loops appear.",
    "elif_Density": "How often elif branches appear.",
    "else_Density": "How often else branches appear.",
    "break_Density": "How often break statements appear.",
    "continue_Density": "How often continue statements appear.",
    "return_Density": "How often return statements appear.",
    "yield_Density": "How often yield statements appear.",
    "try_Density": "How often try blocks appear.",
    "except_Density": "How often except blocks appear.",
    "import_Density": "How often import statements appear.",
    "from_Density": "How often from ... import statements appear.",
    "class_Density": "How often class definitions appear.",
    "lambda_Density": "How often lambda expressions appear.",
    "emptyLinesDensity": "How often empty lines appear in the code.",
    "whiteSpaceRatio": "How much spacing and indentation are used in the code.",
}


def get_feature_explanation(feature_name: str) -> str:
    name = str(feature_name or "").strip()
    if not name:
        return "No explanation available yet."

    if name in FEATURE_EXPLANATIONS:
        return FEATURE_EXPLANATIONS[name]

    if name.startswith("emb_"):
        return (
        "This is a learned embedding feature. "
        "It captures hidden code patterns and is linked to readable feature groups using SHAP-based correlation."
    )
    if name.startswith("nttf_"):
        nttf_explanations = {
            "nttf_Call": (
                "This feature shows how often the code calls functions or methods, "
                "such as input(), print(), len(), open(), or file.read()."
            ),

            "nttf_Name": (
                "This feature shows how often names appear in the code, "
                "such as variable names, function names, or imported names."
            ),

            "nttf_Return": (
                "This feature shows how often return statements appear in the code."
            ),

            "nttf_Tuple": (
                "This feature shows how often tuple structures appear in the code, "
                "such as values grouped with commas or parentheses."
            ),

            "nttf_List": (
                "This feature shows how often list structures appear in the code, "
                "such as [1, 2, 3] or empty lists []."
            ),

            "nttf_Dict": (
                "This feature shows how often dictionary structures appear in the code, "
                "such as {'key': value}."
            ),

            "nttf_Assign": (
                "This feature shows how often assignment statements appear in the code, "
                "such as x = 5 or total = total + 1."
            ),

            "nttf_AugAssign": (
                "This feature shows how often shortcut assignment statements appear, "
                "such as x += 1 or total -= value."
            ),

            "nttf_If": (
                "This feature shows how often if conditions appear in the code."
            ),

            "nttf_For": (
                "This feature shows how often for loops appear in the code."
            ),

            "nttf_While": (
                "This feature shows how often while loops appear in the code."
            ),

            "nttf_Compare": (
                "This feature shows how often comparison expressions appear, "
                "such as x > 5, a == b, or value != 0."
            ),

            "nttf_BoolOp": (
                "This feature shows how often boolean operations appear, "
                "such as and / or conditions."
            ),

            "nttf_BinOp": (
                "This feature shows how often arithmetic or binary operations appear, "
                "such as +, -, *, /, or %."
            ),

            "nttf_Expr": (
                "This feature shows how often expression statements appear, "
                "such as standalone function calls or calculations."
            ),

            "nttf_Attribute": (
                "This feature shows how often object attributes or methods are accessed, "
                "such as file.read, text.split, or item.value."
            ),

            "nttf_Subscript": (
                "This feature shows how often indexing or slicing is used, "
                "such as list[0], text[1:3], or dict['key']."
            ),

            "nttf_FunctionDef": (
                "This feature shows how often function definitions appear in the code."
            ),

            "nttf_Import": (
                "This feature shows how often normal import statements appear, "
                "such as import math."
            ),

            "nttf_ImportFrom": (
                "This feature shows how often from-import statements appear, "
                "such as from math import sqrt."
            ),

            "nttf_Try": (
                "This feature shows how often try-except error handling blocks appear."
            ),

            "nttf_ExceptHandler": (
                "This feature shows how often except blocks appear in the code."
            ),

            "nttf_With": (
                "This feature shows how often with statements appear, "
                "such as with open(...) as file."
            ),

            "nttf_Lambda": (
                "This feature shows how often lambda functions appear in the code."
            ),

            "nttf_ListComp": (
                "This feature shows how often list comprehensions appear, "
                "such as [x for x in items]."
            ),

            "nttf_GeneratorExp": (
                "This feature shows how often generator expressions appear, "
                "such as (x for x in items)."
            ),

            "nttf_comprehension": (
                "This feature shows how often comprehension parts appear, "
                "such as the for part inside a list comprehension."
            ),

            "nttf_keyword": (
                "This feature shows how often keyword arguments are used in function calls, "
                "such as print(value, end=' ') or open(file, mode='w')."
            ),

            "nttf_arguments": (
                "This feature shows how often function argument structures appear in the code."
            ),

            "nttf_Module": (
                "This feature represents the whole Python file structure."
            ),
            "nttf_IfExp": (
            "This feature shows how often inline if-else expressions appear in the code, "
            "such as x if condition else y."
            ),
            "nttf_arguments": (
            "This feature shows how often function parameter structures appear in the code, "
            "such as the parameters inside def calculate(x, y)."
        ),

        }

        if name in nttf_explanations:
            return nttf_explanations[name]

        suffix = name.replace("nttf_", "", 1).replace("_", " ")
        return f"This feature shows how often {suffix} structures appear in the code."

    if name.startswith("ntad_"):
        ntad_explanations = {
            "ntad_Module": (
                "This feature shows the depth of the whole Python file structure."
            ),

            "ntad_Assign": (
                "This feature shows how deeply assignment statements appear in the code, "
                "such as x = 5 or total = total + 1."
            ),

            "ntad_AugAssign": (
                "This feature shows how deeply shortcut assignment statements appear, "
                "such as x += 1 or total -= value."
            ),

            "ntad_While": (
                "This feature shows how deeply while loops appear in the code."
            ),

            "ntad_For": (
                "This feature shows how deeply for loops appear in the code."
            ),

            "ntad_If": (
                "This feature shows how deeply if statements appear in the code."
            ),

            "ntad_IfExp": (
                "This feature shows how deeply inline if-else expressions appear, "
                "such as x if condition else y."
            ),

            "ntad_Return": (
                "This feature shows how deeply return statements appear in the code."
            ),

            "ntad_Call": (
                "This feature shows how deeply function or method calls appear, "
                "such as input(), print(), len(), open(), or file.read()."
            ),

            "ntad_Name": (
                "This feature shows how deeply names appear in the code, "
                "such as variable names, function names, or imported names."
            ),

            "ntad_arguments": (
                "This feature shows how deeply function parameter structures appear, "
                "such as the parameters inside def calculate(x, y)."
            ),

            "ntad_keyword": (
                "This feature shows how deeply keyword arguments appear in function calls, "
                "such as print(value, end=' ') or open(file, mode='w')."
            ),

            "ntad_Lambda": (
                "This feature shows how deeply lambda functions appear in the code."
            ),

            "ntad_Try": (
                "This feature shows how deeply try-except error handling blocks appear."
            ),

            "ntad_Import": (
                "This feature shows how deeply normal import statements appear, "
                "such as import math."
            ),

            "ntad_ImportFrom": (
                "This feature shows how deeply from-import statements appear, "
                "such as from math import sqrt."
            ),

            "ntad_BinOp": (
                "This feature shows how deeply arithmetic or binary operations appear, "
                "such as +, -, *, /, or %."
            ),

            "ntad_BoolOp": (
                "This feature shows how deeply boolean operations appear, "
                "such as and / or conditions."
            ),

            "ntad_UnaryOp": (
                "This feature shows how deeply unary operations appear, "
                "such as -x or not condition."
            ),

            "ntad_Compare": (
                "This feature shows how deeply comparison expressions appear, "
                "such as x > 5, a == b, or value != 0."
            ),

            "ntad_Attribute": (
                "This feature shows how deeply object attributes or methods are accessed, "
                "such as file.read, text.split, or item.value."
            ),

            "ntad_Expr": (
                "This feature shows how deeply expression statements appear, "
                "such as standalone function calls or calculations."
            ),

            "ntad_Tuple": (
                "This feature shows how deeply tuple structures appear in the code, "
                "such as values grouped with commas or parentheses."
            ),

            "ntad_List": (
                "This feature shows how deeply list structures appear in the code, "
                "such as [1, 2, 3] or empty lists []."
            ),

            "ntad_Dict": (
                "This feature shows how deeply dictionary structures appear in the code, "
                "such as {'key': value}."
            ),

            "ntad_Slice": (
                "This feature shows how deeply slicing appears in the code, "
                "such as text[1:3] or items[:5]."
            ),

            "ntad_ListComp": (
                "This feature shows how deeply list comprehensions appear, "
                "such as [x for x in items]."
            ),

            "ntad_SetComp": (
                "This feature shows how deeply set comprehensions appear, "
                "such as {x for x in items}."
            ),

            "ntad_GeneratorExp": (
                "This feature shows how deeply generator expressions appear, "
                "such as (x for x in items)."
            ),

            "ntad_comprehension": (
                "This feature shows how deeply comprehension parts appear, "
                "such as the for part inside a list comprehension."
            ),

            "ntad_Starred": (
                "This feature shows how deeply starred expressions appear, "
                "such as *items or *args."
            ),

            "ntad_FunctionDef": (
                "This feature shows how deeply function definitions appear in the code."
            ),
        }

        if name in ntad_explanations:
            return ntad_explanations[name]

        suffix = name.replace("ntad_", "", 1).replace("_", " ")
        return f"This feature shows how deeply {suffix} structures appear in the code."

    if name.endswith("_Density"):
        suffix = name.replace("_Density", "").replace("_", " ")
        return f"A density feature showing how often '{suffix}' appears in the code."

    return "Explanation will be added later."


def get_group_explanation(group_name: str) -> str:
    name = str(group_name or "").strip()
    if not name:
        return "No group explanation available yet."
    return GROUP_EXPLANATIONS.get(name, "Explanation will be added later.")


def attach_feature_explanations(features: List[dict]) -> List[dict]:
    enriched: List[dict] = []

    for item in features or []:
        if not isinstance(item, dict):
            continue

        feature_name = (
            item.get("display_name")
            or item.get("feature")
            or item.get("feature_name")
            or item.get("name")
            or ""
        )

        enriched_item = dict(item)
        enriched_item["explanation"] = get_feature_explanation(str(feature_name))
        enriched.append(enriched_item)

    return enriched


def attach_group_explanations(groups: List[dict]) -> List[dict]:
    enriched: List[dict] = []

    for item in groups or []:
        if not isinstance(item, dict):
            continue

        group_name = str(item.get("group") or "OTHER")
        enriched_item = dict(item)
        enriched_item["explanation"] = get_group_explanation(group_name)
        enriched.append(enriched_item)

    return enriched


def get_group_panel_info() -> str:
    return GROUP_PANEL_INFO


def get_group_more_details_url() -> str:
    return GROUP_MORE_DETAILS_URL