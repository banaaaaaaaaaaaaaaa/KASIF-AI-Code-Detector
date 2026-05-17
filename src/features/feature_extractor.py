from __future__ import annotations

import ast
import io
import json
import keyword
import math
import statistics
import sys
import tokenize
import warnings
from collections import Counter, deque
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from radon.metrics import ComplexityVisitor, mi_visit
from radon.raw import analyze

warnings.filterwarnings("ignore", category=SyntaxWarning)

NAN = math.nan
DEFAULT_VALUES = [NAN, -1, -1.0]


class Logger:
    log_file_path = Path(__file__).resolve().parent / "log.log"

    @classmethod
    def error_log(cls, message: str) -> None:
        cls.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cls.log_file_path, "a", encoding="utf-8") as error_f:
            error_f.write(f"Error: {message}\n")

    @classmethod
    def info_log(cls, message: str) -> None:
        cls.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cls.log_file_path, "a", encoding="utf-8") as info_f:
            info_f.write(f"Info: {message}\n")


class CodeMetricsExtractor:
    def __init__(self, code: str):
        self.code = str(code or "")
        self.code_lines = self.code.splitlines()
        self.code_length = len(self.code)

        # library-independent
        self.line_length = self._get_line_length()
        self.whitespaces = self._get_num_whitespaces()
        self.empty_lines = self._get_empty_lines()

        # ast-dependent
        self.tree = self._get_ast()
        self.num_functions = self._get_num_functions()
        self.functions_params = self._get_functions_params()

        # tokenize-dependent
        self.tokens = self._get_tokens()
        self.keywords = self._get_keywords()

        # radon-dependent
        self.radon_raw_metrics = self._get_radon_metrics()

        self.metrics: Dict[str, Any] = {}
        if self.code_length == 0 or not self.code.strip():
            Logger.error_log("File is empty or contains only whitespaces")
        elif self.tree is None:
            Logger.error_log("Failed to parse code to AST.")
        elif self.tokens is None:
            Logger.error_log("Failed to tokenize code.")
        elif self.radon_raw_metrics is None:
            Logger.error_log("Failed to calculate Radon metrics.")
        else:
            self._get_library_independent_metrics()
            self._get_ast_dependent_metrics()
            self._get_tokenize_dependent_metrics()
            self._get_radon_dependent_metrics()

    def _get_library_independent_metrics(self) -> None:
        self.metrics.update(
            {
                "avgLineLength": self._get_avg_line_length(),
                "stdDevLineLength": self._get_line_length_stdev(),
                "whiteSpaceRatio": self._get_whitespace_ratio(),
            }
        )

    def _get_ast_dependent_metrics(self) -> None:
        if self.tree is not None:
            self.metrics.update(
                {
                    "maxDecisionTokens": self._get_max_decision_tokens(),
                    "numLiteralsDensity": self._get_literals_density(),
                    "nestingDepth": self._get_max_nesting_depth(),
                    "maxDepthASTNode": self._get_max_ast_node_depth(),
                    "branchingFactor": self._get_branching_factor(),
                    "avgParams": self._get_avg_func_params(),
                    "stdDevNumParams": self._get_func_params_stdev(),
                    "avgFunctionLength": self._get_avg_function_length(),
                    "avgIdentifierLength": self._get_avg_identifier_length(),
                }
            )
            self.metrics.update(self._get_node_type_term_frequency())
            self.metrics.update(self._get_node_type_avg_depth())

    def _get_tokenize_dependent_metrics(self) -> None:
        if self.tokens is not None:
            self.metrics.update({"numKeywordsDensity": self._get_num_keywords_density()})
            self.metrics.update(self._get_keywords_density())

    def _get_radon_dependent_metrics(self) -> None:
        if self.radon_raw_metrics is not None:
            self.metrics.update(
                {
                    "sloc": self.radon_raw_metrics.sloc,
                    "numVariablesDensity": self._get_num_variables_density(),
                    "numFunctionsDensity": self._get_functions_density(),
                    "numInputStmtsDensity": self._get_input_statements_density(),
                    "numAssignmentStmtDensity": self._get_assignment_statements_density(),
                    "numFunctionCallsDensity": self._get_function_calls_density(),
                    "numStatementsDensity": self._get_num_statements_density(),
                    "numClassesDensity": self._get_num_classes_density(),
                    "emptyLinesDensity": self._get_empty_lines_density(),
                    "cyclomaticComplexity": self._get_radon_cyclomatic_complexity(),
                    "maintainabilityIndex": self._get_radon_maintainability_index(),
                }
            )

    def _get_ast(self):
        try:
            return ast.parse(self.code)
        except SyntaxError:
            Logger.error_log("Failed to parse code to AST. Returning None for AST.")
            return None

    def _get_tokens(self):
        try:
            return list(tokenize.tokenize(BytesIO(self.code.encode("utf-8")).readline))
        except tokenize.TokenError:
            Logger.error_log("Failed to tokenize code. Returning None for tokens.")
            return None

    def _get_keywords(self):
        if self.tokens is None:
            Logger.error_log("Failed to tokenize code. Returning None for keywords.")
            return None

        keywords_found: Dict[str, int] = {}
        comment_state = False

        try:
            for token in self.tokens:
                if token.type == tokenize.COMMENT:
                    comment_state = True
                elif token.type in (tokenize.NL, tokenize.NEWLINE):
                    comment_state = False
                elif token.string in keyword.kwlist and not comment_state:
                    keywords_found[token.string] = keywords_found.get(token.string, 0) + 1
        except Exception:
            Logger.error_log("Failed while reading tokens. Returning empty dictionary for keywords.")
            return {}

        return keywords_found

    def _get_radon_metrics(self):
        try:
            return analyze(self.code)
        except Exception:
            Logger.error_log("Failed to calculate Radon metrics. Returning None.")
            return None

    def _get_line_length(self) -> List[int]:
        return [len(line) for line in self.code.split("\n")]

    def _get_num_whitespaces(self) -> int:
        return sum(1 for char in self.code if char.isspace())

    def _get_empty_lines(self) -> int:
        return sum(1 for line in self.code.splitlines() if line.strip() == "")

    def _get_avg_line_length(self):
        if len(self.line_length) == 0:
            Logger.error_log(f"No code lines. Returning default value: {DEFAULT_VALUES[0]} for avg_line_length")
            return DEFAULT_VALUES[0]
        return round(sum(self.line_length) / len(self.line_length), 2)

    def _get_line_length_stdev(self):
        return round(statistics.stdev(self.line_length), 2) if len(self.line_length) > 1 else 0

    def _get_whitespace_ratio(self):
        non_whitespace_chars = self.code_length - self.whitespaces
        if non_whitespace_chars == 0:
            Logger.error_log(
                f"File contains only whitespaces. Returning default value: {DEFAULT_VALUES[0]} for whitespace_ratio"
            )
            return DEFAULT_VALUES[0]
        return round(self.whitespaces / non_whitespace_chars, 2) if non_whitespace_chars > 0 else 0.0

    def _get_max_decision_tokens(self):
        decision_path_tokens = []

        try:
            for node in ast.walk(self.tree):
                if isinstance(node, (ast.If, ast.For, ast.While)):
                    if isinstance(node, (ast.If, ast.While)):
                        condition = ast.get_source_segment(self.code, node.test)
                    else:
                        condition = ast.get_source_segment(self.code, node)
                    tokens = self._get_for_loop_tokens(condition or "")
                    decision_path_tokens.append(tokens)

            if decision_path_tokens:
                return max(len(tokens) for tokens in decision_path_tokens)
            return 0
        except Exception as e:
            Logger.error_log(
                f"Failed to get max_decision_tokens: {e}. Returning default value: {DEFAULT_VALUES[0]}"
            )
            return DEFAULT_VALUES[0]

    def _get_for_loop_tokens(self, condition: str):
        split = 1
        while True:
            try:
                split_parts = condition.split(":", split)
                if split == 1:
                    condition_split = split_parts[0]
                else:
                    condition_split = ":".join(split_parts[:split])

                tokens = tokenize.tokenize(io.BytesIO(condition_split.encode("utf-8")).readline)
                return [
                    token.string.strip()
                    for token in tokens
                    if token.string.strip() and token.string.strip() not in ("if", "while", "for", "utf-8")
                ]
            except tokenize.TokenError:
                if ":" in condition[split + 1:]:
                    split += 1
                else:
                    return []

    def _get_literals_density(self):
        if self.tree is None:
            Logger.error_log(f"Failed to parse code to AST. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]
        if self.radon_raw_metrics is None:
            Logger.error_log(f"Failed to calculate Radon metrics. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]
        if self.radon_raw_metrics.sloc == 0:
            Logger.error_log(f"File is empty. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]

        literals_sum = sum(1 for node in ast.walk(self.tree) if isinstance(node, ast.Constant))
        return round(literals_sum / self.radon_raw_metrics.sloc, 2)

    def _get_num_functions(self):
        if self.tree is None:
            Logger.error_log(f"Failed to parse code to AST. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]
        return sum(isinstance(node, ast.FunctionDef) for node in ast.walk(self.tree))

    def _get_functions_params(self):
        if self.tree is None:
            Logger.error_log(f"Failed to parse code to AST. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]

        arguments_per_function = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                arguments_per_function.append(len(node.args.args))
        return arguments_per_function

    def _get_max_nesting_depth(self):
        if self.tree is None:
            Logger.error_log(f"Failed to parse code to AST. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]

        max_nesting_depth = 0
        queue = deque([(self.tree, 0)])

        while queue:
            node, depth = queue.popleft()
            if isinstance(node, (ast.If, ast.While, ast.For, ast.FunctionDef)):
                max_nesting_depth = max(max_nesting_depth, depth)
            for child_node in ast.iter_child_nodes(node):
                queue.append((child_node, depth + 1))

        return max_nesting_depth

    def _get_branching_factor(self):
        if self.tree is None:
            Logger.error_log(f"Failed to parse code to AST. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]

        queue = deque([self.tree])
        branches = []

        while queue:
            current_node = queue.popleft()
            current_node_branches = sum(1 for _ in ast.iter_child_nodes(current_node))
            if current_node_branches > 0:
                branches.append(current_node_branches)
            queue.extend(ast.iter_child_nodes(current_node))

        total_branches = sum(branches)
        total_parent_nodes = len(branches)
        return round(total_branches / total_parent_nodes, 2) if total_parent_nodes != 0 else 0

    def _get_avg_func_params(self):
        if self.functions_params == DEFAULT_VALUES[0]:
            Logger.error_log(f"Failed to parse code to AST. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]
        return round(sum(self.functions_params) / len(self.functions_params), 2) if len(self.functions_params) > 0 else 0.0

    def _get_func_params_stdev(self):
        if self.functions_params == DEFAULT_VALUES[0]:
            Logger.error_log(f"Failed to parse code to AST. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]
        return round(statistics.stdev(self.functions_params), 2) if len(self.functions_params) > 1 else 0

    def _get_max_ast_node_depth(self):
        if self.tree is None:
            Logger.error_log(f"Failed to parse code to AST. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]

        max_depth = 0
        queue = deque([(self.tree, 0)])

        while queue:
            current_node, depth = queue.popleft()
            max_depth = max(max_depth, depth)
            for child_node in ast.iter_child_nodes(current_node):
                queue.append((child_node, depth + 1))

        return max_depth

    def _get_input_statements_density(self):
        if self.tree is None or self.radon_raw_metrics is None or self.radon_raw_metrics.sloc == 0:
            Logger.error_log(f"Cannot calculate num_input_statements. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]

        num_input_statements = sum(
            isinstance(node, ast.Call) and hasattr(node.func, "id") and node.func.id == "input"
            for node in ast.walk(self.tree)
        )
        return round(num_input_statements / self.radon_raw_metrics.sloc, 2)

    def _get_assignment_statements_density(self):
        if self.radon_raw_metrics is None or self.radon_raw_metrics.sloc == 0:
            Logger.error_log(f"Cannot calculate num_assignment_statements. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]

        assignment_statements = sum(isinstance(node, ast.Assign) for node in ast.walk(self.tree))
        return round(assignment_statements / self.radon_raw_metrics.sloc, 2)

    def _get_function_calls_density(self):
        function_calls = sum(isinstance(node, ast.Call) for node in ast.walk(self.tree))
        if self.radon_raw_metrics is None or self.radon_raw_metrics.sloc == 0:
            Logger.error_log(f"Cannot calculate num_function_calls. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]
        return round(function_calls / self.radon_raw_metrics.sloc, 2)

    def _get_num_statements_density(self):
        if self.radon_raw_metrics is None or self.radon_raw_metrics.sloc == 0:
            Logger.error_log(f"Cannot calculate num_statements. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]

        num_statements = sum(isinstance(node, ast.stmt) for node in ast.walk(self.tree))
        return round(num_statements / self.radon_raw_metrics.sloc, 2)

    def _get_avg_function_length(self):
        function_lengths = []

        if self.tree is None:
            Logger.error_log(f"Failed to parse code to AST. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]

        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                end_lineno = getattr(node, "end_lineno", node.lineno)
                function_length = end_lineno - node.lineno
                function_lengths.append(function_length)

        return round(sum(function_lengths) / len(function_lengths), 2) if len(function_lengths) > 0 else 0.0

    def _get_avg_identifier_length(self):
        identifiers = set()

        if self.tree is None:
            Logger.error_log(f"Failed to parse code to AST. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)

        lengths = [len(identifier) for identifier in identifiers]
        return round(sum(lengths) / len(lengths), 2) if len(lengths) > 0 else 0.0

    def _get_num_classes_density(self):
        if self.radon_raw_metrics is None or self.radon_raw_metrics.sloc == 0:
            Logger.error_log(f"Cannot calculate num_classes_density. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]

        num_classes = sum(isinstance(node, ast.ClassDef) for node in ast.walk(self.tree))
        return round(num_classes / self.radon_raw_metrics.sloc, 2)

    def _get_node_type_term_frequency(self):
        term_frequency = {}
        if self.tree is None:
            Logger.error_log("Failed to parse code to AST. Returning empty dictionary for node_type_term_frequency")
            return term_frequency

        node_types = [node.__class__.__name__ for node in ast.walk(self.tree) if list(ast.iter_child_nodes(node))]
        frequency = Counter(node_types)
        return {"nttf_" + key: value for key, value in frequency.items()}

    def _get_node_type_avg_depth(self):
        if self.tree is None:
            Logger.error_log("Failed to parse code to AST. Returning empty dictionary for node_type_avg_depth")
            return {}

        node_queue = deque([(self.tree, 0)])
        depth_dict = {}

        while node_queue:
            current_node, depth = node_queue.popleft()
            node_type = type(current_node).__name__
            depth_dict.setdefault(node_type, []).append(depth)

            for child_node in ast.iter_child_nodes(current_node):
                if list(ast.iter_child_nodes(child_node)):
                    node_queue.append((child_node, depth + 1))

        average_type_depths = {
            node_type: round(sum(depths) / len(depths), 2)
            for node_type, depths in depth_dict.items()
        }
        return {"ntad_" + key: value for key, value in average_type_depths.items()}

    def _get_num_keywords_density(self):
        if self.radon_raw_metrics is None or self.radon_raw_metrics.sloc == 0:
            Logger.error_log(f"Cannot calculate keywords_density. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]

        keywords_sum = sum(self.keywords.values()) if self.keywords else 0
        return round(keywords_sum / self.radon_raw_metrics.sloc, 2)

    def _get_keywords_density(self):
        if self.radon_raw_metrics is None or self.radon_raw_metrics.sloc == 0:
            Logger.error_log("Cannot calculate keywords_density. Returning empty dictionary.")
            return {}

        keywords_density = {
            key: round(value / self.radon_raw_metrics.sloc, 2)
            for key, value in (self.keywords or {}).items()
        }
        return {key + "_Density": value for key, value in sorted(keywords_density.items(), key=lambda item: item[0])}

    def _get_radon_cyclomatic_complexity(self):
        try:
            cc = ComplexityVisitor.from_code(self.code)
            return cc.total_complexity
        except Exception:
            Logger.error_log(
                f"Failed to calculate Cyclomatic Complexity. Returning default value: {DEFAULT_VALUES[0]}"
            )
            return DEFAULT_VALUES[0]

    def _get_radon_maintainability_index(self):
        try:
            return mi_visit(self.code, False)
        except Exception:
            Logger.error_log(
                f"Failed to calculate Maintainability Index. Returning default value: {DEFAULT_VALUES[0]}"
            )
            return DEFAULT_VALUES[0]

    def _get_empty_lines_density(self):
        if self.radon_raw_metrics is None or self.radon_raw_metrics.sloc == 0:
            Logger.error_log(f"Cannot calculate empty_lines_density. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]
        return round(self.empty_lines / self.radon_raw_metrics.sloc, 2)

    def _get_functions_density(self):
        if self.radon_raw_metrics is None or self.radon_raw_metrics.sloc == 0:
            Logger.error_log(f"Cannot calculate functions_density. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]
        return round(self.num_functions / self.radon_raw_metrics.sloc, 2)

    def _get_num_variables_density(self):
        variables = set()

        if self.tree is None or self.radon_raw_metrics is None or self.radon_raw_metrics.sloc == 0:
            Logger.error_log(f"Cannot calculate num_variables_density. Returning default value: {DEFAULT_VALUES[0]}")
            return DEFAULT_VALUES[0]

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.add(target.id)

        return round(len(variables) / self.radon_raw_metrics.sloc, 2)


def get_uniform_metrics(metrics_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique_metrics = set()
    for metrics in metrics_list:
        unique_metrics.update(metrics.keys())

    for metrics in metrics_list:
        metrics.setdefault("filename", "unknown")
        metrics.setdefault("label", 0)
        for key in unique_metrics:
            if key not in metrics:
                metrics[key] = 0

    ordered_keys = ["filename", "label"] + sorted(
        key for key in unique_metrics if key not in {"filename", "label"}
    )
    return [{key: metrics[key] for key in ordered_keys} for metrics in metrics_list]


def save_metrics_to_csv(metrics: List[Dict[str, Any]], file_path: str) -> None:
    out_path = Path(file_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving metrics to {out_path}")
    if not metrics:
        print("No metrics to write.", file=sys.stderr)
        return

    import csv

    with open(out_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    if out_path.exists():
        print(f"Wrote {len(metrics)} rows to {out_path} (size: {out_path.stat().st_size} bytes)")
    else:
        print(f"Write failed: {out_path} not found after to_csv", file=sys.stderr)


def extract_metrics_from_jsonl(jsonl_path: str, output_file: str | None = None):
    metrics_list = []
    jsonl_path_obj = Path(jsonl_path).expanduser().resolve()

    if not jsonl_path_obj.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path_obj}")

    with open(jsonl_path_obj, "r", encoding="utf-8") as input_file:
        for i, line in enumerate(input_file):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
                code = item.get("code")
                if code is None:
                    code = item.get("content", "")

                identifier = item.get("filename") or f"sample_{i}"
                label = item.get("label", 0)

                extracted = CodeMetricsExtractor(code)
                metrics = extracted.metrics

                if metrics and isinstance(metrics, dict):
                    metrics["filename"] = identifier
                    metrics["label"] = label
                    metrics_list.append(metrics)
                else:
                    Logger.error_log(f"Line {i}: Metrics not extracted properly.")
            except Exception as e:
                Logger.error_log(f"Line {i} error: {e}")
                continue

    uniform_metrics = get_uniform_metrics(metrics_list)

    if output_file:
        save_metrics_to_csv(uniform_metrics, output_file)

    return uniform_metrics
