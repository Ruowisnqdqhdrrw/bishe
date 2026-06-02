from __future__ import annotations

from dataclasses import dataclass
import re
from io import BytesIO
from xml.sax.saxutils import escape

import matplotlib
from PIL import Image


matplotlib.use("Agg")
from matplotlib import pyplot as plt

_FUNCTION_NAMES = ("sqrt", "sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "int")
_GREEK_NAMES = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "theta",
    "lambda",
    "mu",
    "pi",
    "sigma",
    "phi",
    "omega",
)
_GREEK_CHAR_MAP = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "θ": "theta",
    "λ": "lambda",
    "μ": "mu",
    "π": "pi",
    "σ": "sigma",
    "φ": "phi",
    "ω": "omega",
}
_GREEK_XML_MAP = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ϵ",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "pi": "π",
    "sigma": "σ",
    "phi": "φ",
    "omega": "ω",
}
_LATEX_FUNCTION_MAP = {
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "cot": r"\cot",
    "sec": r"\sec",
    "csc": r"\csc",
    "log": r"\log",
    "ln": r"\ln",
    "exp": r"\exp",
    "int": r"\int",
}


@dataclass
class Token:
    kind: str
    value: str


@dataclass
class NumberNode:
    value: str


@dataclass
class IdentifierNode:
    value: str


@dataclass
class UnaryNode:
    operator: str
    operand: object


@dataclass
class BinaryNode:
    operator: str
    left: object
    right: object


@dataclass
class FunctionNode:
    name: str
    argument: object
    uses_parentheses: bool = False


@dataclass
class GroupNode:
    inner: object


def to_display_formula(text: str) -> str:
    return (text or "").replace("**", "^")


def _normalize_formula_text(text: str) -> str:
    normalized = (text or "").strip()
    normalized = normalized.replace("**", "^")
    normalized = normalized.replace("−", "-").replace("×", "*").replace("÷", "/")
    for source, target in _GREEK_CHAR_MAP.items():
        normalized = normalized.replace(source, target)
    return normalized


def _tokenize_identifier(chunk: str) -> list[Token]:
    tokens: list[Token] = []
    rest = chunk
    while rest:
        underscore_match = re.match(r"^([A-Za-z]+)_([A-Za-z0-9]+)$", rest)
        if underscore_match:
            base, suffix = underscore_match.groups()
            if suffix.isalpha() and len(suffix) > 1:
                tokens.append(Token("IDENT", f"{base}_{suffix[0]}"))
                rest = suffix[1:]
            else:
                tokens.append(Token("IDENT", rest))
                rest = ""
            continue

        matched = False
        for func_name in sorted(_FUNCTION_NAMES, key=len, reverse=True):
            if rest.startswith(func_name):
                tokens.append(Token("FUNC", func_name))
                rest = rest[len(func_name) :]
                matched = True
                break
        if matched:
            continue

        for greek_name in sorted(_GREEK_NAMES, key=len, reverse=True):
            if rest == greek_name:
                tokens.append(Token("IDENT", greek_name))
                rest = ""
                matched = True
                break
            if rest.startswith(greek_name + "_"):
                tokens.append(Token("IDENT", greek_name + rest[len(greek_name) :]))
                rest = ""
                matched = True
                break
        if matched:
            continue

        subscript_prefix = re.match(r"^([A-Za-z]+_[A-Za-z0-9]+)", rest)
        if subscript_prefix:
            token_value = subscript_prefix.group(1)
            tokens.append(Token("IDENT", token_value))
            rest = rest[len(token_value) :]
            continue

        tokens.append(Token("IDENT", rest[0]))
        rest = rest[1:]
    return tokens


def _tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    source = _normalize_formula_text(text)
    index = 0
    while index < len(source):
        char = source[index]
        if char.isspace():
            index += 1
            continue
        if char.isdigit() or (char == "." and index + 1 < len(source) and source[index + 1].isdigit()):
            start = index
            index += 1
            while index < len(source) and (source[index].isdigit() or source[index] == "."):
                index += 1
            tokens.append(Token("NUMBER", source[start:index]))
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < len(source) and (source[index].isalnum() or source[index] == "_"):
                index += 1
            tokens.extend(_tokenize_identifier(source[start:index]))
            continue
        if char in "+-*/()=^":
            kind = {
                "(": "LPAREN",
                ")": "RPAREN",
                "=": "EQUALS",
            }.get(char, "OP")
            tokens.append(Token(kind, char))
            index += 1
            continue
        tokens.append(Token("IDENT", char))
        index += 1
    return tokens


def _factor_can_start(token: Token | None) -> bool:
    return token is not None and token.kind in {"NUMBER", "IDENT", "FUNC", "LPAREN"}


class FormulaParser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.index = 0

    def current(self) -> Token | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def consume(self) -> Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def parse(self):
        if not self.tokens:
            return IdentifierNode("公式为空")
        node = self.parse_relation()
        return node

    def parse_relation(self):
        left = self.parse_expression()
        token = self.current()
        if token and token.kind == "EQUALS":
            self.consume()
            right = self.parse_expression()
            return BinaryNode("=", left, right)
        return left

    def parse_expression(self):
        node = self.parse_term()
        while True:
            token = self.current()
            if token and token.kind == "OP" and token.value in {"+", "-"}:
                operator = self.consume().value
                node = BinaryNode(operator, node, self.parse_term())
                continue
            return node

    def parse_term(self):
        node = self.parse_power()
        while True:
            token = self.current()
            if token and token.kind == "OP" and token.value in {"*", "/"}:
                operator = self.consume().value
                node = BinaryNode(operator, node, self.parse_power())
                continue
            if _factor_can_start(token):
                node = BinaryNode("*", node, self.parse_power())
                continue
            return node

    def parse_power(self):
        node = self.parse_unary()
        token = self.current()
        if token and token.kind == "OP" and token.value == "^":
            self.consume()
            node = BinaryNode("^", node, self.parse_power())
        return node

    def parse_unary(self):
        token = self.current()
        if token and token.kind == "OP" and token.value in {"+", "-"}:
            operator = self.consume().value
            return UnaryNode(operator, self.parse_unary())
        return self.parse_primary()

    def parse_primary(self):
        token = self.current()
        if token is None:
            return IdentifierNode(" ")

        if token.kind == "NUMBER":
            return NumberNode(self.consume().value)
        if token.kind == "IDENT":
            return IdentifierNode(self.consume().value)
        if token.kind == "FUNC":
            function_name = self.consume().value
            next_token = self.current()
            if next_token and next_token.kind == "LPAREN":
                self.consume()
                argument = self.parse_relation()
                if self.current() and self.current().kind == "RPAREN":
                    self.consume()
                return FunctionNode(function_name, argument, True)
            return FunctionNode(function_name, self.parse_power(), False)
        if token.kind == "LPAREN":
            self.consume()
            inner = self.parse_relation()
            if self.current() and self.current().kind == "RPAREN":
                self.consume()
            return GroupNode(inner)
        return IdentifierNode(self.consume().value)


def _parse_formula(text: str):
    return FormulaParser(_tokenize(text)).parse()


def _node_precedence(node) -> int:
    if isinstance(node, BinaryNode):
        return {"=": 1, "+": 2, "-": 2, "*": 3, "/": 3, "^": 4}.get(node.operator, 6)
    if isinstance(node, UnaryNode):
        return 5
    return 6


def _latex_identifier(name: str) -> str:
    if "_" not in name:
        if name in _GREEK_NAMES:
            return "\\" + name
        return name

    base, subscript = name.split("_", 1)
    base_latex = _latex_identifier(base)
    subscript_latex = subscript if len(subscript) == 1 else r"\mathrm{" + subscript + "}"
    return f"{base_latex}_{{{subscript_latex}}}"


def _needs_parentheses(child, parent_operator: str, right_side: bool = False) -> bool:
    if not isinstance(child, (BinaryNode, UnaryNode, GroupNode)):
        return False
    if isinstance(child, GroupNode):
        return False
    child_precedence = _node_precedence(child)
    parent_precedence = {"=": 1, "+": 2, "-": 2, "*": 3, "/": 3, "^": 4, "u": 5}[parent_operator]
    if child_precedence < parent_precedence:
        return True
    if parent_operator == "^" and right_side and child_precedence <= parent_precedence:
        return True
    if parent_operator in {"-", "/"} and right_side and child_precedence == parent_precedence:
        return True
    return False


def _wrap_latex(node, parent_operator: str, right_side: bool = False) -> str:
    latex_text = _node_to_latex(node)
    if _needs_parentheses(node, parent_operator, right_side):
        return r"\left(" + latex_text + r"\right)"
    return latex_text


def _node_to_latex(node) -> str:
    if isinstance(node, NumberNode):
        return node.value
    if isinstance(node, IdentifierNode):
        return _latex_identifier(node.value)
    if isinstance(node, GroupNode):
        return r"\left(" + _node_to_latex(node.inner) + r"\right)"
    if isinstance(node, UnaryNode):
        if node.operator == "+":
            return _node_to_latex(node.operand)
        return "-" + _wrap_latex(node.operand, "u")
    if isinstance(node, FunctionNode):
        if node.name == "sqrt":
            return r"\sqrt{" + _node_to_latex(node.argument) + "}"
        if node.name == "int":
            # `int(...)` comes from the LaTeX normalizer and should render as an integral operator.
            return r"\int " + _node_to_latex(_strip_group(node.argument))
        function_name = _LATEX_FUNCTION_MAP.get(node.name, "\\" + node.name)
        if node.uses_parentheses or isinstance(node.argument, (BinaryNode, GroupNode)):
            return function_name + r"\left(" + _node_to_latex(node.argument) + r"\right)"
        return function_name + " " + _node_to_latex(node.argument)
    if isinstance(node, BinaryNode):
        if node.operator == "=":
            return _node_to_latex(node.left) + "=" + _node_to_latex(node.right)
        if node.operator == "+":
            if isinstance(node.right, UnaryNode) and node.right.operator == "-":
                return _wrap_latex(node.left, "+") + r"\pm " + _wrap_latex(node.right.operand, "+", True)
            return _wrap_latex(node.left, "+") + "+" + _wrap_latex(node.right, "+", True)
        if node.operator == "-":
            return _wrap_latex(node.left, "-") + "-" + _wrap_latex(node.right, "-", True)
        if node.operator == "*":
            left = _wrap_latex(node.left, "*")
            right = _wrap_latex(node.right, "*", True)
            if isinstance(node.left, NumberNode) and isinstance(node.right, NumberNode):
                return left + r"\cdot " + right
            return left + right
        if node.operator == "/":
            return r"\frac{" + _node_to_latex(_strip_group(node.left)) + "}{" + _node_to_latex(_strip_group(node.right)) + "}"
        if node.operator == "^":
            return _wrap_latex(node.left, "^") + "^{" + _node_to_latex(_strip_groups(node.right)) + "}"
    return escape(str(node))


def _node_to_word_linear(node) -> str:
    if isinstance(node, NumberNode):
        return node.value
    if isinstance(node, IdentifierNode):
        return to_display_formula(node.value)
    if isinstance(node, GroupNode):
        return "(" + _node_to_word_linear(node.inner) + ")"
    if isinstance(node, UnaryNode):
        if node.operator == "+":
            return _node_to_word_linear(node.operand)
        return "-" + _node_to_word_linear(node.operand)
    if isinstance(node, FunctionNode):
        if node.name == "sqrt":
            return r"\sqrt(" + _node_to_word_linear(node.argument) + ")"
        if node.name == "int":
            body = _node_to_word_linear(_strip_group(node.argument))
            body = re.sub(r"\*d\*([A-Za-z])\b", r" d\1", body)
            return r"\int " + body
        return node.name + "(" + _node_to_word_linear(node.argument) + ")"
    if isinstance(node, BinaryNode):
        if node.operator == "=":
            return _node_to_word_linear(node.left) + "=" + _node_to_word_linear(node.right)
        if node.operator == "/":
            return "(" + _node_to_word_linear(node.left) + ")/(" + _node_to_word_linear(node.right) + ")"
        if node.operator == "^":
            return _node_to_word_linear(node.left) + "^(" + _node_to_word_linear(_strip_groups(node.right)) + ")"
        return _node_to_word_linear(node.left) + node.operator + _node_to_word_linear(node.right)
    return to_display_formula(str(node))


def _mathml_identifier(name: str) -> str:
    if "_" not in name:
        return f"<mi>{escape(_GREEK_XML_MAP.get(name, name))}</mi>"

    base, subscript = name.split("_", 1)
    base_xml = _mathml_identifier(base)
    subscript_tag = "mn" if subscript.isdigit() else "mi"
    return f"<msub>{base_xml}<{subscript_tag}>{escape(subscript)}</{subscript_tag}></msub>"


def _strip_group(node):
    if isinstance(node, GroupNode):
        return node.inner
    return node


def _strip_groups(node):
    while isinstance(node, GroupNode):
        node = node.inner
    return node


def _node_to_mathml(node) -> str:
    if isinstance(node, NumberNode):
        return f"<mn>{escape(node.value)}</mn>"
    if isinstance(node, IdentifierNode):
        return _mathml_identifier(node.value)
    if isinstance(node, GroupNode):
        return f"<mrow><mo>(</mo>{_node_to_mathml(node.inner)}<mo>)</mo></mrow>"
    if isinstance(node, UnaryNode):
        if node.operator == "+":
            return _node_to_mathml(node.operand)
        return f"<mrow><mo>-</mo>{_node_to_mathml(node.operand)}</mrow>"
    if isinstance(node, FunctionNode):
        if node.name == "sqrt":
            return f"<msqrt>{_node_to_mathml(node.argument)}</msqrt>"
        if node.name == "int":
            return f"<mrow><mo>&#x222B;</mo>{_node_to_mathml(_strip_group(node.argument))}</mrow>"
        function_xml = f'<mi mathvariant="normal">{escape(node.name)}</mi>'
        if node.uses_parentheses or isinstance(node.argument, (BinaryNode, GroupNode)):
            return f"<mrow>{function_xml}<mo>(</mo>{_node_to_mathml(node.argument)}<mo>)</mo></mrow>"
        return f"<mrow>{function_xml}{_node_to_mathml(node.argument)}</mrow>"
    if isinstance(node, BinaryNode):
        if node.operator == "=":
            return f"<mrow>{_node_to_mathml(node.left)}<mo>=</mo>{_node_to_mathml(node.right)}</mrow>"
        if node.operator == "+":
            return f"<mrow>{_node_to_mathml(node.left)}<mo>+</mo>{_node_to_mathml(node.right)}</mrow>"
        if node.operator == "-":
            return f"<mrow>{_node_to_mathml(node.left)}<mo>-</mo>{_node_to_mathml(node.right)}</mrow>"
        if node.operator == "*":
            return f"<mrow>{_node_to_mathml(node.left)}<mo>&InvisibleTimes;</mo>{_node_to_mathml(node.right)}</mrow>"
        if node.operator == "/":
            return f"<mfrac>{_node_to_mathml(_strip_group(node.left))}{_node_to_mathml(_strip_group(node.right))}</mfrac>"
        if node.operator == "^":
            return f"<msup>{_node_to_mathml(node.left)}{_node_to_mathml(_strip_groups(node.right))}</msup>"
    return f"<mtext>{escape(str(node))}</mtext>"


def _fallback_latex(text: str) -> str:
    plain = to_display_formula(text).strip()
    if not plain:
        return r"\mathrm{公式为空}"
    plain = plain.replace("\\", r"\\")
    plain = re.sub(r"([{}_#$%&])", r"\\\1", plain)
    plain = plain.replace(" ", r"\ ")
    return rf"\mathrm{{{plain}}}"


def to_latex_formula(text: str) -> str:
    try:
        return _node_to_latex(_parse_formula(text))
    except Exception:
        return _fallback_latex(text)


def to_word_linear_formula(text: str) -> str:
    try:
        return _node_to_word_linear(_parse_formula(text))
    except Exception:
        expr = to_display_formula(text).strip()
        expr = re.sub(r"sqrt\((.+)\)", r"\\sqrt(\1)", expr)
        return expr.replace("*", "·")


def to_word_mathml(text: str) -> str:
    try:
        body = _node_to_mathml(_parse_formula(text))
    except Exception:
        body = f"<mtext>{escape(to_display_formula(text).strip() or '公式为空')}</mtext>"
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">{body}</math>'


def render_formula_png_bytes(formula_text: str) -> bytes:
    latex_formula = to_latex_formula(formula_text)

    figure = plt.figure(figsize=(6, 1.6), dpi=220)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        f"${latex_formula}$",
        fontsize=24,
        ha="center",
        va="center",
        color="#17324d",
    )

    rendered = BytesIO()
    figure.savefig(rendered, format="png", bbox_inches="tight", pad_inches=0.16, facecolor="white")
    plt.close(figure)
    rendered.seek(0)

    formula_image = Image.open(rendered).convert("RGB")
    width = max(560, formula_image.width + 120)
    height = max(180, formula_image.height + 80)
    canvas = Image.new("RGB", (width, height), "white")
    offset = ((width - formula_image.width) // 2, (height - formula_image.height) // 2)
    canvas.paste(formula_image, offset)

    output = BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()
