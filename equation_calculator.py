"""equation_calculator.py

公式识别与求解入口。

说明：
- 本项目已重构为端到端（E2E）公式识别：整图 -> token 序列（近似 LaTeX）。
- 推理入口在 inference/predictor_e2e.py。
- 本模块负责调用 E2E 推理器，并对识别出的字符串做后处理以供求解。

优化说明（相比旧版简单字符串替换）：
- 实现了递归下降的 LaTeX 解析器，正确处理嵌套花括号
- 支持 \frac{a}{b} -> (a)/(b) 的结构化转换
- 支持 \sqrt[n]{x} -> (x)**(1/(n)) 的n次根号
- 支持上下标 x^{2} -> x**(2), x_{i} -> x_i
- 支持嵌套结构如 \frac{\sqrt{x}}{y+1}
- 保持向后兼容：equation_solver_function 接口不变
"""

from __future__ import annotations

import re
from typing import List, Tuple

from inference.predictor_e2e import E2EPredictor, create_predictor

# 初始化端到端推理器（全局单例，避免重复加载模型）
# 优先使用 V3 模型（checkpoints_e2e_v3/），若不存在则回退到 V2
try:
    _predictor = create_predictor(use_v3=True)
    print("端到端模型加载成功。")
except FileNotFoundError as e:
    print(f"[错误] {e}")
    print("请先运行 python -m train.train_e2e --data_root ... 训练模型")
    _predictor = None


# ============================================================
# LaTeX -> Sympy/Python 递归下降解析器
# ============================================================

# LaTeX命令到Python/Sympy函数的映射表
_FUNC_MAP = {
    r"\sin": "sin",
    r"\cos": "cos",
    r"\tan": "tan",
    r"\cot": "cot",
    r"\sec": "sec",
    r"\csc": "csc",
    r"\arcsin": "asin",
    r"\arccos": "acos",
    r"\arctan": "atan",
    r"\log": "log",
    r"\ln": "ln",
    r"\exp": "exp",
    r"\abs": "Abs",
}

# LaTeX符号到Python/Sympy符号的映射表
_SYMBOL_MAP = {
    r"\pi": "pi",
    r"\theta": "theta",
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\epsilon": "epsilon",
    r"\lambda": "lambda_",
    r"\mu": "mu",
    r"\sigma": "sigma",
    r"\omega": "omega",
    r"\phi": "phi",
    r"\psi": "psi",
    r"\infty": "oo",
    r"\pm": "+-",
    r"\times": "*",
    r"\cdot": "*",
    r"\div": "/",
    r"\leq": "<=",
    r"\geq": ">=",
    r"\neq": "!=",
    r"\le": "<=",
    r"\ge": ">=",
    r"\ne": "!=",
    r"\left": "",
    r"\right": "",
}


def _tokenize_latex(text: str) -> List[str]:
    """将LaTeX字符串分词为token列表。

    处理规则：
    - 反斜杠命令（如 \\frac, \\sqrt）作为单个token
    - 花括号 { } 作为单个token
    - 方括号 [ ] 作为单个token
    - 数字序列合并为单个token
    - 其他字符各自为一个token
    """
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]

        # 跳过空格
        if ch == ' ':
            i += 1
            continue

        # 反斜杠命令
        if ch == '\\':
            j = i + 1
            while j < len(text) and text[j].isalpha():
                j += 1
            if j == i + 1:
                # 单字符转义如 \{ \}
                if j < len(text):
                    tokens.append(text[i:j + 1])
                    i = j + 1
                else:
                    i = j
            else:
                tokens.append(text[i:j])
                i = j
            continue

        # 花括号、方括号、圆括号
        if ch in '{}[]()':
            tokens.append(ch)
            i += 1
            continue

        # 数字（含小数点）
        if ch.isdigit() or (ch == '.' and i + 1 < len(text) and text[i + 1].isdigit()):
            j = i
            while j < len(text) and (text[j].isdigit() or text[j] == '.'):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue

        # 运算符和其他字符
        tokens.append(ch)
        i += 1

    return tokens


def _parse_group(tokens: List[str], pos: int) -> Tuple[str, int]:
    """解析一个花括号组 { ... } 或单个token。

    返回 (解析结果字符串, 新位置)。
    递归处理嵌套花括号，正确匹配层级。
    """
    if pos >= len(tokens):
        return "", pos

    if tokens[pos] == '{':
        # 收集花括号内的所有token
        depth = 1
        inner_start = pos + 1
        j = inner_start
        while j < len(tokens) and depth > 0:
            if tokens[j] == '{':
                depth += 1
            elif tokens[j] == '}':
                depth -= 1
            j += 1
        inner_tokens = tokens[inner_start:j - 1]
        result = _convert_tokens(inner_tokens)
        return result, j
    else:
        # 单个token
        tok = tokens[pos]
        converted = _convert_single_token(tok)
        return converted, pos + 1


def _convert_single_token(tok: str) -> str:
    """转换单个LaTeX token为Python/Sympy格式。"""
    if tok in _SYMBOL_MAP:
        return _SYMBOL_MAP[tok]
    if tok in _FUNC_MAP:
        return _FUNC_MAP[tok]
    return tok


def _convert_tokens(tokens: List[str]) -> str:
    """将token列表转换为Python/Sympy表达式字符串。

    核心解析逻辑，处理：
    - \\frac{a}{b} -> (a)/(b)
    - \\sqrt{x} -> sqrt(x), \\sqrt[n]{x} -> (x)**(1/(n))
    - x^{n} -> x**(n)
    - x_{i} -> x_i
    - 三角函数、对数等 \\sin{x} -> sin(x)
    """
    result = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        # \frac{分子}{分母} -> (分子)/(分母)
        if tok == r"\frac":
            i += 1
            numerator, i = _parse_group(tokens, i)
            denominator, i = _parse_group(tokens, i)
            result.append(f"({numerator})/({denominator})")
            continue

        # \sqrt[n]{x} 或 \sqrt{x}
        if tok == r"\sqrt":
            i += 1
            # 检查是否有可选参数 [n]
            if i < len(tokens) and tokens[i] == '[':
                # 收集方括号内容
                i += 1  # 跳过 [
                n_parts = []
                while i < len(tokens) and tokens[i] != ']':
                    n_parts.append(_convert_single_token(tokens[i]))
                    i += 1
                if i < len(tokens):
                    i += 1  # 跳过 ]
                n_str = "".join(n_parts)
                body, i = _parse_group(tokens, i)
                result.append(f"({body})**(1/({n_str}))")
            else:
                body, i = _parse_group(tokens, i)
                result.append(f"sqrt({body})")
            continue

        # \sum, \prod, \int 等大型运算符（简化处理）
        if tok in (r"\sum", r"\prod", r"\int"):
            op_name = tok[1:]  # 去掉反斜杠
            i += 1
            # 跳过可能的上下界 _{}^{}
            if i < len(tokens) and tokens[i] == '_':
                i += 1
                _, i = _parse_group(tokens, i)
            if i < len(tokens) and tokens[i] == '^':
                i += 1
                _, i = _parse_group(tokens, i)
            result.append(f"{op_name}(")
            # 后续内容会自然追加，需要在合适位置关闭括号
            # 简化处理：不自动关闭，依赖后续token
            continue

        # 上标 ^
        if tok == '^':
            i += 1
            exponent, i = _parse_group(tokens, i)
            result.append(f"**({exponent})")
            continue

        # 下标 _ （保留为Python风格下划线）
        if tok == '_':
            i += 1
            subscript, i = _parse_group(tokens, i)
            result.append(f"_{subscript}")
            continue

        # 函数命令如 \sin, \cos, \log 等
        if tok in _FUNC_MAP:
            func_name = _FUNC_MAP[tok]
            i += 1
            # 如果后面跟着花括号组，解析为函数参数
            if i < len(tokens) and tokens[i] == '{':
                arg, i = _parse_group(tokens, i)
                result.append(f"{func_name}({arg})")
            else:
                result.append(func_name)
            continue

        # 符号映射
        if tok in _SYMBOL_MAP:
            result.append(_SYMBOL_MAP[tok])
            i += 1
            continue

        # 花括号组（独立出现的）
        if tok == '{':
            body, i = _parse_group(tokens, i)
            result.append(f"({body})")
            continue

        # 普通token（数字、变量、运算符）
        result.append(tok)
        i += 1

    return "".join(result)


def latex_to_sympy(raw_text: str) -> str:
    """将模型输出的LaTeX token序列转换为Sympy可解析的表达式。

    这是主入口函数，替代旧版的简单字符串替换。

    参数:
        raw_text: 模型输出的token序列，如 '\\frac { 1 } { 2 } + \\sqrt { 3 }'

    返回:
        Sympy可解析的字符串，如 '(1)/(2)+sqrt(3)'

    示例:
        >>> latex_to_sympy(r'\\frac { x + 1 } { y }')
        '(x+1)/(y)'
        >>> latex_to_sympy(r'\\sqrt { 4 8 }')
        'sqrt(48)'
        >>> latex_to_sympy(r'x ^ { 2 } + y ^ { 2 }')
        'x**(2)+y**(2)'
        >>> latex_to_sympy(r'\\sin { \\theta }')
        'sin(theta)'
    """
    tokens = _tokenize_latex(raw_text)
    return _convert_tokens(tokens)


def latex_add_spaces(text: str) -> str:
    """在 LaTeX 各 token 之间加空格，不破坏结构。

    复用 _tokenize_latex 将紧凑的 LaTeX 拆成 token，再用空格拼接。

    示例:
        >>> latex_add_spaces(r"\\frac{x}{y}+1")
        '\\\\frac { x } { y } + 1'
        >>> latex_add_spaces(r"x^{2}+\\sqrt{3}")
        'x ^ { 2 } + \\\\sqrt { 3 }'
    """
    return ' '.join(_tokenize_latex(text))


def equation_solver_function(img_path: str) -> str:
    """调用端到端模型识别公式图片，并返回可供 sympy/eval 使用的字符串。

    API向后兼容：函数签名和返回值格式不变。
    """
    if _predictor is None:
        return "错误：模型未加载，请检查训练是否完成。"

    try:
        # 1. 端到端模型直接输出 token 序列字符串
        raw_text = _predictor.predict_text(img_path)

        # 2. 使用递归下降解析器转换LaTeX到Sympy格式
        processed_text = latex_to_sympy(raw_text)
        return processed_text

    except Exception as e:
        print(f"[识别异常] {e}")
        return "识别过程中发生错误。"
