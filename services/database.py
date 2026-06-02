from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher

from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "classroom_app.json"
VALID_ROLES = {"admin", "teacher", "student"}
_SUPER_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

_COMMON_CORRECTION_SEEDS = (
    {
        "label": "quadratic_roots",
        "wrong_text": "-b+sqrt(b^2-4ac)",
        "correct_text": "x=(-b+-sqrt(b**(2)-4*a*c))/(2*a)",
        "match_texts": [
            "-b+sqrt(b^2-4ac)",
            "-b-sqrt(b^2-4ac)",
            "b^2-4ac",
            "4ac-b^2",
            "sqrt(b^2-4ac)",
            "\\sqrt{b^{2}-4ac}",
        ],
    },
    {
        "label": "quadratic_discriminant",
        "wrong_text": "b^2-4ac",
        "correct_text": "D=b**(2)-4*a*c",
        "match_texts": ["b^2-4ac", "4ac-b^2", "delta=b^2-4ac", "D=b^2-4ac"],
    },
    {
        "label": "quadratic_vertex_x",
        "wrong_text": "-b/(2a)",
        "correct_text": "x=-b/(2*a)",
        "match_texts": ["-b/(2a)", "-b/2a", "x=-b/(2a)"],
    },
    {
        "label": "quadratic_vertex_y",
        "wrong_text": "4ac-b^2",
        "correct_text": "y=(4*a*c-b**(2))/(4*a)",
        "match_texts": ["(4ac-b^2)/(4a)", "4ac-b^2", "4*a*c-b^2", "b^2-4ac"],
    },
    {
        "label": "pythagorean",
        "wrong_text": "a^2+b^2",
        "correct_text": "c**(2)=a**(2)+b**(2)",
        "match_texts": ["a^2+b^2", "c^2=a^2+b^2", "sqrt(a^2+b^2)", "a2+b2"],
    },
    {
        "label": "distance",
        "wrong_text": "sqrt((x2-x1)^2+(y2-y1)^2)",
        "correct_text": "d=sqrt((x_2-x_1)**(2)+(y_2-y_1)**(2))",
        "match_texts": ["sqrt((x2-x1)^2+(y2-y1)^2)", "(x2-x1)^2+(y2-y1)^2"],
    },
    {
        "label": "slope",
        "wrong_text": "(y2-y1)/(x2-x1)",
        "correct_text": "k=(y_2-y_1)/(x_2-x_1)",
        "match_texts": ["(y2-y1)/(x2-x1)", "y2-y1/x2-x1", "k=(y2-y1)/(x2-x1)"],
    },
    {
        "label": "circle_area",
        "wrong_text": "pi*r^2",
        "correct_text": "S=pi*r**(2)",
        "match_texts": ["pi*r^2", "pir2", "S=pi*r^2", "area=pi*r^2"],
    },
    {
        "label": "circle_circumference",
        "wrong_text": "2*pi*r",
        "correct_text": "C=2*pi*r",
        "match_texts": ["2*pi*r", "2pir", "C=2*pi*r"],
    },
    {
        "label": "arithmetic_series",
        "wrong_text": "n(a1+an)/2",
        "correct_text": "S=n*(a_1+a_n)/2",
        "match_texts": ["n(a1+an)/2", "n*(a1+an)/2", "Sn=n(a1+an)/2"],
    },
    {
        "label": "trig_identity",
        "wrong_text": "sin^2x+cos^2x",
        "correct_text": "sin(x)**(2)+cos(x)**(2)=1",
        "match_texts": ["sin^2x+cos^2x", "sin(x)^2+cos(x)^2", "sin2x+cos2x"],
    },
)


def _normalize_username(username: str) -> str:
    return username.strip().casefold()


def _empty_store() -> dict:
    return {
        "next_user_id": 1,
        "next_history_id": 1,
        "next_correction_id": 1,
        "users": [],
        "recognition_history": [],
        "correction_table": [],
    }


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_formula_text(text: str | None) -> str:
    return " ".join(str(text or "").strip().split())


def _formula_match_key(text: str | None) -> str:
    value = _normalize_formula_text(text).casefold().translate(_SUPER_DIGITS)
    replacements = {
        "−": "-",
        "×": "*",
        "·": "*",
        "÷": "/",
        "√": "sqrt",
        "**": "^",
        "\\left": "",
        "\\right": "",
        "\\sqrt": "sqrt",
        "\\cdot": "*",
        "\\times": "*",
        "\\div": "/",
        "\\pi": "pi",
        "\\theta": "theta",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = re.sub(r"\\([a-z]+)", r"\1", value)
    value = value.replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")")
    value = re.sub(r"\^\(([a-z0-9]+)\)", r"^\1", value)
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[^a-z0-9+\-*/^=().]", "", value)
    return value.replace("*", "")


def _ngram_score(left: str, right: str, size: int = 3) -> float:
    if len(left) < size or len(right) < size:
        return 0.0
    left_grams = {left[index : index + size] for index in range(len(left) - size + 1)}
    right_grams = {right[index : index + size] for index in range(len(right) - size + 1)}
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _formula_match_score(recognized_key: str, candidate_key: str) -> float:
    if not recognized_key or not candidate_key:
        return 0.0
    if recognized_key == candidate_key:
        return 1.0
    if len(candidate_key) >= 5 and candidate_key in recognized_key:
        return 0.94
    if len(recognized_key) >= 5 and recognized_key in candidate_key:
        coverage = len(recognized_key) / max(len(candidate_key), 1)
        return 0.88 if coverage >= 0.35 else 0.72
    sequence_score = SequenceMatcher(None, recognized_key, candidate_key).ratio()
    return max(sequence_score, _ngram_score(recognized_key, candidate_key))


def _ensure_store_shape(store: dict) -> dict:
    store.setdefault("users", [])
    store.setdefault("recognition_history", [])
    store.setdefault("correction_table", [])
    store.setdefault("next_user_id", 1)
    store.setdefault("next_history_id", 1)
    if "next_correction_id" not in store:
        max_id = max((int(item.get("id", 0)) for item in store["correction_table"]), default=0)
        store["next_correction_id"] = max_id + 1

    for item in store["recognition_history"]:
        item.setdefault("original_recognized_text", "")
        item.setdefault("is_corrected", 0)
        item.setdefault("corrected_at", "")
    return store


def _load_store() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        return _empty_store()
    with DB_PATH.open("r", encoding="utf-8") as f:
        return _ensure_store_shape(json.load(f))


def _save_store(store: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with DB_PATH.open("w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def init_db() -> None:
    if not DB_PATH.exists():
        _save_store(_empty_store())
    else:
        _save_store(_load_store())
    seed_admin_user()
    seed_common_corrections()


def seed_admin_user() -> None:
    if get_user_by_username("admin"):
        return
    create_user(
        username="admin",
        password="admin123",
        role="admin",
        full_name="系统管理员",
    )


def seed_common_corrections() -> None:
    store = _load_store()
    now = _now_text()
    existing_builtin = {
        (
            str(item.get("source_type", "")),
            _normalize_formula_text(item.get("label")),
            _normalize_formula_text(item.get("wrong_text")),
        ): item
        for item in store["correction_table"]
    }

    changed = False
    for seed in _COMMON_CORRECTION_SEEDS:
        entry_key = (
            "builtin_formula",
            _normalize_formula_text(seed["label"]),
            _normalize_formula_text(seed["wrong_text"]),
        )
        existing_item = existing_builtin.get(entry_key)
        if existing_item is not None:
            if (
                existing_item.get("correct_text") != seed["correct_text"]
                or existing_item.get("match_texts") != seed.get("match_texts", [])
            ):
                existing_item["correct_text"] = seed["correct_text"]
                existing_item["match_texts"] = seed.get("match_texts", [])
                existing_item["updated_at"] = now
                changed = True
            continue

        correction_id = int(store["next_correction_id"])
        store["next_correction_id"] += 1
        store["correction_table"].append(
            {
                "id": correction_id,
                "user_id": 0,
                "label": seed["label"],
                "wrong_text": seed["wrong_text"],
                "match_texts": seed.get("match_texts", []),
                "correct_text": seed["correct_text"],
                "source_type": "builtin_formula",
                "is_builtin": 1,
                "count": 0,
                "created_at": now,
                "updated_at": now,
            }
        )
        changed = True

    if changed:
        _save_store(store)


def create_user(username: str, password: str, role: str, full_name: str) -> int:
    store = _load_store()
    username = username.strip()
    full_name = full_name.strip()
    role = role.strip()

    if role not in VALID_ROLES:
        raise ValueError("invalid_role")
    if not username or not full_name or not password:
        raise ValueError("missing_fields")

    normalized_username = _normalize_username(username)
    if any(_normalize_username(user["username"]) == normalized_username for user in store["users"]):
        raise ValueError("username already exists")

    user_id = int(store["next_user_id"])
    store["next_user_id"] += 1
    store["users"].append(
        {
            "id": user_id,
            "username": username,
            "password_hash": generate_password_hash(password),
            "role": role,
            "full_name": full_name,
            "is_active": 1,
            "created_at": _now_text(),
        }
    )
    _save_store(store)
    return user_id


def list_users() -> list[dict]:
    store = _load_store()
    role_order = {"admin": 0, "teacher": 1, "student": 2}
    return sorted(
        store["users"],
        key=lambda item: (role_order.get(item["role"], 99), -int(item["id"])),
    )


def get_user_by_username(username: str) -> dict | None:
    store = _load_store()
    normalized_username = _normalize_username(username)
    return next(
        (user for user in store["users"] if _normalize_username(user["username"]) == normalized_username),
        None,
    )


def get_user_by_id(user_id: int) -> dict | None:
    store = _load_store()
    return next((user for user in store["users"] if int(user["id"]) == int(user_id)), None)


def authenticate_user(username: str, password: str, role: str | None = None) -> dict | None:
    user = get_user_by_username(username)
    if not user or not user["is_active"]:
        return None
    if role and user["role"] != role:
        return None
    if not check_password_hash(user["password_hash"], password):
        return None
    return user


def change_user_password(user_id: int, current_password: str, new_password: str) -> None:
    store = _load_store()
    for user in store["users"]:
        if int(user["id"]) != int(user_id):
            continue
        if not check_password_hash(user["password_hash"], current_password):
            raise ValueError("current_password_incorrect")
        user["password_hash"] = generate_password_hash(new_password)
        _save_store(store)
        return
    raise ValueError("user_not_found")


def reset_user_password(user_id: int, new_password: str = "123456") -> None:
    store = _load_store()
    for user in store["users"]:
        if int(user["id"]) == int(user_id):
            user["password_hash"] = generate_password_hash(new_password)
            _save_store(store)
            return
    raise ValueError("user_not_found")


def delete_user(user_id: int) -> dict:
    store = _load_store()
    for index, user in enumerate(store["users"]):
        if int(user["id"]) != int(user_id):
            continue
        deleted_user = dict(user)
        for item in store["recognition_history"]:
            if int(item["user_id"]) != int(user_id):
                continue
            item["username_snapshot"] = item.get("username_snapshot") or deleted_user["username"]
            item["full_name_snapshot"] = item.get("full_name_snapshot") or deleted_user["full_name"]
            item["role_snapshot"] = item.get("role_snapshot") or deleted_user["role"]
        del store["users"][index]
        _save_store(store)
        return deleted_user
    raise ValueError("user_not_found")


def _is_history_visible_to_user(item: dict, user: dict) -> bool:
    return user["role"] == "admin" or int(item["user_id"]) == int(user["id"])


def delete_history_items(user: dict, item_ids: list[int]) -> int:
    target_ids = {int(item_id) for item_id in item_ids}
    if not target_ids:
        return 0

    store = _load_store()
    remaining_items = []
    deleted_count = 0
    for item in store["recognition_history"]:
        if int(item["id"]) in target_ids and _is_history_visible_to_user(item, user):
            deleted_count += 1
            continue
        remaining_items.append(item)
    store["recognition_history"] = remaining_items
    _save_store(store)
    return deleted_count


def delete_all_history_for_user(user: dict) -> int:
    store = _load_store()
    original_count = len(store["recognition_history"])
    if user["role"] == "admin":
        store["recognition_history"] = []
    else:
        store["recognition_history"] = [
            item for item in store["recognition_history"] if int(item["user_id"]) != int(user["id"])
        ]
    deleted_count = original_count - len(store["recognition_history"])
    _save_store(store)
    return deleted_count


def record_history(
    user_id: int,
    source_type: str,
    original_path: str | None,
    processed_path: str | None,
    recognized_text: str,
    word_linear_text: str,
    word_mathml_text: str | None = None,
) -> int:
    store = _load_store()
    history_user = next((entry for entry in store["users"] if int(entry["id"]) == int(user_id)), None)
    item_id = int(store["next_history_id"])
    store["next_history_id"] += 1
    store["recognition_history"].append(
        {
            "id": item_id,
            "user_id": user_id,
            "source_type": source_type,
            "original_path": original_path,
            "processed_path": processed_path,
            "recognized_text": recognized_text,
            "original_recognized_text": "",
            "is_corrected": 0,
            "corrected_at": "",
            "word_linear_text": word_linear_text,
            "word_mathml_text": word_mathml_text or "",
            "username_snapshot": history_user.get("username") if history_user else "",
            "full_name_snapshot": history_user.get("full_name") if history_user else "",
            "role_snapshot": history_user.get("role") if history_user else "",
            "created_at": _now_text(),
        }
    )
    _save_store(store)
    return item_id


def list_correction_suggestions(wrong_text: str, limit: int = 5) -> list[dict]:
    store = _load_store()
    normalized_wrong = _normalize_formula_text(wrong_text)
    if not normalized_wrong:
        return []

    recognized_key = _formula_match_key(normalized_wrong)
    ranked: list[tuple[int, float, int, int, int, dict]] = []
    for item in store["correction_table"]:
        candidate_texts = [
            _normalize_formula_text(item.get("wrong_text")),
            *[_normalize_formula_text(value) for value in (item.get("match_texts") or [])],
        ]
        candidate_texts = [value for value in candidate_texts if value]
        if not candidate_texts:
            continue

        exact_match = normalized_wrong in candidate_texts
        if exact_match:
            match_group = 0
            score = 1.0
        else:
            score = max(
                _formula_match_score(recognized_key, _formula_match_key(candidate))
                for candidate in candidate_texts
            )
            if score < 0.72:
                continue
            match_group = 1 if score >= 0.86 else 2
        ranked.append(
            (
                match_group,
                -score,
                int(item.get("is_builtin", 0)),
                -int(item.get("count", 1)),
                int(item["id"]),
                item,
            )
        )

    ranked.sort()
    suggestions = []
    seen_formulas = set()
    for _, score, _, _, _, item in ranked:
        formula = item.get("correct_text", "")
        normalized_formula = _normalize_formula_text(formula)
        if not normalized_formula or normalized_formula in seen_formulas:
            continue
        seen_formulas.add(normalized_formula)
        suggestions.append(
            {
                "id": int(item["id"]),
                "formula": formula,
                "count": int(item.get("count", 1)),
                "source_type": item.get("source_type", ""),
                "is_builtin": int(item.get("is_builtin", 0)),
                "score": round(-score, 3),
            }
        )
        if len(suggestions) >= limit:
            break
    return suggestions


def record_correction(user_id: int, wrong_text: str, correct_text: str, source_type: str) -> int | None:
    normalized_wrong = _normalize_formula_text(wrong_text)
    normalized_correct = _normalize_formula_text(correct_text)
    if not normalized_wrong or not normalized_correct or normalized_wrong == normalized_correct:
        return None

    store = _load_store()
    now = _now_text()
    for item in store["correction_table"]:
        if (
            _normalize_formula_text(item.get("wrong_text")) == normalized_wrong
            and _normalize_formula_text(item.get("correct_text")) == normalized_correct
        ):
            item["count"] = int(item.get("count", 1)) + 1
            item["updated_at"] = now
            _save_store(store)
            return int(item["id"])

    correction_id = int(store["next_correction_id"])
    store["next_correction_id"] += 1
    store["correction_table"].append(
        {
            "id": correction_id,
            "user_id": user_id,
            "wrong_text": wrong_text.strip(),
            "correct_text": correct_text.strip(),
            "source_type": source_type,
            "count": 1,
            "created_at": now,
            "updated_at": now,
        }
    )
    _save_store(store)
    return correction_id


def update_history_correction(
    user: dict,
    item_id: int,
    correct_text: str,
    word_linear_text: str,
    word_mathml_text: str,
) -> dict | None:
    store = _load_store()
    for item in store["recognition_history"]:
        if int(item["id"]) != int(item_id):
            continue
        if user["role"] != "admin" and int(item["user_id"]) != int(user["id"]):
            return None

        original_text = item.get("original_recognized_text") or item.get("recognized_text", "")
        item["original_recognized_text"] = original_text
        item["recognized_text"] = correct_text.strip()
        item["word_linear_text"] = word_linear_text
        item["word_mathml_text"] = word_mathml_text
        item["is_corrected"] = 1
        item["corrected_at"] = _now_text()
        _save_store(store)
        updated = _join_history_with_user(item, store["users"])
        updated["correction_wrong_text"] = original_text
        return updated


def _join_history_with_user(item: dict, users: list[dict]) -> dict:
    user = next((entry for entry in users if int(entry["id"]) == int(item["user_id"])), None)
    merged = dict(item)
    if user:
        merged["username"] = user["username"]
        merged["full_name"] = user["full_name"]
        merged["role"] = user["role"]
    else:
        merged["username"] = item.get("username_snapshot") or f"已删除用户#{item['user_id']}"
        merged["full_name"] = item.get("full_name_snapshot") or "已删除用户"
        merged["role"] = item.get("role_snapshot") or "student"
    return merged


def list_history_for_user(user: dict) -> list[dict]:
    store = _load_store()
    items = store["recognition_history"]
    if user["role"] != "admin":
        items = [item for item in items if int(item["user_id"]) == int(user["id"])]
    merged = [_join_history_with_user(item, store["users"]) for item in items]
    return sorted(merged, key=lambda item: -int(item["id"]))


def get_history_item(item_id: int, user: dict) -> dict | None:
    store = _load_store()
    for item in store["recognition_history"]:
        if int(item["id"]) != int(item_id):
            continue
        if user["role"] != "admin" and int(item["user_id"]) != int(user["id"]):
            return None
        return _join_history_with_user(item, store["users"])
    return None
