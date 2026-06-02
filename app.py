"""Flask application for the classroom-oriented handwritten formula system."""

from __future__ import annotations

import base64
from functools import wraps
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from flask import (
    Flask,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from equation_calculator import equation_solver_function, latex_to_sympy
from inference.image_preprocessing import decode_image_bytes, preprocess_and_save, save_image_unicode
from services.database import (
    change_user_password,
    create_user,
    delete_all_history_for_user,
    delete_history_items,
    delete_user,
    get_history_item,
    get_user_by_id,
    authenticate_user,
    init_db,
    list_correction_suggestions,
    list_history_for_user,
    list_users,
    record_correction,
    record_history,
    reset_user_password,
    update_history_correction,
)
from services.formula_formats import (
    render_formula_png_bytes,
    to_display_formula,
    to_latex_formula,
    to_word_linear_formula,
    to_word_mathml,
)
from solve_equation_file import solve_equation


app = Flask(__name__)
app.config["SECRET_KEY"] = "abcdef"
app.config["UPLOAD_FOLDER"] = "static"

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / app.config["UPLOAD_FOLDER"]
HISTORY_DIR = STATIC_DIR / "history"
UPLOAD_IMAGE_PATH = STATIC_DIR / "equation.png"
UPLOAD_ORIGINAL_PATH = STATIC_DIR / "equation_original.png"
SKETCH_IMAGE_PATH = STATIC_DIR / "sketch.png"
SKETCH_ORIGINAL_PATH = STATIC_DIR / "sketch_original.png"
ROLE_LABELS = {"admin": "管理员", "teacher": "老师", "student": "学生"}
FORMULA_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789=+-*/^_()\\{}[]")


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except PermissionError:
        pass


def _clear_cached_images() -> None:
    for path in (UPLOAD_IMAGE_PATH, UPLOAD_ORIGINAL_PATH, SKETCH_IMAGE_PATH, SKETCH_ORIGINAL_PATH):
        _safe_unlink(path)


def _read_request_image(file_storage) -> object:
    image_bytes = file_storage.read()
    if not image_bytes:
        return None
    return decode_image_bytes(image_bytes)


def _relative_static_path(path: Path) -> str:
    return path.relative_to(BASE_DIR).as_posix()


def _make_history_paths(prefix: str) -> tuple[Path, Path]:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    suffix = uuid4().hex
    return (
        HISTORY_DIR / f"{prefix}_{suffix}_original.png",
        HISTORY_DIR / f"{prefix}_{suffix}_processed.png",
    )


def _build_result_payload(history_id: int, recognized_text: str, correction_suggestions: list[dict] | None = None) -> dict:
    latex_text = _to_latex_output(recognized_text)
    suggestions = (
        correction_suggestions
        if correction_suggestions is not None
        else list_correction_suggestions(recognized_text)
    )
    return {
        "success": True,
        "latex": latex_text,
        "word_linear": to_word_linear_formula(recognized_text),
        "word_mathml": to_word_mathml(recognized_text),
        "history_id": history_id,
        "formula_png_url": url_for("formula_image", item_id=history_id),
        "history_url": url_for("history_page"),
        "correction_suggestions": _format_correction_suggestions(suggestions),
    }


def _looks_like_formula(text: str) -> bool:
    return any(char in FORMULA_CHARS for char in text)


def _normalize_formula_for_storage(text: str) -> str:
    value = (text or "").strip()
    if not value or not _looks_like_formula(value):
        return value
    try:
        return latex_to_sympy(value)
    except Exception:
        return value.replace("^", "**")


def _to_latex_output(text: str) -> str:
    value = (text or "").strip()
    if not value or not _looks_like_formula(value):
        return value
    try:
        return to_latex_formula(_normalize_formula_for_storage(value))
    except Exception:
        return to_display_formula(value)


def _format_correction_suggestions(suggestions: list[dict]) -> list[dict]:
    formatted = []
    for item in suggestions:
        formatted_item = dict(item)
        formatted_item["formula"] = _to_latex_output(str(item.get("formula", "")))
        formatted.append(formatted_item)
    return formatted


def _parse_history_ids(values: list[str]) -> list[int]:
    history_ids: list[int] = []
    for value in values:
        try:
            history_ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return history_ids


def _parse_positive_int(value: str | None, default: int = 1) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


def admin_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            return redirect(url_for("login"))
        if g.user["role"] != "admin":
            flash("只有管理员可以访问该页面。", "warning")
            return redirect(url_for("index"))
        return view(*args, **kwargs)

    return wrapped_view


@app.before_request
def load_logged_in_user() -> None:
    user_id = session.get("user_id")
    if not user_id:
        g.user = None
        return

    user = get_user_by_id(int(user_id))
    if not user or not user.get("is_active"):
        session.clear()
        if request.endpoint not in {"login", "static"}:
            flash("当前账号已被注销，请联系管理员。", "warning")
        g.user = None
        return
    g.user = user


@app.context_processor
def inject_current_user() -> dict:
    return {"current_user": g.user, "role_labels": ROLE_LABELS}


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        role = request.form.get("role", "").strip()
        user = authenticate_user(username, password, role or None)
        if user is None:
            flash("身份、用户名或密码错误。", "warning")
        else:
            session.clear()
            session["user_id"] = user["id"]
            return redirect(url_for("index"))

    return render_template("login.html", auth_mode="login")


@app.route("/register", methods=["GET", "POST"])
def register():
    if g.user:
        return redirect(url_for("index"))

    if request.method == "GET":
        return render_template("login.html", auth_mode="register")

    full_name = request.form.get("full_name", "").strip()
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")
    role = request.form.get("role", "").strip()

    if role not in {"teacher", "student"}:
        flash("注册时只能选择教师或学生身份。", "warning")
    elif not full_name or not username or not password or not confirm_password:
        flash("请完整填写注册信息。", "warning")
    elif password != confirm_password:
        flash("两次输入的密码不一致。", "warning")
    elif len(password) < 6:
        flash("注册密码长度不能少于 6 位。", "warning")
    else:
        try:
            create_user(username=username, password=password, role=role, full_name=full_name)
            flash("注册成功，请登录。", "success")
            return redirect(url_for("login"))
        except ValueError:
            flash("账号已存在或注册信息不合法。", "warning")

    return redirect(url_for("register"))


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/", methods=["GET"])
@login_required
def index():
    return render_template("homepage.html")


@app.route("/upload_image", methods=["GET"])
@login_required
def upload_image():
    _clear_cached_images()
    return render_template("uploadimage.html")


@app.route("/account/password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not current_password or not new_password or not confirm_password:
            flash("请完整填写密码信息。", "warning")
        elif new_password != confirm_password:
            flash("两次输入的新密码不一致。", "warning")
        elif len(new_password) < 6:
            flash("新密码长度不能少于 6 位。", "warning")
        elif current_password == new_password:
            flash("新密码不能与当前密码相同。", "warning")
        else:
            try:
                change_user_password(g.user["id"], current_password, new_password)
                flash("密码修改成功。", "success")
                return redirect(url_for("change_password"))
            except ValueError as exc:
                if str(exc) == "current_password_incorrect":
                    flash("当前密码输入错误。", "warning")
                else:
                    flash("账号不存在，无法修改密码。", "warning")

    return render_template("change_password.html")


@app.route("/admin/users", methods=["GET"])
@admin_required
def admin_users():
    return render_template("admin_users.html", users=list_users())


@app.route("/admin/users", methods=["POST"])
@admin_required
def admin_create_user():
    username = request.form.get("username", "").strip()
    full_name = request.form.get("full_name", "").strip()
    role = request.form.get("role", "student").strip()
    password = request.form.get("password", "")

    if not username or not full_name or not password:
        flash("请完整填写账号信息。", "warning")
        return redirect(url_for("admin_users"))

    try:
        create_user(username=username, password=password, role=role, full_name=full_name)
        flash("用户创建成功。", "success")
    except ValueError:
        flash("用户名已存在或数据不合法。", "warning")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/<int:user_id>/reset-password", methods=["POST"])
@admin_required
def admin_reset_user_password(user_id: int):
    target = get_user_by_id(user_id)
    if not target:
        flash("用户不存在。", "warning")
    else:
        reset_user_password(user_id, "123456")
        flash(f"已将 {target['username']} 的密码重置为 123456。", "success")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def admin_delete_user(user_id: int):
    target = get_user_by_id(user_id)
    if not target:
        flash("用户不存在。", "warning")
    elif target["role"] == "admin":
        flash("管理员账号不能删除。", "warning")
    else:
        deleted = delete_user(user_id)
        flash(f"账号 {deleted['username']} 已清理，历史记录已保留。", "success")
    return redirect(url_for("admin_users"))


@app.route("/history", methods=["GET"])
@login_required
def history_page():
    all_history_items = list_history_for_user(g.user)
    total_items = len(all_history_items)
    page_size = 10
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    current_page = min(_parse_positive_int(request.args.get("page"), 1), total_pages)
    start_index = (current_page - 1) * page_size
    end_index = start_index + page_size
    history_items = all_history_items[start_index:end_index]

    for item in history_items:
        normalized_recognized_text = _normalize_formula_for_storage(item["recognized_text"])
        item["word_linear_text"] = to_word_linear_formula(normalized_recognized_text)
        item["word_mathml_text"] = to_word_mathml(normalized_recognized_text)
        item["latex_text"] = _to_latex_output(normalized_recognized_text)
        item["original_latex_text"] = _to_latex_output(item.get("original_recognized_text", ""))
    return render_template(
        "history.html",
        history_items=history_items,
        current_page=current_page,
        total_pages=total_pages,
        total_items=total_items,
    )


@app.route("/history/delete", methods=["POST"])
@login_required
def history_delete():
    history_ids = _parse_history_ids(request.form.getlist("history_ids"))
    page = _parse_positive_int(request.form.get("page"), 1)
    deleted_count = delete_history_items(g.user, history_ids)
    if deleted_count:
        flash(f"已清理 {deleted_count} 条识别记录。", "success")
    else:
        flash("未选择可清理的识别记录。", "warning")
    return redirect(url_for("history_page", page=page))


@app.route("/history/clear", methods=["POST"])
@login_required
def history_clear():
    page = _parse_positive_int(request.form.get("page"), 1)
    deleted_count = delete_all_history_for_user(g.user)
    if deleted_count:
        flash(f"已清空 {deleted_count} 条识别记录。", "success")
    else:
        flash("当前没有可清理的识别记录。", "warning")
    return redirect(url_for("history_page", page=page))


@app.route("/history/<int:item_id>/formula.png", methods=["GET"])
@login_required
def formula_image(item_id: int):
    item = get_history_item(item_id, g.user)
    if not item:
        flash("记录不存在。", "warning")
        return redirect(url_for("history_page"))

    png_bytes = render_formula_png_bytes(_normalize_formula_for_storage(item["recognized_text"]))
    return send_file(
        BytesIO(png_bytes),
        mimetype="image/png",
        download_name=f"formula_{item_id}.png",
    )


@app.route("/history/<int:item_id>/word.txt", methods=["GET"])
@login_required
def download_word_formula(item_id: int):
    item = get_history_item(item_id, g.user)
    if not item:
        flash("记录不存在。", "warning")
        return redirect(url_for("history_page"))

    content = to_word_linear_formula(_normalize_formula_for_storage(item["recognized_text"])).encode("utf-8")
    return send_file(
        BytesIO(content),
        mimetype="text/plain; charset=utf-8",
        as_attachment=True,
        download_name=f"word_formula_{item_id}.txt",
    )


@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    image_file = request.files.get("image")
    if image_file is None or not image_file.filename:
        return jsonify(success=False, error="请选择图片文件。"), 400

    img = _read_request_image(image_file)
    if img is None:
        return jsonify(success=False, error="图片读取失败，请重新选择文件。"), 400

    save_image_unicode(UPLOAD_ORIGINAL_PATH, img)
    preprocess_and_save(img, UPLOAD_IMAGE_PATH)

    history_original_path, history_processed_path = _make_history_paths("upload")
    save_image_unicode(history_original_path, img)
    preprocess_and_save(img, history_processed_path)

    recognized_text = equation_solver_function(str(UPLOAD_IMAGE_PATH))
    if not recognized_text:
        recognized_text = "空白图片，请重新输入。"

    history_id = record_history(
        user_id=g.user["id"],
        source_type="upload",
        original_path=_relative_static_path(history_original_path),
        processed_path=_relative_static_path(history_processed_path),
        recognized_text=recognized_text,
        word_linear_text=to_word_linear_formula(recognized_text),
        word_mathml_text=to_word_mathml(recognized_text),
    )

    payload = _build_result_payload(history_id, recognized_text)
    payload["original_preview"] = "static/equation_original.png"
    payload["processed_preview"] = "static/equation.png"
    return jsonify(payload)


@app.route("/predict_upload_image", methods=["GET"])
@login_required
def predict_upload_image():
    if not UPLOAD_IMAGE_PATH.exists():
        return jsonify("请先上传图片或在画板上书写公式。")

    equation = equation_solver_function(str(UPLOAD_IMAGE_PATH))
    if not equation:
        equation = "空白图片，请重新输入。"
    else:
        equation = _to_latex_output(equation)
    return jsonify(equation)


@app.route("/solve_equation_func", methods=["POST"])
@login_required
def solve_equation_func():
    input_text = _normalize_formula_for_storage(request.form["inputequation"])
    result = solve_equation(input_text)
    return jsonify(str(result))


@app.route("/corrections", methods=["POST"])
@login_required
def submit_correction():
    data = request.get_json(silent=True) or {}
    try:
        history_id = int(data.get("history_id"))
    except (TypeError, ValueError):
        return jsonify(success=False, error="缺少可纠错的识别记录。"), 400

    correct_text = str(data.get("correct_text", "")).strip()
    if not correct_text:
        return jsonify(success=False, error="请输入或选择正确公式。"), 400

    normalized_correct_text = _normalize_formula_for_storage(correct_text)

    updated_item = update_history_correction(
        user=g.user,
        item_id=history_id,
        correct_text=normalized_correct_text,
        word_linear_text=to_word_linear_formula(normalized_correct_text),
        word_mathml_text=to_word_mathml(normalized_correct_text),
    )
    if not updated_item:
        return jsonify(success=False, error="识别记录不存在或无权修改。"), 404

    wrong_text = updated_item.get("correction_wrong_text") or updated_item.get("original_recognized_text") or ""
    record_correction(
        user_id=g.user["id"],
        wrong_text=wrong_text,
        correct_text=normalized_correct_text,
        source_type=updated_item.get("source_type", ""),
    )

    payload = _build_result_payload(history_id, normalized_correct_text, correction_suggestions=[])
    payload["original_recognized_text"] = wrong_text
    payload["is_corrected"] = True
    return jsonify(payload)


@app.route("/recognize_sketch", methods=["POST"])
@login_required
def recognize_sketch():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify(success=False, latex="", error="未收到图片数据。"), 400

    img_b64 = data["image"]
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]

    img_bytes = base64.b64decode(img_b64)
    img = decode_image_bytes(img_bytes)
    if img is None:
        return jsonify(success=False, latex="", error="图片解码失败。"), 400

    save_image_unicode(SKETCH_ORIGINAL_PATH, img)
    preprocess_and_save(img, SKETCH_IMAGE_PATH)

    history_original_path, history_processed_path = _make_history_paths("sketch")
    save_image_unicode(history_original_path, img)
    preprocess_and_save(img, history_processed_path)

    recognized_text = equation_solver_function(str(SKETCH_IMAGE_PATH))
    if not recognized_text:
        recognized_text = "空白图片，请重新输入。"

    history_id = record_history(
        user_id=g.user["id"],
        source_type="sketch",
        original_path=_relative_static_path(history_original_path),
        processed_path=_relative_static_path(history_processed_path),
        recognized_text=recognized_text,
        word_linear_text=to_word_linear_formula(recognized_text),
        word_mathml_text=to_word_mathml(recognized_text),
    )
    return jsonify(_build_result_payload(history_id, recognized_text))


init_db()


if __name__ == "__main__":
    app.run(debug=True)
