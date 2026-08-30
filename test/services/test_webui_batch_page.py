import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.config import config
from app.services.batch_video import BatchItemStatus
from webui.batch_page import filter_batch_items, status_value


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"
WEBUI_BATCH_PAGE = ROOT_DIR / "webui" / "batch_page.py"


def _attribute_name(node):
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
    return ".".join(reversed(names))


def test_batch_status_filter_preserves_input_order():
    items = [
        SimpleNamespace(position=1, status="complete"),
        SimpleNamespace(position=2, status="failed"),
        SimpleNamespace(position=3, status="complete"),
    ]

    assert [item.position for item in filter_batch_items(items, "complete")] == [1, 3]
    assert filter_batch_items(items, "all") == items


def test_batch_status_filter_accepts_typed_status_values():
    item = SimpleNamespace(position=1, status=BatchItemStatus.complete)

    assert filter_batch_items([item], "complete") == [item]
    assert status_value(item.status) == "complete"


def test_batch_module_never_calls_media_pipeline_directly():
    tree = ast.parse(WEBUI_BATCH_PAGE.read_text(encoding="utf-8"))
    calls = {
        _attribute_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    assert "tm.start" not in calls
    assert "task.start" not in calls


def test_main_routes_to_batch_application():
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    calls = {
        _attribute_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    assert "_render_batch_application" in function_names
    assert "_render_batch_application" in calls
    assert "batch_page.render_batch_history" in calls


def test_batch_mode_saves_runtime_config_before_returning():
    app = AppTest.from_file(WEBUI_MAIN, default_timeout=60)
    app.session_state["ui_language"] = "en"

    with patch.object(config, "try_save_config", return_value=True) as save_config:
        app.run()
        save_config.reset_mock()

        generation_mode = next(
            item
            for item in app.get("button_group")
            if item.key == "generation_mode"
        )
        generation_mode.select("Batch Generation").run()

    save_config.assert_called_once_with()
