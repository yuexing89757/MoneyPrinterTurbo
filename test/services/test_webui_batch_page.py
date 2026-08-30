import ast
from pathlib import Path
from types import SimpleNamespace

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
