import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.config import config
from app.services.batch_video import BatchItemStatus
from webui import batch_page
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


def test_completed_batch_history_does_not_need_refresh():
    batches = [SimpleNamespace(items=[SimpleNamespace(status="complete")])]
    needs_refresh = getattr(
        batch_page,
        "batch_history_needs_refresh",
        lambda _batches: True,
    )

    assert needs_refresh(batches) is False


def test_processing_batch_history_needs_refresh():
    batches = [SimpleNamespace(items=[SimpleNamespace(status="video")])]
    needs_refresh = getattr(
        batch_page,
        "batch_history_needs_refresh",
        lambda _batches: False,
    )

    assert needs_refresh(batches) is True


def test_video_download_reader_does_not_read_until_invoked(tmp_path):
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"initial")
    reader_factory = getattr(batch_page, "video_download_reader", None)

    assert reader_factory is not None
    reader = reader_factory(video_file)
    video_file.write_bytes(b"updated")

    assert callable(reader)
    assert reader() == b"updated"


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


def test_completed_batch_item_only_renders_video_download(tmp_path):
    video_file = tmp_path / "preview.mp4"
    video_file.write_bytes(b"preview")
    app = AppTest.from_string(
        f"""
from datetime import datetime

from app.services.batch_video import BatchItemStatus, BatchItemView, BatchView
from webui.batch_page import render_batch_table

item = BatchItemView(
    item_id="item-1",
    task_id="task-1",
    position=1,
    keyword="preview",
    status=BatchItemStatus.complete,
    progress=100,
    videos=[{str(video_file)!r}],
)
batch = BatchView(
    batch_id="batch-1",
    created_at=datetime.now(),
    updated_at=datetime.now(),
    settings_summary={{}},
    items=[item],
)
render_batch_table(batch, lambda value: value, None, lambda *args: "preview.mp4", "all")
"""
    ).run()

    assert not app.exception
    assert len(app.get("video")) == 0
    assert len(app.get("download_button")) == 1


def test_active_batch_renders_cancel_button_that_cancels_selected_batch():
    app = AppTest.from_string(
        """
from datetime import datetime
import streamlit as st

from app.services.batch_video import BatchItemStatus, BatchItemView, BatchView
from webui.batch_page import _render_batch_history_contents

class Coordinator:
    def cancel_batch(self, batch_id):
        st.session_state["cancelled_batch"] = batch_id
        return 1

item = BatchItemView(
    item_id="item-1",
    task_id="task-1",
    position=1,
    keyword="active",
    status=BatchItemStatus.waiting,
    progress=0,
)
batch = BatchView(
    batch_id="batch-1",
    created_at=datetime.now(),
    updated_at=datetime.now(),
    settings_summary={},
    items=[item],
)
_render_batch_history_contents(
    [batch], [], lambda value: value, Coordinator(), lambda *args: "video.mp4"
)
"""
    ).run()

    cancel_button = next(
        button for button in app.button if button.key == "cancel_batch_batch-1"
    )
    cancel_button.click().run()

    assert not app.exception
    assert app.session_state["cancelled_batch"] == "batch-1"
