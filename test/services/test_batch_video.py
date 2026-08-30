import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.models import const
from app.models.schema import VideoParams
from app.services.batch_store import BatchAttempt, BatchStore
from app.services.batch_video import (
    build_knowledge_script_prompt,
    derive_item_view,
    parse_keywords,
)


def _record_with_attempt(tmp_path, owner="current"):
    store = BatchStore(tmp_path / "batches")
    record = store.create(["自律"], VideoParams(video_subject=""))
    attempt = BatchAttempt(
        task_id=str(uuid4()),
        created_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        process_owner=owner,
    )
    record.items[0].attempts.append(attempt)
    return record, record.items[0], attempt


def test_parse_keywords_ignores_blank_lines_and_preserves_duplicates():
    assert parse_keywords(" 自律 \n\n长期主义\n自律\n") == [
        "自律",
        "长期主义",
        "自律",
    ]


def test_parse_keywords_rejects_empty_and_more_than_one_hundred_items():
    with pytest.raises(ValueError, match="at least one"):
        parse_keywords(" \n\t")
    with pytest.raises(ValueError, match="at most 100"):
        parse_keywords("\n".join(f"keyword-{index}" for index in range(101)))


def test_parse_keywords_rejects_a_subject_over_api_limit():
    with pytest.raises(ValueError, match="at most 500"):
        parse_keywords("词" * 501)


def test_prompt_requires_three_topics_and_preserves_style():
    prompt = build_knowledge_script_prompt("语气温和")

    for phrase in ("700 至 900", "出处", "含义", "现实生活", "不得虚构", "语气温和"):
        assert phrase in prompt
    assert "Markdown" in prompt


def test_prompt_rejects_a_combination_over_existing_prompt_limit():
    with pytest.raises(ValueError, match="exceed 2000"):
        build_knowledge_script_prompt("风" * 2000)


@pytest.mark.parametrize(
    ("progress", "expected"),
    [(0, "waiting"), (5, "script"), (20, "audio"), (40, "materials"), (50, "video")],
)
def test_processing_progress_maps_to_batch_stage(tmp_path, progress, expected):
    record, item, _ = _record_with_attempt(tmp_path)
    runtime = {"state": const.TASK_STATE_PROCESSING, "progress": progress}

    view = derive_item_view(record, item, runtime, tmp_path / "tasks", "current")

    assert view.status == expected
    assert view.progress == progress


def test_old_process_processing_task_becomes_interrupted(tmp_path):
    record, item, _ = _record_with_attempt(tmp_path, owner="old-process")
    runtime = {"state": const.TASK_STATE_PROCESSING, "progress": 50}

    view = derive_item_view(record, item, runtime, tmp_path / "tasks", "new-process")

    assert view.status == "interrupted"


def test_failed_task_preserves_stage_and_error(tmp_path):
    record, item, _ = _record_with_attempt(tmp_path)
    runtime = {
        "state": const.TASK_STATE_FAILED,
        "progress": 20,
        "failed_stage": "audio",
        "error": "voice unavailable",
    }

    view = derive_item_view(record, item, runtime, tmp_path / "tasks", "current")

    assert view.status == "failed"
    assert view.failed_stage == "audio"
    assert view.error == "voice unavailable"


def test_existing_final_video_and_script_win_over_stale_runtime_state(tmp_path):
    record, item, attempt = _record_with_attempt(tmp_path, owner="old-process")
    task_dir = tmp_path / "tasks" / attempt.task_id
    task_dir.mkdir(parents=True)
    final_video = task_dir / "final-1.mp4"
    final_video.write_bytes(b"video")
    script = "这是一段已经生成的完整旁白。"
    (task_dir / "script.json").write_text(
        json.dumps({"script": script}, ensure_ascii=False), encoding="utf-8"
    )

    view = derive_item_view(
        record,
        item,
        {"state": const.TASK_STATE_PROCESSING, "progress": 50},
        tmp_path / "tasks",
        "new-process",
    )

    assert view.status == "complete"
    assert view.progress == 100
    assert view.videos == [str(final_video.resolve())]
    assert view.script_summary == script
    assert view.script_length == len(script)
