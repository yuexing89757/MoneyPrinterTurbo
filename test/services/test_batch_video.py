import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.models import const
from app.models.schema import VideoParams
from app.services.state import MemoryState
from app.services.batch_store import BatchAttempt, BatchStore
from app.services.batch_video import (
    BatchCoordinator,
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


def test_coordinator_creates_one_independent_task_per_keyword(tmp_path):
    submitted = []

    def submitter(entries, capture_logs=True):
        submitted.extend(entries)

    coordinator = BatchCoordinator(
        store=BatchStore(tmp_path / "batches"),
        tasks_root=tmp_path / "tasks",
        state=MemoryState(),
        submitter=submitter,
        process_owner="owner-a",
    )
    base = VideoParams(video_subject="", video_script_prompt="语气温和")

    record = coordinator.create_and_submit("自律\n复利", base, capture_logs=False)

    assert [params.video_subject for _, params in submitted] == ["自律", "复利"]
    assert len({task_id for task_id, _ in submitted}) == 2
    assert all("700 至 900" in params.video_script_prompt for _, params in submitted)
    assert base.video_subject == ""
    assert base.video_script_prompt == "语气温和"
    assert [len(item.attempts) for item in record.items] == [1, 1]
    assert coordinator.store.load(record.batch_id) == record


def test_coordinator_removes_new_record_when_atomic_submission_fails(tmp_path):
    def reject(_entries, capture_logs=True):
        raise ValueError("queue full")

    coordinator = BatchCoordinator(
        store=BatchStore(tmp_path / "batches"),
        tasks_root=tmp_path / "tasks",
        state=MemoryState(),
        submitter=reject,
        process_owner="owner-a",
    )

    with pytest.raises(ValueError, match="queue full"):
        coordinator.create_and_submit("自律", VideoParams(video_subject=""))

    records, warnings = coordinator.store.list_records()
    assert records == []
    assert warnings == []


def test_retry_appends_attempt_and_preserves_old_task(tmp_path):
    submitted = []

    def submitter(entries, capture_logs=True):
        submitted.extend(entries)

    state = MemoryState()
    coordinator = BatchCoordinator(
        store=BatchStore(tmp_path / "batches"),
        tasks_root=tmp_path / "tasks",
        state=state,
        submitter=submitter,
        process_owner="owner-a",
    )
    record = coordinator.create_and_submit("沉没成本", VideoParams(video_subject=""))
    old_task_id = record.items[0].attempts[-1].task_id
    state.update_task(
        old_task_id,
        state=const.TASK_STATE_FAILED,
        failed_stage="script",
        error="model unavailable",
    )

    updated = coordinator.retry_item(record.batch_id, record.items[0].item_id)

    attempts = updated.items[0].attempts
    assert len(attempts) == 2
    assert attempts[0].task_id == old_task_id
    assert attempts[1].task_id != old_task_id
    assert submitted[-1][1].video_subject == "沉没成本"


def test_list_batch_views_marks_old_owner_as_interrupted(tmp_path):
    state = MemoryState()
    coordinator = BatchCoordinator(
        store=BatchStore(tmp_path / "batches"),
        tasks_root=tmp_path / "tasks",
        state=state,
        submitter=lambda entries, capture_logs=True: None,
        process_owner="old-owner",
    )
    record = coordinator.create_and_submit("机会成本", VideoParams(video_subject=""))
    task_id = record.items[0].attempts[-1].task_id
    state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    restarted = BatchCoordinator(
        store=coordinator.store,
        tasks_root=tmp_path / "tasks",
        state=state,
        submitter=lambda entries, capture_logs=True: None,
        process_owner="new-owner",
    )
    batches, warnings = restarted.list_batch_views()

    assert warnings == []
    assert batches[0].items[0].status == "interrupted"
