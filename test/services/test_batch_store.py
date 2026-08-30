import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.models.schema import VideoParams
from app.services.batch_store import BatchAttempt, BatchStore


def test_create_save_and_load_preserves_duplicate_keyword_order(tmp_path):
    store = BatchStore(tmp_path)
    params = VideoParams(video_subject="", video_source="pexels")
    record = store.create(["自律", "长期主义", "自律"], params)

    store.save(record)
    restored = store.load(record.batch_id)

    assert [item.position for item in restored.items] == [1, 2, 3]
    assert [item.keyword for item in restored.items] == ["自律", "长期主义", "自律"]
    assert restored.params_snapshot["video_source"] == "pexels"
    assert all(item.attempts == [] for item in restored.items)


def test_batch_json_excludes_subject_script_and_temporary_audio(tmp_path):
    store = BatchStore(tmp_path)
    params = VideoParams(
        video_subject="secret subject",
        video_script="prepared script",
        custom_audio_file="C:/temporary/private.wav",
    )
    record = store.create(["边界感"], params)

    store.save(record)
    payload = json.loads(
        (tmp_path / f"{record.batch_id}.json").read_text(encoding="utf-8")
    )

    assert payload["params_snapshot"]["video_subject"] == ""
    assert payload["params_snapshot"]["video_script"] == ""
    assert payload["params_snapshot"]["custom_audio_file"] is None
    serialized = json.dumps(payload).lower()
    assert "api_key" not in serialized
    assert "token" not in serialized


def test_list_records_isolates_invalid_json(tmp_path):
    store = BatchStore(tmp_path)
    valid = store.create(["复利"], VideoParams(video_subject=""))
    store.save(valid)
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")

    records, warnings = store.list_records()

    assert [record.batch_id for record in records] == [valid.batch_id]
    assert len(warnings) == 1
    assert warnings[0].filename == "broken.json"


def test_append_attempt_keeps_previous_attempts(tmp_path):
    store = BatchStore(tmp_path)
    record = store.create(["沉没成本"], VideoParams(video_subject=""))
    store.save(record)
    first = BatchAttempt(
        task_id=str(uuid4()),
        created_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        process_owner="one",
    )
    second = BatchAttempt(
        task_id=str(uuid4()),
        created_at=datetime(2026, 8, 30, 0, 1, tzinfo=timezone.utc),
        process_owner="two",
    )

    store.append_attempt(record.batch_id, record.items[0].item_id, first)
    updated = store.append_attempt(record.batch_id, record.items[0].item_id, second)

    assert updated.items[0].attempts == [first, second]


def test_load_rejects_non_uuid_path_components(tmp_path):
    store = BatchStore(tmp_path)

    with pytest.raises(ValueError, match="invalid batch id"):
        store.load("../config")
