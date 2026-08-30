import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.models.schema import VideoParams
from app.utils import utils


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BatchAttempt(BaseModel):
    task_id: str
    created_at: datetime
    process_owner: str


class BatchItem(BaseModel):
    item_id: str
    position: int
    keyword: str
    attempts: list[BatchAttempt] = Field(default_factory=list)


class BatchRecord(BaseModel):
    schema_version: Literal[1] = 1
    batch_id: str
    created_at: datetime
    updated_at: datetime
    settings_summary: dict[str, Any]
    params_snapshot: dict[str, Any]
    items: list[BatchItem]


class BatchLoadWarning(BaseModel):
    filename: str
    error: str


class BatchStore:
    def __init__(self, root: str | Path | None = None):
        default_root = Path(utils.storage_dir("batches", create=True))
        self.root = Path(root) if root is not None else default_root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _normalize_batch_id(batch_id: str) -> str:
        try:
            normalized = str(UUID(str(batch_id)))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("invalid batch id") from exc
        if normalized != str(batch_id).lower():
            raise ValueError("invalid batch id")
        return normalized

    def _record_path(self, batch_id: str) -> Path:
        normalized = self._normalize_batch_id(batch_id)
        return self.root / f"{normalized}.json"

    @staticmethod
    def _params_snapshot(params: VideoParams) -> dict[str, Any]:
        snapshot = {
            field: value
            for field, value in params.model_dump(mode="json", warnings=False).items()
            if field in VideoParams.model_fields
        }
        snapshot["video_subject"] = ""
        snapshot["video_script"] = ""
        snapshot["custom_audio_file"] = None
        return snapshot

    @staticmethod
    def _settings_summary(params: VideoParams) -> dict[str, Any]:
        values = params.model_dump(mode="json", warnings=False)
        return {
            "video_source": values.get("video_source"),
            "video_aspect": values.get("video_aspect"),
            "voice_name": values.get("voice_name"),
            "video_count": values.get("video_count"),
            "requires_custom_audio": bool(params.custom_audio_file),
        }

    def create(self, keywords: list[str], params: VideoParams) -> BatchRecord:
        now = _utc_now()
        return BatchRecord(
            batch_id=str(uuid4()),
            created_at=now,
            updated_at=now,
            settings_summary=self._settings_summary(params),
            params_snapshot=self._params_snapshot(params),
            items=[
                BatchItem(item_id=str(uuid4()), position=index, keyword=keyword)
                for index, keyword in enumerate(keywords, start=1)
            ],
        )

    def save(self, record: BatchRecord) -> None:
        destination = self._record_path(record.batch_id)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        payload = record.model_dump_json(indent=2)
        with self._lock:
            try:
                with temporary.open("w", encoding="utf-8", newline="\n") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)

    def load(self, batch_id: str) -> BatchRecord:
        path = self._record_path(batch_id)
        with self._lock:
            return BatchRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_records(
        self,
    ) -> tuple[list[BatchRecord], list[BatchLoadWarning]]:
        records: list[BatchRecord] = []
        warnings: list[BatchLoadWarning] = []
        with self._lock:
            paths = list(self.root.glob("*.json"))
            for path in paths:
                try:
                    records.append(self.load(path.stem))
                except Exception as exc:
                    warnings.append(
                        BatchLoadWarning(filename=path.name, error=str(exc))
                    )
        records.sort(key=lambda record: record.created_at, reverse=True)
        return records, warnings

    def append_attempt(
        self,
        batch_id: str,
        item_id: str,
        attempt: BatchAttempt,
    ) -> BatchRecord:
        with self._lock:
            record = self.load(batch_id)
            item = next((value for value in record.items if value.item_id == item_id), None)
            if item is None:
                raise ValueError("batch item not found")
            item.attempts.append(attempt)
            record.updated_at = _utc_now()
            self.save(record)
            return record

    def delete(self, batch_id: str) -> None:
        with self._lock:
            self._record_path(batch_id).unlink(missing_ok=True)
