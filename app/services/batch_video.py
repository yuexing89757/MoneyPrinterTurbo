import json
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.models import const
from app.models.schema import VideoParams
from app.services import llm
from app.services.batch_store import (
    BatchAttempt,
    BatchItem,
    BatchLoadWarning,
    BatchRecord,
    BatchStore,
)
from app.utils import utils


MAX_BATCH_KEYWORDS = 100
MAX_KEYWORD_LENGTH = 500
MAX_SCRIPT_FILE_BYTES = 1024 * 1024
BATCH_PROCESS_OWNER = str(uuid4())


def parse_keywords(raw: str) -> list[str]:
    keywords = [line.strip() for line in str(raw or "").splitlines() if line.strip()]
    if not keywords:
        raise ValueError("enter at least one keyword")
    if len(keywords) > MAX_BATCH_KEYWORDS:
        raise ValueError("a batch accepts at most 100 keywords")
    if any(len(keyword) > MAX_KEYWORD_LENGTH for keyword in keywords):
        raise ValueError("each keyword must contain at most 500 characters")
    return keywords


def build_knowledge_script_prompt(existing_style: str = "") -> str:
    style = str(existing_style or "").strip()
    rules = (
        "请围绕当前关键词撰写适合中文视频连续旁白的知识解读文案。"
        "全文以 800 个中文字为目标，接受 700 至 900 个中文字。"
        "内容依次自然说明该词的出处或形成背景、核心含义，以及它对现实生活的启发。"
        "不得输出 Markdown、章节标题、字数说明或模型说明。"
        "若没有可靠或公认出处，请明确说明；不得虚构典故、人物、年代或来源。"
        "现代术语、网络用语和外来词应说明传播背景与常见用法，不得强行附会古代典故。"
    )
    combined = f"{style}\n\n{rules}" if style else rules
    if len(combined) > llm.MAX_SCRIPT_PROMPT_LENGTH:
        raise ValueError("batch script style and required rules exceed 2000 characters")
    return combined


class BatchItemStatus(str, Enum):
    waiting = "waiting"
    script = "script"
    audio = "audio"
    materials = "materials"
    video = "video"
    complete = "complete"
    failed = "failed"
    interrupted = "interrupted"


class BatchItemView(BaseModel):
    item_id: str
    task_id: str | None
    position: int
    keyword: str
    status: BatchItemStatus
    progress: int
    script_summary: str = ""
    script_length: int = 0
    failed_stage: str = ""
    error: str = ""
    videos: list[str] = Field(default_factory=list)


class BatchView(BaseModel):
    batch_id: str
    created_at: datetime
    updated_at: datetime
    settings_summary: dict[str, Any]
    items: list[BatchItemView]


class BatchUploadRequiredError(ValueError):
    pass


def _safe_task_dir(tasks_root: Path, task_id: str) -> Path:
    try:
        normalized_id = str(UUID(str(task_id)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("invalid task id") from exc
    root = tasks_root.resolve()
    task_dir = (root / normalized_id).resolve()
    if task_dir.parent != root:
        raise ValueError("task directory is outside the task root")
    return task_dir


def _read_script(task_dir: Path) -> str:
    path = task_dir / "script.json"
    try:
        if not path.is_file() or path.stat().st_size > MAX_SCRIPT_FILE_BYTES:
            return ""
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("script") or "").strip()


def _is_complete_mp4(path: Path) -> bool:
    try:
        file_size = path.stat().st_size
        offset = 0
        found_moov = False
        with path.open("rb") as video_file:
            while offset + 8 <= file_size:
                video_file.seek(offset)
                header = video_file.read(8)
                if len(header) != 8:
                    return False
                box_size = int.from_bytes(header[:4], "big")
                box_type = header[4:]
                header_size = 8
                if box_size == 1:
                    extended_size = video_file.read(8)
                    if len(extended_size) != 8:
                        return False
                    box_size = int.from_bytes(extended_size, "big")
                    header_size = 16
                elif box_size == 0:
                    box_size = file_size - offset
                if box_size < header_size or offset + box_size > file_size:
                    return False
                found_moov = found_moov or box_type == b"moov"
                offset += box_size
        return found_moov and offset == file_size
    except OSError:
        return False


def _final_videos(task_dir: Path) -> list[str]:
    if not task_dir.is_dir():
        return []
    return [
        str(path.resolve())
        for path in sorted(task_dir.glob("final-*.mp4"))
        if path.is_file()
        and path.resolve().parent == task_dir
        and _is_complete_mp4(path)
    ]


def _processing_status(progress: int) -> BatchItemStatus:
    if progress < 5:
        return BatchItemStatus.waiting
    if progress < 20:
        return BatchItemStatus.script
    if progress < 40:
        return BatchItemStatus.audio
    if progress < 50:
        return BatchItemStatus.materials
    return BatchItemStatus.video


def derive_item_view(
    record: BatchRecord,
    item: BatchItem,
    runtime_task: Mapping[str, Any] | None,
    tasks_root: Path,
    current_process_owner: str,
) -> BatchItemView:
    del record
    attempt = item.attempts[-1] if item.attempts else None
    if attempt is None:
        return BatchItemView(
            item_id=item.item_id,
            task_id=None,
            position=item.position,
            keyword=item.keyword,
            status=BatchItemStatus.waiting,
            progress=0,
        )

    task_dir = _safe_task_dir(Path(tasks_root), attempt.task_id)
    videos = _final_videos(task_dir)
    script = _read_script(task_dir)
    runtime = dict(runtime_task or {})
    state = runtime.get("state")
    published_videos = (
        videos
        if attempt.cancelled_at is None
        and (
            state == const.TASK_STATE_COMPLETE
            or attempt.process_owner != current_process_owner
        )
        else []
    )
    common = {
        "item_id": item.item_id,
        "task_id": attempt.task_id,
        "position": item.position,
        "keyword": item.keyword,
        "script_summary": script[:160],
        "script_length": len(script),
        "videos": published_videos,
    }
    if attempt.cancelled_at is not None:
        return BatchItemView(
            **common,
            status=BatchItemStatus.interrupted,
            progress=max(0, min(100, int(runtime.get("progress", 0) or 0))),
        )
    if published_videos:
        return BatchItemView(
            **common,
            status=BatchItemStatus.complete,
            progress=100,
        )

    progress = max(0, min(100, int(runtime.get("progress", 0) or 0)))
    if state == const.TASK_STATE_FAILED:
        return BatchItemView(
            **common,
            status=BatchItemStatus.failed,
            progress=progress,
            failed_stage=str(runtime.get("failed_stage") or ""),
            error=str(runtime.get("error") or ""),
        )
    if attempt.process_owner != current_process_owner:
        return BatchItemView(
            **common,
            status=BatchItemStatus.interrupted,
            progress=progress,
        )
    if state == const.TASK_STATE_COMPLETE:
        return BatchItemView(
            **common,
            status=BatchItemStatus.failed,
            progress=progress,
            error="generated video is unavailable",
        )
    return BatchItemView(
        **common,
        status=_processing_status(progress),
        progress=progress,
    )


class BatchCoordinator:
    def __init__(
        self,
        store: BatchStore | None = None,
        tasks_root: str | Path | None = None,
        state: Any = None,
        submitter: Callable[..., None] | None = None,
        canceller: Callable[[list[str]], list[str]] | None = None,
        process_owner: str = BATCH_PROCESS_OWNER,
    ):
        if state is None:
            from app.services import state as state_module

            state = state_module.state
        if submitter is None:
            from app.services import webui_task

            submitter = webui_task.submit_generations
        if canceller is None:
            from app.services import webui_task

            canceller = webui_task.cancel_generations
        self.store = store or BatchStore()
        self.tasks_root = Path(tasks_root or utils.task_dir())
        self.state = state
        self.submitter = submitter
        self.canceller = canceller
        self.process_owner = process_owner

    @staticmethod
    def _task_params(base_params: VideoParams, keyword: str) -> VideoParams:
        params = base_params.model_copy(deep=True)
        params.video_subject = keyword
        params.video_script = ""
        params.video_script_prompt = build_knowledge_script_prompt(
            base_params.video_script_prompt
        )
        return params

    def create_and_submit(
        self,
        raw_keywords: str,
        base_params: VideoParams,
        capture_logs: bool = True,
    ) -> BatchRecord:
        keywords = parse_keywords(raw_keywords)
        record = self.store.create(keywords, base_params)
        entries: list[tuple[str, VideoParams]] = []
        created_task_dirs: list[Path] = []
        now = datetime.now(timezone.utc)
        for item in record.items:
            task_id = str(uuid4())
            item.attempts.append(
                BatchAttempt(
                    task_id=task_id,
                    created_at=now,
                    process_owner=self.process_owner,
                )
            )
            params = self._task_params(base_params, item.keyword)
            if base_params.custom_audio_file:
                source = Path(base_params.custom_audio_file).resolve()
                if not source.is_file():
                    raise BatchUploadRequiredError("custom narration file is unavailable")
                if source.suffix.lower() not in {
                    ".mp3",
                    ".wav",
                    ".m4a",
                    ".aac",
                    ".flac",
                    ".ogg",
                }:
                    raise BatchUploadRequiredError("custom narration file type is unsupported")
                task_dir = _safe_task_dir(self.tasks_root, task_id)
                task_dir.mkdir(parents=True, exist_ok=False)
                created_task_dirs.append(task_dir)
                target = task_dir / f"custom-audio{source.suffix.lower()}"
                shutil.copyfile(source, target)
                params.custom_audio_file = str(target)
            entries.append((task_id, params))
        record.updated_at = now
        self.store.save(record)
        try:
            self.submitter(entries, capture_logs=capture_logs)
        except Exception:
            self.store.delete(record.batch_id)
            for task_dir in created_task_dirs:
                if task_dir.parent == self.tasks_root.resolve():
                    shutil.rmtree(task_dir, ignore_errors=True)
            raise
        return record

    def list_batch_views(
        self,
    ) -> tuple[list[BatchView], list[BatchLoadWarning]]:
        records, warnings = self.store.list_records()
        batches = []
        for record in records:
            views = []
            for item in record.items:
                task_id = item.attempts[-1].task_id if item.attempts else ""
                runtime = self.state.get_task(task_id) if task_id else None
                views.append(
                    derive_item_view(
                        record,
                        item,
                        runtime,
                        self.tasks_root,
                        self.process_owner,
                    )
                )
            batches.append(
                BatchView(
                    batch_id=record.batch_id,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    settings_summary=record.settings_summary,
                    items=views,
                )
            )
        return batches, warnings

    def retry_item(
        self,
        batch_id: str,
        item_id: str,
        capture_logs: bool = True,
    ) -> BatchRecord:
        record = self.store.load(batch_id)
        item = next((value for value in record.items if value.item_id == item_id), None)
        if item is None:
            raise ValueError("batch item not found")
        task_id = item.attempts[-1].task_id if item.attempts else ""
        runtime = self.state.get_task(task_id) if task_id else None
        view = derive_item_view(
            record,
            item,
            runtime,
            self.tasks_root,
            self.process_owner,
        )
        if view.status not in {
            BatchItemStatus.failed,
            BatchItemStatus.interrupted,
        }:
            raise ValueError("only failed or interrupted batch items can be retried")

        params = VideoParams.model_validate(record.params_snapshot)
        params = self._task_params(params, item.keyword)
        attempt = BatchAttempt(
            task_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            process_owner=self.process_owner,
        )
        retry_task_dir = None
        if record.settings_summary.get("requires_custom_audio"):
            previous_dir = _safe_task_dir(self.tasks_root, task_id)
            audio_sources = [
                path
                for path in previous_dir.glob("custom-audio.*")
                if path.is_file() and path.resolve().parent == previous_dir
            ]
            if len(audio_sources) != 1:
                raise BatchUploadRequiredError(
                    "custom narration must be uploaded again before retrying"
                )
            retry_task_dir = _safe_task_dir(self.tasks_root, attempt.task_id)
            retry_task_dir.mkdir(parents=True, exist_ok=False)
            audio_target = retry_task_dir / audio_sources[0].name
            shutil.copyfile(audio_sources[0], audio_target)
            params.custom_audio_file = str(audio_target)
        item.attempts.append(attempt)
        record.updated_at = attempt.created_at
        self.store.save(record)
        try:
            self.submitter([(attempt.task_id, params)], capture_logs=capture_logs)
        except Exception:
            item.attempts.pop()
            record.updated_at = datetime.now(timezone.utc)
            self.store.save(record)
            if (
                retry_task_dir is not None
                and retry_task_dir.parent == self.tasks_root.resolve()
            ):
                shutil.rmtree(retry_task_dir, ignore_errors=True)
            raise
        return record

    def cancel_batch(self, batch_id: str) -> int:
        record = self.store.load(batch_id)
        task_ids = []
        attempts_by_task_id = {}
        for item in record.items:
            if not item.attempts:
                continue
            attempt = item.attempts[-1]
            runtime = self.state.get_task(attempt.task_id)
            view = derive_item_view(
                record,
                item,
                runtime,
                self.tasks_root,
                self.process_owner,
            )
            if view.status in {
                BatchItemStatus.waiting,
                BatchItemStatus.script,
                BatchItemStatus.audio,
                BatchItemStatus.materials,
                BatchItemStatus.video,
            }:
                task_ids.append(attempt.task_id)
                attempts_by_task_id[attempt.task_id] = attempt

        cancelled_ids = set(self.canceller(task_ids))
        if not cancelled_ids:
            return 0

        now = datetime.now(timezone.utc)
        for task_id in cancelled_ids:
            attempt = attempts_by_task_id.get(task_id)
            if attempt is not None:
                attempt.cancelled_at = now
        record.updated_at = now
        self.store.save(record)
        return len(cancelled_ids & attempts_by_task_id.keys())
