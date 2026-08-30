import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.models import const
from app.services import llm
from app.services.batch_store import BatchItem, BatchRecord


MAX_BATCH_KEYWORDS = 100
MAX_KEYWORD_LENGTH = 500
MAX_SCRIPT_FILE_BYTES = 1024 * 1024


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


def _final_videos(task_dir: Path) -> list[str]:
    if not task_dir.is_dir():
        return []
    return [
        str(path.resolve())
        for path in sorted(task_dir.glob("final-*.mp4"))
        if path.is_file() and path.resolve().parent == task_dir
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
    common = {
        "item_id": item.item_id,
        "task_id": attempt.task_id,
        "position": item.position,
        "keyword": item.keyword,
        "script_summary": script[:160],
        "script_length": len(script),
        "videos": videos,
    }
    if videos:
        return BatchItemView(
            **common,
            status=BatchItemStatus.complete,
            progress=100,
        )

    runtime = dict(runtime_task or {})
    state = runtime.get("state")
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
