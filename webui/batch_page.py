import mimetypes
import os
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

import streamlit as st

from app.services.batch_video import BatchCoordinator, BatchItemView, BatchView


STATUS_FILTERS = ("all", "processing", "complete", "failed", "interrupted")
PROCESSING_STATUSES = {"waiting", "script", "audio", "materials", "video"}


def status_value(status) -> str:
    return str(getattr(status, "value", status))


def filter_batch_items(
    items: Iterable[BatchItemView], status_filter: str
) -> list[BatchItemView]:
    values = list(items)
    if status_filter == "all":
        return values
    if status_filter == "processing":
        return [
            item for item in values if status_value(item.status) in PROCESSING_STATUSES
        ]
    return [item for item in values if status_value(item.status) == status_filter]


def batch_history_needs_refresh(batches: Iterable[BatchView]) -> bool:
    return any(
        status_value(item.status) in PROCESSING_STATUSES
        for batch in batches
        for item in batch.items
    )


def video_download_reader(video_path: str | os.PathLike[str]) -> Callable[[], bytes]:
    return Path(video_path).read_bytes


def render_keyword_input(tr: Callable[[str], str]) -> str:
    raw = st.text_area(
        tr("Batch Keywords"),
        placeholder=tr("Batch Keywords Placeholder"),
        height=220,
        key="batch_keywords",
    )
    count = len([line for line in raw.splitlines() if line.strip()])
    st.caption(tr("Batch Keyword Count").format(count=count))
    return raw


def _status_label(status: str, tr: Callable[[str], str]) -> str:
    return tr(f"Batch Status {status.title()}")


def render_batch_table(
    batch: BatchView,
    tr: Callable[[str], str],
    coordinator: BatchCoordinator,
    download_name_builder: Callable[[str, int, int], str],
    status_filter: str,
) -> None:
    items = filter_batch_items(batch.items, status_filter)
    if not items:
        st.info(tr("No Tasks Match Filter"))
        return

    for item in items:
        with st.container(border=True, key=f"batch_item_{item.item_id}"):
            columns = st.columns([0.45, 1.6, 1.1, 2.8, 1.0])
            columns[0].write(item.position)
            columns[1].write(item.keyword)
            current_status = status_value(item.status)
            columns[2].write(_status_label(current_status, tr))
            with columns[3]:
                st.progress(item.progress, text=f"{item.progress}%")
                if item.script_summary:
                    st.caption(
                        tr("Batch Script Summary").format(
                            count=item.script_length,
                            summary=item.script_summary,
                        )
                    )
                if item.error:
                    st.error(item.error)
            with columns[4]:
                if current_status in {"failed", "interrupted"}:
                    if st.button(
                        tr("Retry"),
                        key=f"retry_batch_{batch.batch_id}_{item.item_id}",
                        use_container_width=True,
                    ):
                        coordinator.retry_item(batch.batch_id, item.item_id)
                        st.rerun()

            if current_status == "complete":
                for index, video_path in enumerate(item.videos, start=1):
                    if not os.path.isfile(video_path):
                        continue
                    st.download_button(
                        tr("Download Video"),
                        data=video_download_reader(video_path),
                        file_name=download_name_builder(
                            item.keyword,
                            index,
                            len(item.videos),
                        ),
                        mime=mimetypes.guess_type(video_path)[0] or "video/mp4",
                        key=f"batch_download_{item.task_id}_{index}",
                        icon=":material/download:",
                        on_click="ignore",
                        use_container_width=True,
                    )


def _render_batch_history_contents(
    batches: list[BatchView],
    warnings,
    tr: Callable[[str], str],
    coordinator: BatchCoordinator,
    download_name_builder: Callable[[str, int, int], str],
) -> None:
    for warning in warnings:
        st.warning(
            tr("Batch Record Load Warning").format(filename=warning.filename)
        )
    if not batches:
        st.info(tr("No Batch History"))
        return

    batch_by_id = {batch.batch_id: batch for batch in batches}
    selected_id = st.selectbox(
        tr("Batch History"),
        options=list(batch_by_id),
        format_func=lambda batch_id: datetime.fromisoformat(
            batch_by_id[batch_id].created_at.isoformat()
        ).strftime("%Y-%m-%d %H:%M:%S"),
        key="selected_batch_id",
    )
    filter_value = st.selectbox(
        tr("Batch Status Filter"),
        options=list(STATUS_FILTERS),
        format_func=lambda value: tr(f"Batch Filter {value.title()}"),
        key="batch_status_filter",
    )
    selected_batch = batch_by_id[selected_id]
    if batch_history_needs_refresh([selected_batch]) and st.button(
        tr("Cancel Batch"),
        key=f"cancel_batch_{selected_id}",
        icon=":material/cancel:",
        use_container_width=True,
    ):
        coordinator.cancel_batch(selected_id)
        st.rerun()
    render_batch_table(
        selected_batch,
        tr,
        coordinator,
        download_name_builder,
        filter_value,
    )


@st.fragment(run_every=1.0)
def _render_live_batch_history(
    coordinator: BatchCoordinator,
    tr: Callable[[str], str],
    download_name_builder: Callable[[str, int, int], str],
) -> None:
    batches, warnings = coordinator.list_batch_views()
    _render_batch_history_contents(
        batches,
        warnings,
        tr,
        coordinator,
        download_name_builder,
    )
    if not batch_history_needs_refresh(batches):
        st.rerun()


def render_batch_history(
    coordinator: BatchCoordinator,
    tr: Callable[[str], str],
    download_name_builder: Callable[[str, int, int], str],
) -> None:
    batches, warnings = coordinator.list_batch_views()
    if batch_history_needs_refresh(batches):
        _render_live_batch_history(
            coordinator,
            tr,
            download_name_builder,
        )
        return
    _render_batch_history_contents(
        batches,
        warnings,
        tr,
        coordinator,
        download_name_builder,
    )
