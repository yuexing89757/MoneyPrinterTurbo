# Batch Keyword Video Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent Streamlit batch page that turns each input keyword into an independent approximately 800-Chinese-character explanatory script and then an independently downloadable video.

**Architecture:** Add a focused JSON batch store and a batch coordinator above the existing `webui_task` queue and `task.start()` media pipeline. Add atomic multi-task admission to the task manager, then render batch input, status reconciliation, retry, preview, and downloads in a dedicated Streamlit module while reusing the existing generation settings.

**Tech Stack:** Python 3.11+, Pydantic 2, Streamlit 1.59.1, pytest 9.1.1, existing MoneyPrinterTurbo task/state services.

**Spec:** `docs/superpowers/specs/2026-08-30-batch-keyword-video-design.md`

## Global Constraints

- Accept 1 to 100 non-empty keywords per batch, one input line per task; preserve duplicate keywords and input order.
- Limit each stripped keyword to 500 characters, matching the existing API subject limit.
- Target 700 to 900 Chinese characters per generated script; do not mechanically truncate LLM output.
- Each task must explain origin/background, meaning, and real-life inspiration, and must not fabricate unverifiable origins.
- Run the complete script-to-video flow automatically without a script review gate.
- Keep WebUI media generation concurrency at one and isolate a task failure from later tasks.
- Persist batch membership and a credential-free `VideoParams` snapshot under `storage/batches`.
- Never persist API keys, tokens, uploaded file bytes, or external-service credentials in a batch JSON file.
- Do not automatically resume interrupted work after process restart; require an explicit retry.
- Resolve preview and download files inside the existing `storage/tasks` allowlisted root.
- Do not add dependencies or a new public HTTP API.

---

### Task 1: Persistent batch records

**Files:**
- Create: `app/services/batch_store.py`
- Create: `test/services/test_batch_store.py`

**Interfaces:**
- Consumes: `app.models.schema.VideoParams`
- Produces: `BatchAttempt`, `BatchItem`, `BatchRecord`, `BatchLoadWarning`, and `BatchStore`
- Produces: `BatchStore.create(keywords: list[str], params: VideoParams) -> BatchRecord`
- Produces: `BatchStore.save(record: BatchRecord) -> None`
- Produces: `BatchStore.load(batch_id: str) -> BatchRecord`
- Produces: `BatchStore.list_records() -> tuple[list[BatchRecord], list[BatchLoadWarning]]`
- Produces: `BatchStore.append_attempt(batch_id: str, item_id: str, attempt: BatchAttempt) -> BatchRecord`
- Produces: `BatchStore.delete(batch_id: str) -> None`

- [ ] **Step 1: Write failing model and round-trip tests**

```python
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


def test_batch_json_does_not_persist_credentials(tmp_path):
    store = BatchStore(tmp_path)
    params = VideoParams(video_subject="")
    record = store.create(["边界感"], params)
    store.save(record)

    payload = (tmp_path / f"{record.batch_id}.json").read_text(encoding="utf-8")
    assert "api_key" not in payload.lower()
    assert "token" not in payload.lower()
```

- [ ] **Step 2: Run the new tests and verify the missing module failure**

Run: `uv run pytest test/services/test_batch_store.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'app.services.batch_store'`.

- [ ] **Step 3: Implement typed records and atomic JSON persistence**

```python
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


def save(self, record: BatchRecord) -> None:
    destination = self._record_path(record.batch_id)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    payload = record.model_dump_json(indent=2)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
```

Implement strict UUID path validation, schema validation, newest-first listing, per-file warning isolation, and deletion restricted to the configured batch root. Serialize `VideoParams.model_dump(mode="json")` through an explicit field allowlist and clear `video_subject`, `video_script`, and unrecoverable temporary `custom_audio_file` values.

- [ ] **Step 4: Add corruption, atomic-replace, attempt, and safe-delete tests**

```python
def test_list_records_isolates_invalid_json(tmp_path):
    valid = BatchStore(tmp_path).create(["复利"], VideoParams(video_subject=""))
    BatchStore(tmp_path).save(valid)
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")

    records, warnings = BatchStore(tmp_path).list_records()

    assert [record.batch_id for record in records] == [valid.batch_id]
    assert warnings[0].filename == "broken.json"


def test_append_attempt_keeps_previous_attempts(tmp_path):
    store = BatchStore(tmp_path)
    record = store.create(["沉没成本"], VideoParams(video_subject=""))
    store.save(record)
    first = BatchAttempt(task_id=str(uuid4()), created_at=utc_now(), process_owner="one")
    second = BatchAttempt(task_id=str(uuid4()), created_at=utc_now(), process_owner="two")

    store.append_attempt(record.batch_id, record.items[0].item_id, first)
    updated = store.append_attempt(record.batch_id, record.items[0].item_id, second)

    assert updated.items[0].attempts == [first, second]
```

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest test/services/test_batch_store.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add app/services/batch_store.py test/services/test_batch_store.py
git commit -m "feat(batch): persist keyword video batches"
```

---

### Task 2: Keyword parsing, script prompt, and display status

**Files:**
- Create: `app/services/batch_video.py`
- Create: `test/services/test_batch_video.py`

**Interfaces:**
- Consumes: `BatchAttempt`, `BatchItem`, and `BatchRecord` from Task 1
- Produces: `parse_keywords(raw: str) -> list[str]`
- Produces: `build_knowledge_script_prompt(existing_style: str = "") -> str`
- Produces: `BatchItemStatus` enum and `BatchItemView` model
- Produces: `derive_item_view(record: BatchRecord, item: BatchItem, runtime_task: Mapping[str, Any] | None, tasks_root: Path, current_process_owner: str) -> BatchItemView`

- [ ] **Step 1: Write failing parsing and prompt tests**

```python
def test_parse_keywords_ignores_blank_lines_and_preserves_duplicates():
    assert parse_keywords(" 自律 \n\n长期主义\n自律\n") == ["自律", "长期主义", "自律"]


def test_parse_keywords_rejects_more_than_one_hundred_items():
    with pytest.raises(ValueError, match="at most 100"):
        parse_keywords("\n".join(f"keyword-{index}" for index in range(101)))


def test_prompt_requires_three_sections_without_inventing_an_origin():
    prompt = build_knowledge_script_prompt("语气温和")
    for phrase in ("700 至 900", "出处", "含义", "现实生活", "不得虚构", "语气温和"):
        assert phrase in prompt
    assert "Markdown" in prompt
```

- [ ] **Step 2: Run parsing and prompt tests to verify they fail**

Run: `uv run pytest test/services/test_batch_video.py -k "parse or prompt" -v`

Expected: collection fails because `app.services.batch_video` does not exist.

- [ ] **Step 3: Implement parsing and prompt composition**

```python
MAX_BATCH_KEYWORDS = 100
MAX_KEYWORD_LENGTH = 500


def parse_keywords(raw: str) -> list[str]:
    keywords = [line.strip() for line in str(raw or "").splitlines() if line.strip()]
    if not keywords:
        raise ValueError("enter at least one keyword")
    if len(keywords) > MAX_BATCH_KEYWORDS:
        raise ValueError("a batch accepts at most 100 keywords")
    oversized = next((keyword for keyword in keywords if len(keyword) > 500), None)
    if oversized is not None:
        raise ValueError("each keyword must contain at most 500 characters")
    return keywords


def build_knowledge_script_prompt(existing_style: str = "") -> str:
    style = str(existing_style or "").strip()
    rules = """请围绕当前关键词撰写适合中文视频连续旁白的知识解读文案。全文以 800 个中文字为目标，接受 700 至 900 个中文字。内容依次自然说明该词的出处或形成背景、核心含义，以及它对现实生活的启发。不得输出 Markdown、章节标题、字数说明或模型说明。若没有可靠或公认出处，请明确说明；不得虚构典故、人物、年代或来源。现代术语、网络用语和外来词应说明传播背景与常见用法，不得强行附会古代典故。"""
    combined = f"{style}\n\n{rules}" if style else rules
    if len(combined) > llm.MAX_SCRIPT_PROMPT_LENGTH:
        raise ValueError("batch script style and required rules exceed 2000 characters")
    return combined
```

- [ ] **Step 4: Write failing status reconciliation tests**

```python
@pytest.mark.parametrize(
    ("progress", "expected"),
    [(0, "waiting"), (5, "script"), (20, "audio"), (40, "materials"), (50, "video")],
)
def test_processing_progress_maps_to_batch_stage(tmp_path, progress, expected):
    record, item, attempt = batch_record_with_attempt(owner="current")
    runtime = {"state": const.TASK_STATE_PROCESSING, "progress": progress}
    view = derive_item_view(record, item, runtime, tmp_path, "current")
    assert view.status == expected


def test_old_process_processing_task_becomes_interrupted(tmp_path):
    record, item, attempt = batch_record_with_attempt(owner="old-process")
    runtime = {"state": const.TASK_STATE_PROCESSING, "progress": 50}
    view = derive_item_view(record, item, runtime, tmp_path, "new-process")
    assert view.status == "interrupted"
```

- [ ] **Step 5: Implement status reconciliation and safe artifact lookup**

Use the current attempt only. Prefer an existing final video under `storage/tasks/<task-id>` over stale runtime state. Read `script.json` with bounded JSON parsing to obtain the script summary and character count. Map `failed_stage`, `error`, progress, and process owner into an immutable `BatchItemView`; reject any artifact path that resolves outside `tasks_root`.

```python
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
    error: str = ""
    videos: list[str] = Field(default_factory=list)
```

- [ ] **Step 6: Run Task 2 tests**

Run: `uv run pytest test/services/test_batch_video.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add app/services/batch_video.py test/services/test_batch_video.py
git commit -m "feat(batch): define keyword scripts and task status"
```

---

### Task 3: Atomic batch admission to the WebUI queue

**Files:**
- Modify: `app/controllers/manager/base_manager.py`
- Modify: `app/services/webui_task.py`
- Modify: `test/services/test_task_manager.py`
- Modify: `test/services/test_webui_task.py`

**Interfaces:**
- Produces: `TaskManager.add_tasks(tasks: list[dict[str, Any]]) -> None`
- Produces: `webui_task.submit_generations(entries: list[tuple[str, VideoParams]], capture_logs: bool = True) -> None`
- Preserves: `webui_task.submit_generation(...)` behavior for single-video requests

- [ ] **Step 1: Write a failing all-or-nothing queue capacity test**

```python
def test_add_tasks_rejects_whole_batch_when_queue_capacity_is_insufficient():
    manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=2)
    blocker = threading.Event()
    manager.add_task(lambda: blocker.wait(timeout=2))

    with pytest.raises(TaskQueueFullError):
        manager.add_tasks([
            {"func": lambda: None, "args": (), "kwargs": {}},
            {"func": lambda: None, "args": (), "kwargs": {}},
            {"func": lambda: None, "args": (), "kwargs": {}},
        ])

    assert manager.queue_size() == 0
    blocker.set()
```

- [ ] **Step 2: Run the focused task-manager test and verify it fails**

Run: `uv run pytest test/services/test_task_manager.py::test_add_tasks_rejects_whole_batch_when_queue_capacity_is_insufficient -v`

Expected: fails with missing `add_tasks`.

- [ ] **Step 3: Implement atomic admission**

Validate every task dictionary before acquiring the lock. Under one lock, compare `queue_size() + len(tasks)` with `max_queued_tasks`, enqueue every task only when the entire list fits, then release the lock and call `check_queue()` once. Queue all batch entries first even when a worker is free; `check_queue()` starts the first entry while preserving input order.

```python
def add_tasks(self, tasks: list[Dict[str, Any]]):
    normalized = []
    for task in tasks:
        func = task["func"]
        if not callable(func):
            raise TypeError("batch task func must be callable")
        normalized.append({
            "func": func,
            "args": tuple(task.get("args", ())),
            "kwargs": dict(task.get("kwargs", {})),
        })
    with self.lock:
        if self.queue_size() + len(normalized) > self.max_queued_tasks:
            raise TaskQueueFullError("task queue is full, please try again later")
        for task in normalized:
            self.enqueue(task)
    self.check_queue()
```

- [ ] **Step 4: Write failing WebUI batch submission tests**

```python
def test_submit_generations_registers_all_states_before_atomic_enqueue():
    entries = [
        ("task-a", VideoParams(video_subject="自律")),
        ("task-b", VideoParams(video_subject="复利")),
    ]
    with patch.object(webui_task._task_manager, "add_tasks") as add_tasks:
        webui_task.submit_generations(entries, capture_logs=False)

    assert add_tasks.call_count == 1
    assert webui_task.sm.state.get_task("task-a")["video_subject"] == "自律"
    assert webui_task.sm.state.get_task("task-b")["video_subject"] == "复利"
```

- [ ] **Step 5: Implement batch submission with deep-copied parameters**

Build the existing `_run_generation` task dictionaries, register each state as processing at progress zero, and call `add_tasks()` once. If admission raises, delete the just-created runtime states and re-raise. Refactor `submit_generation()` to delegate to `submit_generations([(task_id, params)], ...)` so single and batch paths share one implementation.

- [ ] **Step 6: Run manager and WebUI task tests**

Run: `uv run pytest test/services/test_task_manager.py test/services/test_webui_task.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add app/controllers/manager/base_manager.py app/services/webui_task.py test/services/test_task_manager.py test/services/test_webui_task.py
git commit -m "feat(batch): admit WebUI tasks atomically"
```

---

### Task 4: Batch creation, recovery, and retry coordinator

**Files:**
- Modify: `app/services/batch_video.py`
- Modify: `test/services/test_batch_video.py`

**Interfaces:**
- Consumes: `BatchStore` from Task 1 and `webui_task.submit_generations()` from Task 3
- Produces: process-level `BATCH_PROCESS_OWNER: str`
- Produces: `BatchView(batch_id: str, created_at: datetime, updated_at: datetime, items: list[BatchItemView])`
- Produces: `BatchUploadRequiredError`
- Produces: `BatchCoordinator(store: BatchStore, tasks_root: Path, state: BaseState, submitter: Callable, process_owner: str = BATCH_PROCESS_OWNER)`
- Produces: `BatchCoordinator.create_and_submit(raw_keywords: str, base_params: VideoParams, capture_logs: bool = True) -> BatchRecord`
- Produces: `BatchCoordinator.list_batch_views() -> tuple[list[BatchView], list[BatchLoadWarning]]`
- Produces: `BatchCoordinator.retry_item(batch_id: str, item_id: str, capture_logs: bool = True) -> BatchRecord`

- [ ] **Step 1: Write failing creation and isolation tests**

```python
def test_create_and_submit_builds_one_independent_task_per_keyword(tmp_path):
    submitted = []
    coordinator = BatchCoordinator(
        store=BatchStore(tmp_path / "batches"),
        tasks_root=tmp_path / "tasks",
        state=MemoryState(),
        submitter=lambda entries, capture_logs: submitted.extend(entries),
        process_owner="owner-a",
    )
    base = VideoParams(video_subject="", video_script_prompt="语气温和")

    record = coordinator.create_and_submit("自律\n复利", base, capture_logs=False)

    assert [params.video_subject for _, params in submitted] == ["自律", "复利"]
    assert len({task_id for task_id, _ in submitted}) == 2
    assert all("700 至 900" in params.video_script_prompt for _, params in submitted)
    assert base.video_subject == ""
    assert base.video_script_prompt == "语气温和"
    assert len(record.items) == 2
```

- [ ] **Step 2: Run coordinator tests and verify missing interface failures**

Run: `uv run pytest test/services/test_batch_video.py -k "create_and_submit or coordinator" -v`

Expected: fails because `BatchCoordinator` is not defined.

- [ ] **Step 3: Implement transactional batch creation**

Create task IDs and attempts in memory, deep-copy `base_params` per item, set the keyword and composed script prompt, persist the complete batch record, then call `submit_generations()` once. If atomic admission raises, delete the newly created batch record and runtime task states before re-raising. Do not catch runtime failures from worker threads; the existing queue isolates those per task.

For an uploaded custom narration, copy its validated bytes into each `storage/tasks/<task-id>/custom-audio.<ext>` before admission and set that task copy on its parameters. Roll back only these newly created empty task directories if admission fails. Existing local materials remain references to allowlisted `storage/local_videos` files.

- [ ] **Step 4: Write failing recovery and retry tests**

```python
def test_retry_appends_attempt_and_preserves_old_task(tmp_path):
    coordinator, record, submitted = coordinator_with_failed_item(tmp_path)
    old_task_id = record.items[0].attempts[-1].task_id

    updated = coordinator.retry_item(record.batch_id, record.items[0].item_id)

    attempts = updated.items[0].attempts
    assert len(attempts) == 2
    assert attempts[0].task_id == old_task_id
    assert attempts[1].task_id != old_task_id
    assert submitted[-1][1].video_subject == record.items[0].keyword


def test_list_batch_views_marks_old_owner_as_interrupted(tmp_path):
    coordinator, record = persisted_processing_batch(tmp_path, owner="old")
    coordinator.process_owner = "new"
    batches, warnings = coordinator.list_batch_views()
    assert warnings == []
    assert batches[0].items[0].status == "interrupted"
```

- [ ] **Step 5: Implement recovery and retry**

Load records newest first, retrieve runtime state through the existing state abstraction, and call `derive_item_view()` for every item. Retry only failed or interrupted items. Reconstruct `VideoParams` from the stored credential-free snapshot, restore the keyword and knowledge prompt, append a new attempt with the current process owner, save before admission, and revert the appended attempt if atomic admission fails.

```python
class BatchView(BaseModel):
    batch_id: str
    created_at: datetime
    updated_at: datetime
    items: list[BatchItemView]


class BatchUploadRequiredError(ValueError):
    pass
```

If the previous attempt used custom narration, copy only from a file that resolves under its task directory. Raise a typed `BatchUploadRequiredError` when no safe copy exists so the page can request another upload instead of falling back to TTS.

- [ ] **Step 6: Run all batch service tests**

Run: `uv run pytest test/services/test_batch_store.py test/services/test_batch_video.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add app/services/batch_video.py test/services/test_batch_video.py
git commit -m "feat(batch): coordinate persistent video batches"
```

---

### Task 5: Streamlit batch page and shared generation settings

**Files:**
- Create: `webui/batch_page.py`
- Modify: `webui/Main.py`
- Create: `test/services/test_webui_batch_page.py`
- Modify: `test/services/test_webui_generation_defaults.py`
- Modify: `test/services/test_webui_headless_task_actions.py`

**Interfaces:**
- Consumes: `BatchCoordinator`, `BatchView`, and `BatchItemView` from Task 4
- Produces: `batch_page.render_keyword_input(tr: Callable[[str], str]) -> str`
- Produces: `batch_page.render_batch_table(batch: BatchView, tr: Callable[[str], str], on_retry: Callable[[str, str], None]) -> None`
- Produces in `Main.py`: `RenderedGenerationInputs(uploaded_files: list[Any], uploaded_audio_file: Any | None, uploaded_bgm_file: Any | None, voice_mode: str)`
- Produces in `Main.py`: `_render_generation_settings(params: VideoParams, key_prefix: str = "") -> RenderedGenerationInputs`
- Produces in `Main.py`: `_validate_generation_inputs(params: VideoParams, rendered_inputs: RenderedGenerationInputs, session_snapshot: Mapping[str, Any]) -> list[str]`
- Produces in `Main.py`: `_render_batch_application() -> None`
- Preserves in `Main.py`: `_render_application() -> None` as the single-video route

- [ ] **Step 1: Write failing static page-boundary tests**

```python
def test_main_routes_between_single_and_batch_pages():
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    calls = {_attribute_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert "_render_batch_application" in calls
    assert "_render_application" in calls


def test_batch_module_does_not_call_media_pipeline_directly():
    tree = ast.parse(WEBUI_BATCH_PAGE.read_text(encoding="utf-8"))
    calls = {_attribute_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert "tm.start" not in calls
    assert "task.start" not in calls
```

- [ ] **Step 2: Run the new WebUI tests and verify the missing page failure**

Run: `uv run pytest test/services/test_webui_batch_page.py -v`

Expected: fails because `webui/batch_page.py` and the batch route do not exist.

- [ ] **Step 3: Add conditional page navigation and reuse generation controls**

Add a top-level segmented control backed by `st.session_state["generation_page"]` with values `single` and `batch`. Only render the selected page, because Streamlit tabs render both bodies and would duplicate expensive controls and widget keys.

Extract the existing four settings panels into a shared `_render_generation_settings(params, key_prefix)` helper. Preserve existing single-page widget keys when `key_prefix=""`; use `batch_` keys on the batch route. Keep uploads in the current request and return them in a typed `RenderedGenerationInputs` dataclass.

```python
selected_page = st.segmented_control(
    tr("Generation Mode"),
    options=["single", "batch"],
    format_func=lambda value: tr("Single Generation") if value == "single" else tr("Batch Generation"),
    key="generation_page",
)
if selected_page == "batch":
    _render_batch_application()
else:
    _render_application()
```

- [ ] **Step 4: Extract a shared no-side-effect preflight**

Move the current validity checks out of the button branch into `_validate_generation_inputs(params, rendered_inputs, session_snapshot) -> list[str]`. It must return translation keys or safe messages and must not call `st.stop()`, write files, enqueue tasks, or call paid services. Both single and batch submit paths render every returned error and abort before materializing uploads.

Retain charge quote construction after the pure checks. A batch using WaveSpeed, Volcano Seedance, LoomLoom, Sonilo, or ElevenLabs must reuse the same explicit confirmation rules as a single task.

- [ ] **Step 5: Render keyword submission and persistent task table**

The batch page uses a multiline input, live valid-line count, one primary submit button, batch selection, and status filter. On submit, call `parse_keywords()` for feedback, save runtime configuration once, materialize shared uploads, and call `BatchCoordinator.create_and_submit()`.

Use a Fragment polling at one second for the table. Render columns for position, keyword, translated status, progress, script summary/length, update time, and actions. For complete items, use `st.video()` plus `st.download_button()` with `_build_video_download_name()` and an allowlisted open file. For failed/interrupted items, call the injected retry handler.

```python
@st.fragment(run_every=1.0)
def render_batch_status(coordinator, selected_batch_id, status_filter, tr):
    batches, warnings = coordinator.list_batch_views()
    for warning in warnings:
        st.warning(tr("Batch Record Load Warning").format(filename=warning.filename))
    batch = next((value for value in batches if value.batch_id == selected_batch_id), None)
    if batch is not None:
        batch_page.render_batch_table(batch, tr, _retry_batch_item)
```

- [ ] **Step 6: Add interactive rendering tests with fake Streamlit**

Cover effective keyword count, status filtering, script summary and character count, duplicate row keys, failed retry, safe completed video preview, and generated download filename. Add an AST regression test proving the single page still submits via `webui_task.submit_generation()` and the batch page only submits through `BatchCoordinator`.

- [ ] **Step 7: Run WebUI tests**

Run: `uv run pytest test/services/test_webui_batch_page.py test/services/test_webui_generation_defaults.py test/services/test_webui_headless_task_actions.py test/services/test_webui_task.py -v`

Expected: all tests pass.

- [ ] **Step 8: Commit**

```powershell
git add webui/batch_page.py webui/Main.py test/services/test_webui_batch_page.py test/services/test_webui_generation_defaults.py test/services/test_webui_headless_task_actions.py
git commit -m "feat(webui): add persistent batch video page"
```

---

### Task 6: Localization, documentation, and complete verification

**Files:**
- Modify: `webui/i18n/zh.json`
- Modify: `webui/i18n/en.json`
- Modify: `test/services/test_webui_i18n.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: all visible translation keys from Task 5
- Produces: Chinese and English labels with existing English fallback for secondary locales
- Produces: user documentation for batch keyword video generation

- [ ] **Step 1: Extend i18n coverage to the new module and verify failure**

Update `_TrKeyVisitor` coverage to parse both `webui/Main.py` and `webui/batch_page.py`. Add a test asserting Chinese and English contain the batch navigation, input, status, retry, interrupted, warning, and download keys used by the page.

Run: `uv run pytest test/services/test_webui_i18n.py -v`

Expected: fails with the missing batch translation keys.

- [ ] **Step 2: Add Chinese and English translations**

Add exact user-facing translations to `zh.json` and `en.json`. Keep secondary locale JSON files unchanged so runtime falls back to English through the existing `tr()` behavior. Update `ENGLISH_FALLBACK_KEYS` for the new batch-only keys so secondary-locale completeness tests continue to enforce intentional fallback rather than accidental omission.

- [ ] **Step 3: Document the workflow**

Add a “批量关键词生成视频” section after WebUI startup in `README.md` describing one-keyword-per-line input, the automatic approximately 800-character origin/meaning/inspiration script, the 100-item limit, sequential execution, persistent history, retry behavior, and where to download completed videos.

- [ ] **Step 4: Run focused batch and WebUI tests**

Run: `uv run pytest test/services/test_batch_store.py test/services/test_batch_video.py test/services/test_task_manager.py test/services/test_webui_task.py test/services/test_webui_batch_page.py test/services/test_webui_i18n.py -v`

Expected: all tests pass.

- [ ] **Step 5: Run lint and the full test suite**

Run: `uv run ruff check app webui test cli.py main.py`

Expected: exit code 0 with no lint errors.

Run: `uv run pytest -q`

Expected: exit code 0 with no failed tests.

- [ ] **Step 6: Start the WebUI and perform an HTTP smoke test**

Run the project `webui.bat` in a hidden background process, read the selected port from its log, then request the root page.

Expected: Streamlit listens on `127.0.0.1`, the request returns HTTP 200, and the HTML contains the Streamlit application shell. Manually confirm the new navigation renders “批量生成”, accepts at least two lines, and shows both task rows after a mocked or configured submission without requiring a page refresh.

- [ ] **Step 7: Commit**

```powershell
git add webui/i18n/zh.json webui/i18n/en.json test/services/test_webui_i18n.py README.md
git commit -m "docs(batch): localize and document batch generation"
```

---

## Final acceptance checklist

- [ ] A three-keyword batch creates three ordered, unique task IDs and runs them sequentially.
- [ ] Each task receives the knowledge script prompt and automatically continues through video generation.
- [ ] Duplicate input lines remain independent tasks.
- [ ] A failed middle task does not prevent the next queued task from running.
- [ ] Refreshing the browser retains batch membership and live task state.
- [ ] Restarting the app retains completed downloads and marks unfinished old-owner attempts interrupted.
- [ ] Retrying appends a new attempt and preserves the old task directory.
- [ ] Completed videos preview and download only from the task allowlist.
- [ ] Queue-capacity and validation failures reject the entire batch before external calls.
- [ ] No batch record contains API keys, tokens, or uploaded bytes.
- [ ] The existing single-generation WebUI path, CLI, API, and task history tests remain green.
