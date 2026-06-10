"""
In-process integration tests for gpal MCP server.

Uses FastMCP's Client class for zero-network testing of tool registration,
metadata, path validation, and annotation correctness. Live API tests are
gated behind GEMINI_API_KEY.
"""

import inspect
import os
import time
import pytest
from pathlib import Path

from fastmcp import Client
from unittest.mock import patch, AsyncMock, MagicMock
from cachetools import TTLCache

from gpal.server import (
    mcp, _validate_input_path, _validate_output_path,
    record_tokens, tokens_in_window, token_stats, GeminiResponse,
    _sync_throttle, _async_throttle, _throttle_delay, _KNOWN_MODELS,
    MODEL_SEARCH, MODEL_LITE, RATE_LIMITS_TPM, _afc_local,
    _send_with_retry, _EXECUTOR,
    _extract_retry_delay, _is_retriable_genai_error,
    search_project, _sanitize_history,
    sessions, sessions_lock,
    _get_or_create_session_entry, _ensure_session_model,
    MODEL_ALIASES, get_client, _stdin_disconnected,
    _token_windows, _token_lock,
)


# ─────────────────────────────────────────────────────────────────────────────
# A. Tool Registration & Metadata
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_TOOLS = {
    "consult_gemini",
    "consult_gemini_oneshot",
    "gemini_search",
    "gemini_code_exec",
    "upload_file",
    "create_context_cache",
    "delete_context_cache",
    "create_file_store",
    "list_file_stores",
    "delete_file_store",
    "upload_to_file_store",
    "list_models",
    "generate_image",
    "generate_speech",
    "create_batch",
    "get_batch",
    "list_batches",
    "get_batch_results",
    "cancel_batch",
    "delete_batch",
}


@pytest.mark.asyncio
async def test_all_tools_registered():
    """Every expected tool is exposed via the MCP server."""
    async with Client(mcp) as c:
        tools = await c.list_tools()
        names = {t.name for t in tools}
        missing = EXPECTED_TOOLS - names
        assert not missing, f"Missing tools: {missing}"


@pytest.mark.asyncio
async def test_tool_descriptions_non_empty():
    """Every tool has a non-empty description."""
    async with Client(mcp) as c:
        tools = await c.list_tools()
        for tool in tools:
            assert tool.description, f"Tool '{tool.name}' has no description"


@pytest.mark.asyncio
async def test_tool_input_schemas():
    """Key tools have expected parameters in their input schemas."""
    async with Client(mcp) as c:
        tools = await c.list_tools()
        by_name = {t.name: t for t in tools}

        # consult_gemini should have 'query' required and 'model', 'thinking' optional
        schema = by_name["consult_gemini"].inputSchema
        assert "query" in schema.get("properties", {}), "consult_gemini missing 'query' param"
        assert "query" in schema.get("required", []), "consult_gemini: 'query' not required"
        assert "model" in schema.get("properties", {}), "consult_gemini missing 'model' param"
        assert "thinking" in schema.get("properties", {}), "consult_gemini missing 'thinking' param"

        # consult_gemini_oneshot should have 'query' required
        schema = by_name["consult_gemini_oneshot"].inputSchema
        assert "query" in schema.get("properties", {})
        assert "query" in schema.get("required", [])

        # generate_image should have prompt + output_path
        schema = by_name["generate_image"].inputSchema
        assert "prompt" in schema.get("properties", {})
        assert "output_path" in schema.get("properties", {})


# ─────────────────────────────────────────────────────────────────────────────
# B. Tool Timeouts
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_TIMEOUTS = {
    "consult_gemini": 660,
    "consult_gemini_oneshot": 600,
    "gemini_search": 120,
    "gemini_code_exec": 120,
    "generate_image": 300,
    "generate_speech": 180,
    "upload_file": 120,
    "upload_to_file_store": 120,
    "create_context_cache": 60,
    "create_batch": 120,
    "get_batch": 30,
    "list_batches": 30,
    "get_batch_results": 60,
    "cancel_batch": 30,
    "delete_batch": 30,
}


@pytest.mark.asyncio
async def test_tool_timeouts():
    """Verify timeout values on tools that have them."""
    all_tools = await mcp._local_provider.list_tools()
    by_name = {t.name: t for t in all_tools}

    for tool_name, expected in EXPECTED_TIMEOUTS.items():
        tool = by_name[tool_name]
        assert tool.timeout == expected, (
            f"{tool_name}: expected timeout={expected}, got {tool.timeout}"
        )


@pytest.mark.asyncio
async def test_tools_without_timeout():
    """Tools not in the timeout table should have no timeout set."""
    all_tools = await mcp._local_provider.list_tools()
    for tool in all_tools:
        if tool.name not in EXPECTED_TIMEOUTS:
            assert tool.timeout is None, (
                f"{tool.name}: unexpected timeout={tool.timeout}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# C. Path Validation
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_output_path_within_cwd(tmp_path, monkeypatch):
    """Paths under cwd are accepted."""
    monkeypatch.chdir(tmp_path)
    result = _validate_output_path(str(tmp_path / "output" / "file.png"))
    assert result is None
    assert (tmp_path / "output").is_dir()


def test_validate_output_path_rejects_traversal(tmp_path, monkeypatch):
    """Path traversal outside cwd is rejected."""
    monkeypatch.chdir(tmp_path)
    result = _validate_output_path("/tmp/evil.png")
    assert result is not None
    assert "outside" in result.lower() or "denied" in result.lower()


def test_validate_output_path_rejects_etc(tmp_path, monkeypatch):
    """Absolute paths to system dirs are rejected."""
    monkeypatch.chdir(tmp_path)
    result = _validate_output_path("/etc/passwd")
    assert result is not None


def test_validate_output_path_relative_traversal(tmp_path, monkeypatch):
    """../../../etc/passwd style traversal is caught."""
    monkeypatch.chdir(tmp_path)
    result = _validate_output_path("../../../etc/passwd")
    assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# D. Annotation Regression (guards against `from __future__ import annotations`)
# ─────────────────────────────────────────────────────────────────────────────


def test_no_future_annotations_import():
    """server.py must not use `from __future__ import annotations`.

    PEP 563 deferred evaluation breaks Gemini's automatic function calling
    because type annotations become strings instead of real types.
    """
    import gpal.server as mod
    source = inspect.getsource(mod)
    assert "from __future__ import annotations" not in source


def test_tool_functions_have_real_annotations():
    """Gemini AFC tool functions must have concrete type annotations, not strings."""
    from gpal.server import list_directory, read_file, search_project

    for fn in (list_directory, read_file, search_project):
        hints = fn.__annotations__
        for param, annotation in hints.items():
            assert not isinstance(annotation, str), (
                f"{fn.__name__}.{param} has string annotation '{annotation}' — "
                "probably caused by `from __future__ import annotations`"
            )


# ─────────────────────────────────────────────────────────────────────────────
# E. Token Tracking
# ─────────────────────────────────────────────────────────────────────────────


def test_record_and_query_tokens():
    """Token tracking records and queries correctly."""
    from gpal.server import MODEL_FLASH
    record_tokens(MODEL_FLASH, 100)
    record_tokens(MODEL_FLASH, 200)
    assert tokens_in_window(MODEL_FLASH) >= 300


def test_tokens_in_window_expires():
    """Tokens older than the window are not counted."""
    from gpal.server import MODEL_PRO
    import gpal.server as srv
    with srv._token_lock:
        srv._token_windows[MODEL_PRO] = [(time.monotonic() - 120, 9999)]
    # Should be zero — 120s ago is outside the 60s window
    assert tokens_in_window(MODEL_PRO) == 0


def test_token_stats_returns_active():
    """token_stats() returns models with recent usage."""
    from gpal.server import MODEL_FLASH
    record_tokens(MODEL_FLASH, 500)
    stats = token_stats()
    assert MODEL_FLASH in stats
    assert stats[MODEL_FLASH]["tokens_last_60s"] >= 500


def test_record_tokens_rejects_unknown_model():
    """Unknown model strings are silently ignored (DoS prevention)."""
    record_tokens("totally-fake-model", 9999)
    assert tokens_in_window("totally-fake-model") == 0


def test_gemini_response_dataclass():
    """GeminiResponse holds text and token counts."""
    r = GeminiResponse("hello", 10, 20, 30)
    assert r.text == "hello"
    assert r.prompt_tokens == 10
    assert r.completion_tokens == 20
    assert r.total_tokens == 30


# ─────────────────────────────────────────────────────────────────────────────
# F. ToolResult Structure (requires API key)
# ─────────────────────────────────────────────────────────────────────────────

HAS_API_KEY = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_API_KEY, reason="no GEMINI_API_KEY")
async def test_consult_returns_tool_result():
    """consult_gemini returns structured ToolResult with meta including tokens."""
    async with Client(mcp) as c:
        result = await c.call_tool(
            "consult_gemini",
            {"query": "What is 2+2? Reply with just the number.", "model": "flash"},
        )
        assert not result.is_error
        assert result.structured_content is not None
        assert "result" in result.structured_content
        assert "model" in result.structured_content
        assert result.meta is not None
        assert "model" in result.meta
        assert "session_id" in result.meta
        assert "duration_ms" in result.meta
        assert result.meta["duration_ms"] > 0
        # Token counts should be present
        assert "total_tokens" in result.meta


# ─────────────────────────────────────────────────────────────────────────────
# G. Gemini Autonomous Tools (requires API key)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_API_KEY, reason="no GEMINI_API_KEY")
async def test_flash_can_read_file():
    """Flash can use read_file to read a known file autonomously."""
    async with Client(mcp) as c:
        result = await c.call_tool(
            "consult_gemini",
            {
                "query": "Use read_file to read pyproject.toml and tell me the project name.",
                "model": "flash",
            },
        )
        assert not result.is_error
        text = str(result.content)
        assert "gpal" in text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# H. Resources
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_info_resource():
    """gpal://info returns valid JSON with models, limits, and token_usage."""
    async with Client(mcp) as c:
        resources = await c.list_resources()
        uris = {str(r.uri) for r in resources}
        assert "gpal://info" in uris

        import json
        result = await c.read_resource("gpal://info")
        text = result[0].text if isinstance(result, list) else str(result)
        data = json.loads(text)
        assert "models" in data
        assert "limits" in data
        assert "lite" in data["models"]
        assert "flash" in data["models"]
        assert "token_usage" in data


# ─────────────────────────────────────────────────────────────────────────────
# I. Throttle Helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_model_search_in_known_models():
    """MODEL_SEARCH is tracked (present in _KNOWN_MODELS and RATE_LIMITS_TPM)."""
    assert MODEL_SEARCH in _KNOWN_MODELS
    assert MODEL_SEARCH in RATE_LIMITS_TPM


def test_model_lite_in_known_models():
    """MODEL_LITE is tracked (present in _KNOWN_MODELS and RATE_LIMITS_TPM)."""
    assert MODEL_LITE in _KNOWN_MODELS
    assert MODEL_LITE in RATE_LIMITS_TPM


def test_throttle_delay_under_limit():
    """_throttle_delay returns 0 when usage is under 90%."""
    from gpal.server import MODEL_FLASH
    assert _throttle_delay(MODEL_FLASH) == 0.0


def test_throttle_delay_over_limit():
    """_throttle_delay calculates exact sleep time from token window."""
    from gpal.server import MODEL_FLASH, _token_windows, _token_lock
    limit = RATE_LIMITS_TPM[MODEL_FLASH]
    now = time.monotonic()
    # Record tokens that will expire at now + 60 - make it over 90%
    with _token_lock:
        _token_windows[MODEL_FLASH] = [(now, limit)]  # 100% usage
    try:
        delay = _throttle_delay(MODEL_FLASH)
        # Should be ~60s (tokens recorded at 'now', expire at now+60)
        assert 55.0 < delay <= 61.0
    finally:
        with _token_lock:
            _token_windows[MODEL_FLASH] = []


def test_throttle_delay_unknown_model():
    """_throttle_delay returns 0 for models not in RATE_LIMITS_TPM."""
    assert _throttle_delay("some-unknown-model") == 0.0


def test_sync_throttle_sleeps_exact():
    """_sync_throttle sleeps for the delay from _throttle_delay."""
    from gpal.server import MODEL_FLASH
    with patch("gpal.server._throttle_delay", return_value=5.0):
        with patch("gpal.server.time.sleep") as mock_sleep:
            _sync_throttle(MODEL_FLASH)
            mock_sleep.assert_called_once_with(5.0)


def test_sync_throttle_no_sleep_when_clear():
    """_sync_throttle doesn't sleep when _throttle_delay returns 0."""
    from gpal.server import MODEL_FLASH
    with patch("gpal.server._throttle_delay", return_value=0.0):
        with patch("gpal.server.time.sleep") as mock_sleep:
            _sync_throttle(MODEL_FLASH)
            mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_async_throttle_sleeps_exact():
    """_async_throttle sleeps for the delay from _throttle_delay."""
    from gpal.server import MODEL_FLASH
    with patch("gpal.server._throttle_delay", return_value=5.0):
        with patch("gpal.server.asyncio.sleep") as mock_sleep:
            await _async_throttle(MODEL_FLASH)
            mock_sleep.assert_called_once_with(5.0)


@pytest.mark.asyncio
async def test_async_throttle_no_sleep_when_clear():
    """_async_throttle doesn't sleep when _throttle_delay returns 0."""
    from gpal.server import MODEL_FLASH
    with patch("gpal.server._throttle_delay", return_value=0.0):
        with patch("gpal.server.asyncio.sleep") as mock_sleep:
            await _async_throttle(MODEL_FLASH)
            mock_sleep.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# J. History Sanitization
# ─────────────────────────────────────────────────────────────────────────────


class _FakeContent:
    """Minimal stand-in for a Gemini history entry."""
    def __init__(self, role, parts=None):
        self.role = role
        self.parts = parts or []


class _FakePart:
    """Minimal stand-in for a Gemini content part."""
    def __init__(self, function_call=None):
        self.function_call = function_call


def test_sanitize_history_clean():
    """No-op when history is already valid (ends with model, no orphans)."""
    history = [_FakeContent("user"), _FakeContent("model")]
    assert _sanitize_history(history) is False
    assert len(history) == 2


def test_sanitize_history_trailing_user():
    """Strips trailing user turn."""
    history = [_FakeContent("user"), _FakeContent("model"), _FakeContent("user")]
    assert _sanitize_history(history) is True
    assert len(history) == 2
    assert history[-1].role == "model"


def test_sanitize_history_orphaned_function_call():
    """Strips model response with function_call that has no function_response."""
    fc_part = _FakePart(function_call={"name": "read_file", "args": {}})
    history = [
        _FakeContent("user"),
        _FakeContent("model"),
        _FakeContent("user"),
        _FakeContent("model", parts=[fc_part]),  # orphaned function_call
    ]
    assert _sanitize_history(history) is True
    # Should drop the orphaned model turn and its preceding user turn
    assert len(history) == 2
    assert history[-1].role == "model"


def test_sanitize_history_empty():
    """No-op on empty history."""
    history = []
    assert _sanitize_history(history) is False
    assert len(history) == 0


def test_sanitize_history_user_then_orphaned_fc():
    """Handles trailing user + orphaned function_call in sequence."""
    fc_part = _FakePart(function_call={"name": "search", "args": {}})
    history = [
        _FakeContent("user"),
        _FakeContent("model"),
        _FakeContent("user"),
        _FakeContent("model", parts=[fc_part]),
        _FakeContent("user"),  # trailing user turn
    ]
    assert _sanitize_history(history) is True
    # Strips trailing user, then finds orphaned fc, strips that + its user
    assert len(history) == 2


# ─────────────────────────────────────────────────────────────────────────────
# K. AFC Context Flag
# ─────────────────────────────────────────────────────────────────────────────


def test_afc_flag_set_during_send(tmp_path):
    """_afc_local.in_afc is True during send_message and False after."""
    observed = []

    class FakeSession:
        def send_message(self, parts, config=None):
            observed.append(getattr(_afc_local, "in_afc", False))

            class FakeResponse:
                text = "ok"
                candidates = []
                usage_metadata = None
            return FakeResponse()

    from google.genai import types
    result = _send_with_retry(FakeSession(), [], types.GenerateContentConfig())
    assert observed == [True], f"in_afc was {observed} during send_message"
    assert getattr(_afc_local, "in_afc", False) is False


def test_afc_flag_cleared_on_exception():
    """_afc_local.in_afc is cleared even when send_message raises."""
    class FakeSession:
        def send_message(self, parts, config=None):
            raise RuntimeError("boom")

    from google.genai import types
    try:
        _send_with_retry(FakeSession(), [], types.GenerateContentConfig())
    except RuntimeError:
        pass
    assert getattr(_afc_local, "in_afc", False) is False


def test_executor_has_gpal_prefix():
    """_EXECUTOR threads use 'gpal' prefix."""
    assert _EXECUTOR._thread_name_prefix == "gpal"
    assert _EXECUTOR._max_workers == 16


# ─────────────────────────────────────────────────────────────────────────────
# I. Retry Delay Extraction
# ─────────────────────────────────────────────────────────────────────────────


def test_extract_retry_delay_from_retry_info():
    """Extracts retryDelay from google.rpc.RetryInfo in error details."""
    from google.genai.errors import ClientError
    exc = ClientError(429, {
        "error": {
            "code": 429,
            "message": "Rate limited",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "18.5s"},
            ],
        }
    })
    delay = _extract_retry_delay(exc)
    assert delay == pytest.approx(18.5)


def test_extract_retry_delay_integer_seconds():
    """Handles integer-style retryDelay like '18s'."""
    from google.genai.errors import ClientError
    exc = ClientError(429, {
        "error": {
            "code": 429,
            "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "18s"},
            ],
        }
    })
    assert _extract_retry_delay(exc) == 18.0


def test_extract_retry_delay_no_retry_info():
    """Returns None when no RetryInfo is present."""
    from google.genai.errors import ClientError
    exc = ClientError(429, {"error": {"code": 429, "details": []}})
    assert _extract_retry_delay(exc) is None


def test_extract_retry_delay_not_api_error():
    """Returns None for non-APIError exceptions."""
    assert _extract_retry_delay(RuntimeError("boom")) is None


def test_extract_retry_delay_string_error():
    """Handles error responses where 'error' is a string, not a dict."""
    from google.genai.errors import ClientError
    exc = ClientError(429, {"error": "Model overloaded"})
    assert _extract_retry_delay(exc) is None


def test_is_retriable_429():
    """429 errors are retriable."""
    from google.genai.errors import ClientError
    exc = ClientError(429, {"error": {"code": 429}})
    assert _is_retriable_genai_error(exc) is True


def test_is_not_retriable_5xx():
    """5xx errors are not retriable — surface immediately."""
    from google.genai.errors import ServerError
    for code in (500, 502, 503, 504):
        exc = ServerError(code, {"error": {"code": code}})
        assert _is_retriable_genai_error(exc) is False, f"{code} should not be retriable"


def test_is_not_retriable_400():
    """400 Bad Request is not retriable."""
    from google.genai.errors import ClientError
    exc = ClientError(400, {"error": {"code": 400}})
    assert _is_retriable_genai_error(exc) is False


def test_is_not_retriable_non_api_error():
    """Non-APIError exceptions are not retriable."""
    assert _is_retriable_genai_error(RuntimeError("boom")) is False


# ─────────────────────────────────────────────────────────────────────────────
# K. Input Path Validation
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_input_path_within_cwd(tmp_path, monkeypatch):
    """Paths under cwd are accepted."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.txt").write_text("hello")
    result = _validate_input_path(str(tmp_path / "data.txt"))
    assert result is None


def test_validate_input_path_rejects_traversal(tmp_path, monkeypatch):
    """Path traversal outside cwd is rejected."""
    monkeypatch.chdir(tmp_path)
    result = _validate_input_path("../../../etc/passwd")
    assert result is not None
    assert "outside" in result.lower() or "denied" in result.lower()


def test_validate_input_path_rejects_absolute(tmp_path, monkeypatch):
    """Absolute paths to system dirs are rejected."""
    monkeypatch.chdir(tmp_path)
    result = _validate_input_path("/etc/passwd")
    assert result is not None
    assert "denied" in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
# L. Search Project Glob Validation
# ─────────────────────────────────────────────────────────────────────────────


def test_search_project_rejects_absolute_glob(tmp_path, monkeypatch):
    """Absolute glob patterns are rejected early."""
    monkeypatch.chdir(tmp_path)
    result = search_project("test", glob_pattern="/etc/**/*")
    assert "absolute" in result.lower()


def test_search_project_rejects_traversal_glob(tmp_path, monkeypatch):
    """Glob patterns with '..' are rejected to prevent filesystem traversal DoS."""
    monkeypatch.chdir(tmp_path)
    result = search_project("test", glob_pattern="../../**/*")
    assert ".." in result


# ─────────────────────────────────────────────────────────────────────────────
# M. consult_gemini Pipeline Contracts
# ─────────────────────────────────────────────────────────────────────────────


def _make_ctx(session_id: str = "test-session") -> MagicMock:
    """Minimal fake Context sufficient for consult_gemini."""
    ctx = MagicMock()
    ctx.session_id = session_id
    ctx.debug = AsyncMock()
    ctx.info = AsyncMock()
    return ctx


async def _get_consult_gemini_fn():
    """Return the underlying async function from the FunctionTool."""
    # FastMCP 3.x wraps the function in a FunctionTool; .fn holds the original.
    tool = await mcp._local_provider._get_tool("consult_gemini")
    return tool.fn


@pytest.mark.asyncio
async def test_phase1_non_error_prefix_aborts_phase2():
    """A plain-str return from _consult that does NOT start with 'Error:' still
    aborts phase 2 — the sentinel is isinstance(result, str), not startswith."""
    error_msg = "Error reading file 'x': boom"  # no "Error:" prefix on the whole string
    consult_fn = await _get_consult_gemini_fn()
    ctx = _make_ctx()

    with patch("gpal.server._consult", new_callable=AsyncMock, return_value=error_msg) as mock_consult:
        result = await consult_fn(query="what?", model="flash", ctx=ctx)

    assert result == error_msg, f"Expected error string to be returned, got: {result!r}"
    assert mock_consult.call_count == 1, (
        f"Phase 2 must not run when phase 1 returns a plain str; got {mock_consult.call_count} calls"
    )


@pytest.mark.asyncio
async def test_cached_content_not_forwarded_to_explore_phase():
    """cached_content is passed as None to phase 1 (caches are model-bound)
    and forwarded to phase 2."""
    from fastmcp.tools.tool import ToolResult

    phase2_result = ToolResult(content="synthesis answer", structured_content={}, meta={})
    consult_fn = await _get_consult_gemini_fn()
    ctx = _make_ctx()

    call_args_list = []

    async def fake_consult(*args, **kwargs):
        call_args_list.append((args, kwargs))
        # Phase 1 (explore) returns a ToolResult so phase 2 proceeds
        if len(call_args_list) == 1:
            return ToolResult(content="explored", structured_content={}, meta={})
        return phase2_result

    with patch("gpal.server._consult", side_effect=fake_consult):
        result = await consult_fn(
            query="analyze this", model="pro", cached_content="caches/x", ctx=ctx
        )

    assert len(call_args_list) == 2, "Expected exactly 2 _consult calls (explore + synthesize)"

    # Phase 1 positional args: (query, ctx, model_alias, file_paths, media_paths,
    #                            file_uris, json_mode, response_schema, cached_content, ...)
    phase1_args, phase1_kwargs = call_args_list[0]
    phase1_cached = phase1_kwargs.get("cached_content") if "cached_content" in phase1_kwargs else phase1_args[8]
    assert phase1_cached is None, f"Phase 1 must receive cached_content=None, got {phase1_cached!r}"

    phase2_args, phase2_kwargs = call_args_list[1]
    phase2_cached = phase2_kwargs.get("cached_content") if "cached_content" in phase2_kwargs else phase2_args[8]
    assert phase2_cached == "caches/x", f"Phase 2 must receive cached_content='caches/x', got {phase2_cached!r}"


@pytest.mark.asyncio
async def test_invalid_thinking_rejected_before_consult():
    """An invalid thinking string is caught before _consult is called."""
    consult_fn = await _get_consult_gemini_fn()
    ctx = _make_ctx()

    with patch("gpal.server._consult", new_callable=AsyncMock) as mock_consult:
        result = await consult_fn(query="anything", thinking="bogus", ctx=ctx)

    assert isinstance(result, str)
    assert "bogus" in result
    assert mock_consult.call_count == 0, "No _consult call expected for invalid thinking"


@pytest.mark.asyncio
async def test_lite_with_thinking_rejected_before_consult():
    """model='lite' with an explicit thinking value returns an error without calling _consult."""
    consult_fn = await _get_consult_gemini_fn()
    ctx = _make_ctx()

    with patch("gpal.server._consult", new_callable=AsyncMock) as mock_consult:
        result = await consult_fn(query="anything", model="lite", thinking="high", ctx=ctx)

    assert isinstance(result, str)
    assert "lite" in result.lower()
    assert mock_consult.call_count == 0, "No _consult call expected when lite+thinking is rejected"


@pytest.mark.asyncio
async def test_ctx_none_returns_error_without_consult():
    """consult_gemini returns an error string without calling _consult when ctx is None."""
    consult_fn = await _get_consult_gemini_fn()

    with patch("gpal.server._consult", new_callable=AsyncMock) as mock_consult:
        result = await consult_fn(query="anything", ctx=None)

    assert isinstance(result, str)
    assert "error" in result.lower()
    assert mock_consult.call_count == 0, "No _consult call expected when ctx is None"


# ─────────────────────────────────────────────────────────────────────────────
# N. Session Layer: TTL Refresh, Migration-Under-Lock, Send-to-Current-Session
# ─────────────────────────────────────────────────────────────────────────────


def _fake_session(model: str) -> MagicMock:
    """Return a MagicMock session with _gpal_model set and empty history."""
    s = MagicMock()
    s._gpal_model = model
    s._curated_history = []
    return s


def _fake_client() -> MagicMock:
    """Return a MagicMock genai.Client whose chats.create returns a fresh fake session."""
    client = MagicMock()
    client.chats.create.side_effect = lambda model, history=None, config=None: _fake_session(model)
    return client


def test_ttl_refresh_on_lookup():
    """_get_or_create_session_entry re-sets the TTLCache entry on every successful lookup,
    preventing long-running sessions from silently expiring.

    We use cachetools' timer= hook to control time without sleeping.
    """
    tick = [0.0]  # mutable cell so the lambda can mutate it

    fake_timer = lambda: tick[0]  # noqa: E731
    local_cache = TTLCache(maxsize=10, ttl=5, timer=fake_timer)

    session_id = "ttl-test-session"
    client = _fake_client()
    target_model = "gemini-flash"
    lock_placeholder = MagicMock()

    # Seed the cache at t=0
    fake_sess = _fake_session(target_model)
    with sessions_lock:
        local_cache[session_id] = (fake_sess, lock_placeholder)

    # Patch global sessions with our controlled cache
    with patch("gpal.server.sessions", local_cache):
        # t=0: entry exists, lookup should succeed and refresh expiry
        result = _get_or_create_session_entry(session_id, client, target_model)
        assert result[0] is fake_sess, "Should return existing session"

        # Advance to t=4 (within TTL but close to expiry) and look up again
        tick[0] = 4.0
        result2 = _get_or_create_session_entry(session_id, client, target_model)
        assert result2[0] is fake_sess, "Should still return existing session at t=4"

        # Advance to t=8: without the re-set on each lookup the entry inserted at
        # t=4 would have expired (4+5=9 > 8 is fine, but if we never refreshed we'd
        # be checking the original t=0 insertion which expired at t=5).
        tick[0] = 8.0
        assert session_id in local_cache, (
            "Entry should still be alive at t=8 because each lookup refreshed the TTL"
        )

        # Advance to t=14 (> last re-set at t=8 + ttl=5 = 13) — now it really expires
        tick[0] = 14.0
        assert session_id not in local_cache, "Entry must expire after TTL with no activity"


def test_ensure_session_model_migrates_and_stores():
    """_ensure_session_model recreates the session when the model changes and
    stores the new object back in sessions under sessions_lock."""
    import asyncio

    session_id = "migrate-test"
    old_model = "gemini-old"
    new_model = "gemini-new"

    old_session = _fake_session(old_model)
    lock = asyncio.Lock()
    client = _fake_client()

    with patch("gpal.server.sessions", {session_id: (old_session, lock)}):
        result = _ensure_session_model(session_id, client, new_model, config=None, lock=lock)

    # Result must be a new session object with the target model set
    assert result is not old_session, "Must return a new session object after migration"
    assert result._gpal_model == new_model, "_gpal_model must be updated to the target model"


def test_ensure_session_model_passes_history_through_sanitize():
    """_ensure_session_model feeds history through _sanitize_history before recreating."""
    import asyncio

    session_id = "sanitize-test"
    model = "gemini-flash"

    # Build a session whose history has a trailing user turn (invalid)
    trailing_user = MagicMock()
    trailing_user.role = "user"
    model_turn = MagicMock()
    model_turn.role = "model"
    model_turn.parts = []

    old_session = MagicMock()
    old_session._gpal_model = model
    old_session._curated_history = [model_turn, trailing_user]

    lock = asyncio.Lock()

    captured_histories = []

    # client.chats.create is called with keyword args model=, history=, config=
    def fake_create(model=None, history=None, config=None):
        captured_histories.append(list(history or []))
        s = _fake_session(model)
        return s

    client = MagicMock()
    client.chats.create.side_effect = fake_create

    with patch("gpal.server.sessions", {session_id: (old_session, lock)}):
        # Same model but history needs sanitization → must recreate
        result = _ensure_session_model(session_id, client, model, config=None, lock=lock)

    assert len(captured_histories) == 1, "create_chat should be called exactly once"
    assert trailing_user not in captured_histories[0], (
        "Trailing user turn must be stripped from history passed to create_chat"
    )


def test_ensure_session_model_no_change_returns_same():
    """_ensure_session_model returns the same session when model matches and
    history is clean (no unnecessary recreation)."""
    import asyncio

    session_id = "no-change-test"
    model = "gemini-flash"

    session = _fake_session(model)
    lock = asyncio.Lock()
    client = _fake_client()

    with patch("gpal.server.sessions", {session_id: (session, lock)}):
        result = _ensure_session_model(session_id, client, model, config=None, lock=lock)

    assert result is session, "Should return the same session when nothing needs to change"
    client.chats.create.assert_not_called()


def test_send_uses_current_session_after_external_replacement():
    """_ensure_session_model re-reads sessions at call time, not a stale reference.

    Simulates the race-fix: we seed sessions[sid] with S1 (the stale entry a
    caller held before acquiring the lock), then replace it with S2 to mimic a
    concurrent coroutine that migrated the entry while we were waiting.
    _ensure_session_model must return S2 (fresh read from sessions), not S1.

    This test CAN fail: if _ensure_session_model took a session object argument
    instead of re-reading sessions it would return S1, failing the assertion.
    """
    import asyncio
    from cachetools import TTLCache

    session_id = "race-test"
    model = "gemini-flash"

    s1 = _fake_session(model)  # stale — held by caller before lock acquisition
    s2 = _fake_session(model)  # current — placed by a concurrent migration
    lock = asyncio.Lock()
    client = _fake_client()

    # Seed with S1 first (the stale view the caller originally had), then
    # replace with S2 before _ensure_session_model runs, mimicking a race.
    local_cache = TTLCache(maxsize=100, ttl=3600)
    local_cache[session_id] = (s1, lock)
    local_cache[session_id] = (s2, lock)  # concurrent replacement

    with patch("gpal.server.sessions", local_cache):
        result = _ensure_session_model(session_id, client, model, config=None, lock=lock)

    assert result is s2, (
        "Must return the current sessions entry (S2), not the stale reference (S1)"
    )
    assert result is not s1, "Must NOT return the stale reference S1"


def test_ensure_session_model_evicted_entry_recovers():
    """_ensure_session_model must not raise KeyError when the TTLCache entry has
    been evicted between _get_or_create_session_entry and lock acquisition.

    On eviction (sessions.get returns None) the function must use current_session
    if provided, re-insert it under the caller's lock, and return it — no KeyError.
    """
    import asyncio
    from cachetools import TTLCache

    session_id = "eviction-test"
    target_model = "gemini-flash"
    lock = asyncio.Lock()
    client = _fake_client()
    current = _fake_session(target_model)

    # An empty cache simulates the evicted state.
    empty_cache = TTLCache(maxsize=100, ttl=3600)

    with patch("gpal.server.sessions", empty_cache):
        result = _ensure_session_model(
            session_id, client, target_model, config=None, lock=lock, current_session=current
        )

    assert result is not None, "Must return a session even after cache eviction"
    assert result._gpal_model == target_model, "_gpal_model must be set on recreated session"
    # current_session is already the target model with clean history — returned as-is and re-inserted
    assert result is current, "Should return current_session when it already matches the target model"
    assert empty_cache[session_id] == (result, lock), "Re-inserted entry must use caller's lock"


def test_ensure_session_model_evicted_no_current_session_creates_fresh():
    """When no current_session is given and the entry is evicted, a brand-new session
    is created, inserted into the cache, and returned."""
    import asyncio
    from cachetools import TTLCache

    session_id = "eviction-no-current-test"
    target_model = "gemini-flash"
    lock = asyncio.Lock()
    client = _fake_client()

    empty_cache = TTLCache(maxsize=100, ttl=3600)

    with patch("gpal.server.sessions", empty_cache):
        result = _ensure_session_model(session_id, client, target_model, config=None, lock=lock)

    assert result is not None, "Must return a session even without current_session"
    assert result._gpal_model == target_model, "_gpal_model must be set on the fresh session"
    assert empty_cache[session_id] == (result, lock), "Re-inserted entry must use caller's lock"


def test_ensure_session_model_orphan_leaves_cache_untouched():
    """Orphan semantics: when the stored lock differs from the caller's lock (the
    entry was evicted and recreated by another coroutine), _ensure_session_model
    must return current_session (the caller's own session) and NOT overwrite the
    cache — so the concurrent coroutine's entry remains authoritative and we do
    not hand two coroutines the same genai Chat object under different locks.
    """
    import asyncio

    session_id = "stale-lock-test"
    model = "gemini-flash"

    caller_lock = asyncio.Lock()
    other_lock = asyncio.Lock()
    # s1: what the caller got from _get_or_create_session_entry (already target model)
    s1 = _fake_session(model)
    # s2: what the concurrent coroutine placed after eviction (different lock)
    s2 = _fake_session(model)
    client = _fake_client()

    # Cache holds s2 under other_lock — s1 was evicted and replaced.
    patched_sessions: dict = {session_id: (s2, other_lock)}

    with patch("gpal.server.sessions", patched_sessions):
        result = _ensure_session_model(
            session_id, client, model, config=None, lock=caller_lock, current_session=s1
        )

    # We are orphaned: must return s1 (or a model-matched recreation of it, but since
    # s1 already has the target model and clean history, it is returned unchanged).
    assert result is s1, "Orphaned caller must operate on its own current_session, not s2"
    assert result is not s2, "Must NOT return or clobber s2 (concurrent coroutine's session)"

    # Cache must be untouched — s2/other_lock must remain authoritative.
    stored_session, stored_lock = patched_sessions[session_id]
    assert stored_session is s2, "Cache entry must remain s2 (concurrent coroutine's session)"
    assert stored_lock is other_lock, "Cache lock must remain other_lock, not caller_lock"


# ─────────────────────────────────────────────────────────────────────────────
# O. search_project size cap
# ─────────────────────────────────────────────────────────────────────────────


def test_search_project_skips_oversized_files(tmp_path, monkeypatch):
    """Files over MAX_FILE_SIZE are skipped; a note is appended to results."""
    monkeypatch.chdir(tmp_path)

    small = tmp_path / "small.txt"
    small.write_text("x")  # 1 byte — under the cap

    big = tmp_path / "big.txt"
    big.write_text("needle" * 10)  # well over the cap

    # Cap at 5 bytes: small (1 byte) fits, big (60 bytes) does not.
    # Neither contains "needle" yet — put it only in big so small won't match.
    # Actually we want small to match: write needle into small too.
    small.write_text("needle")  # 6 bytes
    # Cap at 7 bytes: small (6 bytes) fits, big (60 bytes) does not.
    monkeypatch.setattr("gpal.server.MAX_FILE_SIZE", 7)

    result = search_project("needle", "*.txt")

    assert "small.txt" in result, "small file with match should appear"
    assert "big.txt" not in result or "skipped" in result, "big file must not appear as a match"
    assert "skipped" in result, "skip note must be present"
    assert "size limit" in result, "skip note must mention size limit"


# ─────────────────────────────────────────────────────────────────────────────
# P. get_client caching
# ─────────────────────────────────────────────────────────────────────────────


def test_get_client_returns_cached_instance(monkeypatch):
    """Two get_client() calls return the same object; constructor called once."""
    import gpal.server as srv
    import unittest.mock as mock

    sentinel = mock.MagicMock(name="cached_client")
    call_count = [0]

    def fake_client(api_key=None):
        call_count[0] += 1
        return sentinel

    monkeypatch.setattr(srv, "_cached_client", None)
    monkeypatch.setattr("gpal.server.genai.Client", fake_client)
    monkeypatch.setattr(srv, "_load_api_key", lambda: "fake-key")

    c1 = get_client()
    c2 = get_client()

    assert c1 is c2, "Both calls must return the same cached client"
    assert call_count[0] == 1, f"Constructor called {call_count[0]} times, expected 1"


def test_get_client_force_new_replaces_cache(monkeypatch):
    """get_client(force_new=True) builds a fresh client and subsequent calls return it."""
    import gpal.server as srv
    import unittest.mock as mock

    old_client = mock.MagicMock(name="old_client")
    new_client = mock.MagicMock(name="new_client")
    clients = [new_client]

    def fake_client(api_key=None):
        return clients.pop(0)

    monkeypatch.setattr(srv, "_cached_client", old_client)
    monkeypatch.setattr("gpal.server.genai.Client", fake_client)
    monkeypatch.setattr(srv, "_load_api_key", lambda: "fake-key")

    refreshed = get_client(force_new=True)
    assert refreshed is new_client, "force_new=True must return the new client"

    again = get_client()
    assert again is new_client, "Subsequent call must return the newly cached client"


def test_get_client_force_new_closes_displaced(monkeypatch):
    """get_client(force_new=True) must close the displaced client to release its
    underlying HTTP connection pool; leaked clients exhaust file descriptors over time.
    """
    import gpal.server as srv
    import unittest.mock as mock

    old_client = mock.MagicMock(name="old_client")
    new_client = mock.MagicMock(name="new_client")
    clients = [new_client]

    def fake_client(api_key=None):
        return clients.pop(0)

    monkeypatch.setattr(srv, "_cached_client", old_client)
    monkeypatch.setattr("gpal.server.genai.Client", fake_client)
    monkeypatch.setattr(srv, "_load_api_key", lambda: "fake-key")

    get_client(force_new=True)

    old_client.close.assert_called_once_with()


# ─────────────────────────────────────────────────────────────────────────────
# Q. _stdin_disconnected helper
# ─────────────────────────────────────────────────────────────────────────────


def test_stdin_disconnected_open_pipe():
    """Returns False while the write end of the pipe is still open."""
    r, w = os.pipe()
    try:
        assert _stdin_disconnected(r) is False
    finally:
        os.close(r)
        os.close(w)


def test_stdin_disconnected_closed_write_end():
    """Returns True after the write end is closed (POLLHUP or fstat path)."""
    r, w = os.pipe()
    os.close(w)
    try:
        result = _stdin_disconnected(r)
        assert result is True, "Should detect disconnected write end"
    finally:
        os.close(r)


def test_stdin_disconnected_invalid_fd():
    """Returns True for an already-closed fd (OSError from fstat)."""
    r, w = os.pipe()
    os.close(r)
    os.close(w)
    # r is now invalid — fstat should raise OSError
    assert _stdin_disconnected(r) is True


# ─────────────────────────────────────────────────────────────────────────────
# R. tokens_in_window does not create entries for unknown models
# ─────────────────────────────────────────────────────────────────────────────


def test_tokens_in_window_absent_model_returns_zero_no_entry():
    """tokens_in_window returns 0 for an absent model and leaves no key in _token_windows."""
    absent = "not-a-real-model-xyz"
    with _token_lock:
        _token_windows.pop(absent, None)

    result = tokens_in_window(absent)

    assert result == 0, "Should return 0 for an absent model"
    with _token_lock:
        assert absent not in _token_windows, "Must not create a _token_windows entry for absent model"


# ─────────────────────────────────────────────────────────────────────────────
# S. json_mode omits tools (Gemini API rejects JSON mode + function calling)
# ─────────────────────────────────────────────────────────────────────────────


async def _get_oneshot_fn():
    """Return the underlying async function from the consult_gemini_oneshot FunctionTool."""
    tool = await mcp._local_provider._get_tool("consult_gemini_oneshot")
    return tool.fn


@pytest.mark.asyncio
async def test_oneshot_json_mode_omits_tools():
    """consult_gemini_oneshot with json_mode=True must build a config with no tools
    and response_mime_type='application/json'. The Gemini API 400s on JSON mode + tools.
    """
    from gpal.server import GeminiResponse, _consult

    captured_configs: list = []
    fake_resp = GeminiResponse("[]", 10, 10, 20)

    def fake_send(session, parts, config):
        captured_configs.append(config)
        return fake_resp

    fake_session = MagicMock()
    fake_client = MagicMock()
    fake_client.chats.create.return_value = fake_session

    oneshot_fn = await _get_oneshot_fn()

    with patch("gpal.server.get_client", return_value=fake_client), \
         patch("gpal.server._send_with_retry", side_effect=fake_send), \
         patch("gpal.server._async_throttle", new_callable=AsyncMock), \
         patch("gpal.server._build_afc_tools", return_value=["fake-tool"]):
        result = await oneshot_fn(
            query="return json", model="flash", json_mode=True, ctx=_make_ctx()
        )

    assert len(captured_configs) == 1, f"Expected exactly one send call, got {len(captured_configs)}"
    cfg = captured_configs[0]
    assert cfg.response_mime_type == "application/json", (
        f"response_mime_type must be 'application/json', got {cfg.response_mime_type!r}"
    )
    assert not cfg.tools, (
        "tools must be absent/empty in json_mode — Gemini API rejects JSON mode + function calling"
    )
    assert cfg.automatic_function_calling is None or getattr(cfg.automatic_function_calling, "disable", None) is True, (
        "automatic_function_calling must be absent or disabled in json_mode"
    )


@pytest.mark.asyncio
async def test_oneshot_no_json_mode_includes_tools():
    """consult_gemini_oneshot without json_mode must include tools and no response_mime_type."""
    from gpal.server import GeminiResponse, list_directory

    captured_configs: list = []
    fake_resp = GeminiResponse("answer", 10, 10, 20)
    # Use a real callable so GenerateContentConfig validation passes
    fake_tools = [list_directory]

    def fake_send(session, parts, config):
        captured_configs.append(config)
        return fake_resp

    fake_session = MagicMock()
    fake_client = MagicMock()
    fake_client.chats.create.return_value = fake_session

    oneshot_fn = await _get_oneshot_fn()

    with patch("gpal.server.get_client", return_value=fake_client), \
         patch("gpal.server._send_with_retry", side_effect=fake_send), \
         patch("gpal.server._async_throttle", new_callable=AsyncMock), \
         patch("gpal.server._build_afc_tools", return_value=fake_tools):
        result = await oneshot_fn(
            query="answer this", model="flash", json_mode=False, ctx=_make_ctx()
        )

    assert len(captured_configs) == 1
    cfg = captured_configs[0]
    assert cfg.tools, "tools must be present when json_mode is False"
    assert cfg.response_mime_type is None or cfg.response_mime_type != "application/json", (
        "response_mime_type must not be application/json when json_mode is False"
    )


@pytest.mark.asyncio
async def test_consult_json_mode_omits_tools():
    """_consult with json_mode=True must build a config with no tools and
    response_mime_type='application/json'. The Gemini API 400s on JSON mode + tools.
    """
    import asyncio
    from gpal.server import GeminiResponse, _consult

    captured_configs: list = []
    fake_resp = GeminiResponse("{}", 10, 10, 20)

    def fake_send(session, parts, config):
        captured_configs.append(config)
        return fake_resp

    fake_session = _fake_session("gemini-flash")
    fake_client = MagicMock()
    fake_client.chats.create.return_value = fake_session

    ctx = _make_ctx(session_id="json-mode-test")

    with patch("gpal.server.get_client", return_value=fake_client), \
         patch("gpal.server._send_with_retry", side_effect=fake_send), \
         patch("gpal.server._async_throttle", new_callable=AsyncMock), \
         patch("gpal.server._build_afc_tools", return_value=["fake-tool"]), \
         patch("gpal.server.sessions", {}), \
         patch("gpal.server._get_or_create_session_entry",
               return_value=(fake_session, asyncio.Lock())):
        result = await _consult(
            query="return json", ctx=ctx, model_alias="flash", json_mode=True
        )

    assert len(captured_configs) == 1, f"Expected 1 send call, got {len(captured_configs)}"
    cfg = captured_configs[0]
    assert cfg.response_mime_type == "application/json", (
        f"response_mime_type must be 'application/json', got {cfg.response_mime_type!r}"
    )
    assert not cfg.tools, (
        "tools must be absent/empty in json_mode — Gemini API rejects JSON mode + function calling"
    )


@pytest.mark.asyncio
async def test_consult_no_json_mode_includes_tools():
    """_consult without json_mode must include tools in the config."""
    import asyncio
    from gpal.server import GeminiResponse, _consult, list_directory

    captured_configs: list = []
    fake_resp = GeminiResponse("answer", 10, 10, 20)
    # Use a real callable so GenerateContentConfig validation passes
    fake_tools = [list_directory]

    def fake_send(session, parts, config):
        captured_configs.append(config)
        return fake_resp

    fake_session = _fake_session("gemini-flash")
    fake_client = MagicMock()
    fake_client.chats.create.return_value = fake_session

    ctx = _make_ctx(session_id="no-json-mode-test")

    with patch("gpal.server.get_client", return_value=fake_client), \
         patch("gpal.server._send_with_retry", side_effect=fake_send), \
         patch("gpal.server._async_throttle", new_callable=AsyncMock), \
         patch("gpal.server._build_afc_tools", return_value=fake_tools), \
         patch("gpal.server.sessions", {}), \
         patch("gpal.server._get_or_create_session_entry",
               return_value=(fake_session, asyncio.Lock())):
        result = await _consult(
            query="answer this", ctx=ctx, model_alias="flash", json_mode=False
        )

    assert len(captured_configs) == 1
    cfg = captured_configs[0]
    assert cfg.tools, "tools must be present when json_mode is False"
    assert cfg.response_mime_type is None or cfg.response_mime_type != "application/json", (
        "response_mime_type must not be application/json when json_mode is False"
    )
