import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from google.genai import types
from gpal.server import (
    list_directory, read_file, search_project, detect_mime_type, MIME_TYPES,
    create_batch, get_batch, list_batches, get_batch_results, cancel_batch, delete_batch,
    _number_lines, _build_file_context, generate_image,
)

def test_list_directory(tmp_path, monkeypatch):
    # Create a dummy structure
    (tmp_path / "subdir").mkdir()
    (tmp_path / "file1.txt").write_text("hello")
    (tmp_path / "file2.py").write_text("print('world')")

    # Change cwd so tmp_path is within "project root"
    monkeypatch.chdir(tmp_path)

    # Test listing current directory
    results = list_directory(".")
    assert "subdir/" in results   # directories get trailing "/"
    assert "file1.txt" in results
    assert "file2.py" in results
    assert len(results) == 3

def test_list_directory_nonexistent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = list_directory("non_existent_subdir")
    assert isinstance(result, str)
    assert "does not exist" in result

# --- _number_lines (pure formatting helper) ---

def test_number_lines_basic():
    out = _number_lines(["import os", "import sys"], start=1)
    rows = out.split("\n")
    assert rows[0] == "     1\timport os"   # 6-wide gutter + tab, cat -n style
    assert rows[1] == "     2\timport sys"

def test_number_lines_width_min_six():
    # Single-digit line numbers still pad to width 6 (matches cat -n).
    out = _number_lines(["only"], start=1)
    assert out == "     1\tonly"

def test_number_lines_width_grows_for_large_files():
    rows = [f"line{i}" for i in range(1, 1001)]
    out = _number_lines(rows, start=1).split("\n")
    assert out[0] == "     1\tline1"      # still width 6 (matches 1000)
    assert out[-1] == "  1000\tline1000"

def test_number_lines_respects_start_offset():
    out = _number_lines(["a", "b"], start=100)
    rows = out.split("\n")
    assert rows[0] == "   100\ta"
    assert rows[1] == "   101\tb"

def test_number_lines_empty():
    assert _number_lines([], start=1) == ""

def test_number_lines_preserves_blank_lines():
    out = _number_lines(["x", "", "y"], start=1)
    rows = out.split("\n")
    assert rows[1] == "     2\t"           # blank source line keeps its number


# --- read_file (now numbered + ranged) ---

def test_read_file(tmp_path, monkeypatch):
    test_file = tmp_path / "test.txt"
    test_file.write_text("alpha\nbeta\ngamma\n")
    monkeypatch.chdir(tmp_path)

    result = read_file("test.txt")
    assert "     1\talpha" in result
    assert "     2\tbeta" in result
    assert "     3\tgamma" in result

def test_read_file_whole_file_has_no_range_notice(tmp_path, monkeypatch):
    (tmp_path / "t.txt").write_text("one\ntwo\n")
    monkeypatch.chdir(tmp_path)
    result = read_file("t.txt")
    assert "of 2" not in result           # full read => no truncation header

def test_read_file_offset_and_limit(tmp_path, monkeypatch):
    (tmp_path / "t.txt").write_text("\n".join(f"L{i}" for i in range(1, 11)) + "\n")
    monkeypatch.chdir(tmp_path)

    result = read_file("t.txt", offset=3, limit=2)
    assert "     3\tL3" in result
    assert "     4\tL4" in result
    assert "L2" not in result             # before the window
    assert "L5" not in result             # after the window
    assert "of 10" in result              # notice reveals the file is longer

def test_read_file_offset_beyond_eof(tmp_path, monkeypatch):
    (tmp_path / "t.txt").write_text("only\n")
    monkeypatch.chdir(tmp_path)
    result = read_file("t.txt", offset=50)
    assert "beyond" in result.lower()

def test_read_file_empty(tmp_path, monkeypatch):
    (tmp_path / "empty.txt").write_text("")
    monkeypatch.chdir(tmp_path)
    result = read_file("empty.txt")
    assert "empty" in result.lower()

def test_read_file_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = read_file("this_file_does_not_exist_at_all.txt")
    assert "does not exist" in result

def test_read_file_limit_zero_returns_error(tmp_path, monkeypatch):
    (tmp_path / "t.txt").write_text("one\ntwo\n")
    monkeypatch.chdir(tmp_path)
    result = read_file("t.txt", limit=0)
    assert isinstance(result, str)
    assert "error" in result.lower()

def test_read_file_limit_negative_returns_error(tmp_path, monkeypatch):
    (tmp_path / "t.txt").write_text("one\ntwo\n")
    monkeypatch.chdir(tmp_path)
    result = read_file("t.txt", limit=-5)
    assert isinstance(result, str)
    assert "error" in result.lower()


# --- _build_file_context (inline file_paths) ---

def test_build_file_context_numbers_and_wraps(tmp_path, monkeypatch):
    f = tmp_path / "mod.py"
    f.write_text("def f():\n    return 1\n")
    monkeypatch.chdir(tmp_path)

    out = _build_file_context("mod.py")
    assert "--- START FILE: mod.py ---" in out
    assert "--- END FILE: mod.py ---" in out
    assert "     1\tdef f():" in out
    assert "     2\t    return 1" in out

def test_build_file_context_rejects_oversize(tmp_path, monkeypatch):
    from gpal.server import MAX_FILE_SIZE
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (MAX_FILE_SIZE + 1))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="exceeds"):
        _build_file_context("big.bin")

def test_search_project(tmp_path, monkeypatch):
    # Setup dummy files
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("def my_function(): pass")
    (tmp_path / "app" / "utils.py").write_text("def other_function(): pass")
    (tmp_path / "README.md").write_text("This is the main project readme.")
    
    # Use monkeypatch to change CWD to tmp_path for the search
    monkeypatch.chdir(tmp_path)
    
    # Test finding a term
    result = search_project("my_function")
    assert "app/main.py" in result
    assert "app/utils.py" not in result
    
    # Test globbing
    result = search_project("def", glob_pattern="app/*.py")
    assert "app/main.py" in result
    assert "app/utils.py" in result
    assert "README.md" not in result

def test_search_project_no_matches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = search_project("non_existent_term_xyz_123")
    assert result == "No matches found."


# ─────────────────────────────────────────────────────────────────────────────
# MIME Type Detection Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_detect_mime_type_images():
    assert detect_mime_type("photo.png") == "image/png"
    assert detect_mime_type("photo.PNG") == "image/png"  # case insensitive
    assert detect_mime_type("/path/to/image.jpg") == "image/jpeg"
    assert detect_mime_type("file.jpeg") == "image/jpeg"
    assert detect_mime_type("animation.gif") == "image/gif"
    assert detect_mime_type("modern.webp") == "image/webp"


def test_detect_mime_type_video():
    assert detect_mime_type("video.mp4") == "video/mp4"
    assert detect_mime_type("clip.mov") == "video/mov"
    assert detect_mime_type("stream.webm") == "video/webm"
    assert detect_mime_type("movie.mkv") == "video/x-matroska"


def test_detect_mime_type_audio():
    assert detect_mime_type("sound.wav") == "audio/wav"
    assert detect_mime_type("song.mp3") == "audio/mp3"
    assert detect_mime_type("track.flac") == "audio/flac"
    assert detect_mime_type("podcast.ogg") == "audio/ogg"


def test_detect_mime_type_documents():
    assert detect_mime_type("document.pdf") == "application/pdf"


def test_detect_mime_type_unknown():
    assert detect_mime_type("file.xyz") is None
    assert detect_mime_type("noextension") is None
    assert detect_mime_type(".hidden") is None


# ─────────────────────────────────────────────────────────────────────────────
# generate_image validation tests
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_image_imagen_image_size_error_does_not_create_dir(tmp_path, monkeypatch):
    """imagen model + image_size must error before _validate_output_path creates dirs."""
    monkeypatch.chdir(tmp_path)
    nonexistent_subdir = tmp_path / "new_subdir"
    output_path = str(nonexistent_subdir / "out.png")

    result = generate_image(
        prompt="a cat",
        output_path=output_path,
        model="imagen",
        image_size="1024x1024",
    )

    assert "error" in result.lower()
    assert "image_size" in result
    assert not nonexistent_subdir.exists(), (
        "Output directory must NOT be created when validation fails before _validate_output_path"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch Tool Tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_client():
    """Return a mock genai Client with async batch methods wired up."""
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.batches = MagicMock()
    client.aio.batches.create = AsyncMock()
    client.aio.batches.get = AsyncMock()
    client.aio.batches.list = AsyncMock()
    client.aio.batches.cancel = AsyncMock(return_value=None)
    client.aio.batches.delete = AsyncMock(return_value=None)
    return client


def _make_mock_job(name="batches/abc", state_value="JOB_STATE_SUCCEEDED"):
    job = MagicMock()
    job.name = name
    job.state = MagicMock()
    job.state.value = state_value
    job.completion_stats = None
    job.create_time = None
    job.end_time = None
    job.dest = MagicMock()
    job.dest.inlined_responses = []
    return job


async def _async_gen(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_create_batch_valid():
    client = _make_mock_client()
    job = _make_mock_job(state_value="JOB_STATE_PENDING")
    client.aio.batches.create.return_value = job

    with patch("gpal.server.get_client", return_value=client):
        result = await create_batch([
            {"custom_id": "q1", "prompt": "hello"},
            {"custom_id": "q2", "prompt": "world"},
        ])

    assert result["name"] == "batches/abc"
    assert result["state"] == "JOB_STATE_PENDING"
    assert result["request_count"] == 2
    client.aio.batches.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_batch_missing_custom_id():
    client = _make_mock_client()
    with patch("gpal.server.get_client", return_value=client):
        result = await create_batch([{"prompt": "no id here"}])
    assert "error" in result
    client.aio.batches.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_batch_missing_prompt():
    client = _make_mock_client()
    with patch("gpal.server.get_client", return_value=client):
        result = await create_batch([{"custom_id": "x"}])
    assert "error" in result
    client.aio.batches.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_batch():
    client = _make_mock_client()
    job = _make_mock_job(state_value="JOB_STATE_RUNNING")
    client.aio.batches.get.return_value = job

    with patch("gpal.server.get_client", return_value=client):
        result = await get_batch("batches/abc")

    assert result["name"] == "batches/abc"
    assert result["state"] == "JOB_STATE_RUNNING"
    client.aio.batches.get.assert_awaited_once_with(name="batches/abc")


@pytest.mark.asyncio
async def test_get_batch_with_completion_stats():
    client = _make_mock_client()
    job = _make_mock_job(state_value="JOB_STATE_SUCCEEDED")
    job.completion_stats = MagicMock()
    job.completion_stats.successful_count = 3
    job.completion_stats.failed_count = 0
    job.completion_stats.incomplete_count = 0
    client.aio.batches.get.return_value = job

    with patch("gpal.server.get_client", return_value=client):
        result = await get_batch("batches/abc")

    assert result["completion_stats"]["successful"] == 3
    assert result["completion_stats"]["failed"] == 0


@pytest.mark.asyncio
async def test_list_batches():
    client = _make_mock_client()
    job1 = _make_mock_job("batches/a", "JOB_STATE_SUCCEEDED")
    job2 = _make_mock_job("batches/b", "JOB_STATE_RUNNING")
    client.aio.batches.list.return_value = _async_gen([job1, job2])

    with patch("gpal.server.get_client", return_value=client):
        result = await list_batches(limit=10)

    assert result["count"] == 2
    assert result["batches"][0]["name"] == "batches/a"
    assert result["batches"][1]["state"] == "JOB_STATE_RUNNING"


@pytest.mark.asyncio
async def test_list_batches_respects_limit():
    client = _make_mock_client()
    jobs = [_make_mock_job(f"batches/{i}", "JOB_STATE_SUCCEEDED") for i in range(5)]
    client.aio.batches.list.return_value = _async_gen(jobs)

    with patch("gpal.server.get_client", return_value=client):
        result = await list_batches(limit=2)

    assert result["count"] == 2


@pytest.mark.asyncio
async def test_get_batch_results_succeeded():
    client = _make_mock_client()
    job = _make_mock_job(state_value="JOB_STATE_SUCCEEDED")

    resp = MagicMock()
    resp.metadata = {"custom_id": "q1"}
    resp.error = None
    resp.response = MagicMock()
    resp.response.text = "the answer"
    resp.response.usage_metadata = MagicMock()
    resp.response.usage_metadata.prompt_token_count = 10
    resp.response.usage_metadata.candidates_token_count = 5
    resp.response.usage_metadata.total_token_count = 15
    job.dest.inlined_responses = [resp]

    client.aio.batches.get.return_value = job

    with patch("gpal.server.get_client", return_value=client):
        result = await get_batch_results("batches/abc")

    assert result["count"] == 1
    item = result["results"][0]
    assert item["custom_id"] == "q1"
    assert item["status"] == "succeeded"
    assert item["text"] == "the answer"
    assert item["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_get_batch_results_partially_succeeded():
    """PARTIALLY_SUCCEEDED should be treated as a valid results state."""
    client = _make_mock_client()
    job = _make_mock_job(state_value="JOB_STATE_PARTIALLY_SUCCEEDED")
    job.dest.inlined_responses = []
    client.aio.batches.get.return_value = job

    with patch("gpal.server.get_client", return_value=client):
        result = await get_batch_results("batches/abc")

    assert "error" not in result
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_get_batch_results_dest_none():
    """job.dest=None should not crash — return empty results."""
    client = _make_mock_client()
    job = _make_mock_job(state_value="JOB_STATE_SUCCEEDED")
    job.dest = None
    client.aio.batches.get.return_value = job

    with patch("gpal.server.get_client", return_value=client):
        result = await get_batch_results("batches/abc")

    assert "error" not in result
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_get_batch_results_not_complete():
    client = _make_mock_client()
    job = _make_mock_job(state_value="JOB_STATE_RUNNING")
    client.aio.batches.get.return_value = job

    with patch("gpal.server.get_client", return_value=client):
        result = await get_batch_results("batches/abc")

    assert "error" in result
    assert "JOB_STATE_RUNNING" in result["error"]


@pytest.mark.asyncio
async def test_cancel_batch():
    client = _make_mock_client()
    job = _make_mock_job(state_value="JOB_STATE_CANCELLING")
    client.aio.batches.get.return_value = job

    with patch("gpal.server.get_client", return_value=client):
        result = await cancel_batch("batches/abc")

    client.aio.batches.cancel.assert_awaited_once_with(name="batches/abc")
    client.aio.batches.get.assert_awaited_once_with(name="batches/abc")
    assert result["state"] == "JOB_STATE_CANCELLING"


@pytest.mark.asyncio
async def test_delete_batch():
    client = _make_mock_client()
    job = _make_mock_job(state_value="JOB_STATE_SUCCEEDED")
    client.aio.batches.get.return_value = job

    with patch("gpal.server.get_client", return_value=client):
        result = await delete_batch("batches/abc")

    client.aio.batches.delete.assert_awaited_once_with(name="batches/abc")
    assert result == {"deleted": "batches/abc"}


@pytest.mark.asyncio
async def test_delete_batch_rejects_running():
    """Deleting a running batch should return an error, not crash."""
    client = _make_mock_client()
    job = _make_mock_job(state_value="JOB_STATE_RUNNING")
    client.aio.batches.get.return_value = job

    with patch("gpal.server.get_client", return_value=client):
        result = await delete_batch("batches/abc")

    assert "error" in result
    client.aio.batches.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_batch_partially_succeeded():
    """PARTIALLY_SUCCEEDED should be treated as an ended state for deletion."""
    client = _make_mock_client()
    job = _make_mock_job(state_value="JOB_STATE_PARTIALLY_SUCCEEDED")
    client.aio.batches.get.return_value = job

    with patch("gpal.server.get_client", return_value=client):
        result = await delete_batch("batches/abc")

    client.aio.batches.delete.assert_awaited_once_with(name="batches/abc")
    assert result == {"deleted": "batches/abc"}
