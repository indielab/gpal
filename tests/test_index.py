"""Tests for semantic search index functionality."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gpal.index import (
    CodebaseIndex,
    get_index_path,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BINARY_EXTENSIONS,
    EMBEDDING_BATCH_SIZE,
    MAX_FILE_SIZE,
)


# ─────────────────────────────────────────────────────────────────────────────
# XDG Path Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_xdg_path_default():
    """Verify get_index_path returns XDG-compliant path when XDG_DATA_HOME is not set."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove XDG_DATA_HOME if present
        os.environ.pop("XDG_DATA_HOME", None)
        path = get_index_path()
        assert path == Path.home() / ".local" / "share" / "gpal" / "index"


def test_xdg_path_custom():
    """Verify get_index_path respects XDG_DATA_HOME."""
    with patch.dict(os.environ, {"XDG_DATA_HOME": "/custom/data"}):
        path = get_index_path()
        assert path == Path("/custom/data/gpal/index")


# ─────────────────────────────────────────────────────────────────────────────
# _should_index Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_client():
    """Create a mock Gemini client."""
    return MagicMock()


@pytest.fixture
def index_with_gitignore(tmp_path, mock_client):
    """Create a CodebaseIndex with a .gitignore file."""
    # Create .gitignore
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("node_modules/\n*.log\nbuild/\n__pycache__/\n")

    # Create the index (mocking chromadb)
    with patch("gpal.index.chromadb.PersistentClient"):
        index = CodebaseIndex(tmp_path, mock_client)
    return index


def test_should_index_normal_file(index_with_gitignore, tmp_path):
    """Normal source files should be indexed."""
    test_file = tmp_path / "main.py"
    test_file.write_text("print('hello')")
    assert index_with_gitignore._should_index(test_file) is True


def test_should_index_hidden_file(index_with_gitignore, tmp_path):
    """Hidden files should not be indexed."""
    hidden = tmp_path / ".hidden"
    hidden.write_text("secret")
    assert index_with_gitignore._should_index(hidden) is False


def test_should_index_hidden_dir(index_with_gitignore, tmp_path):
    """Files in hidden directories should not be indexed."""
    hidden_dir = tmp_path / ".git"
    hidden_dir.mkdir()
    config = hidden_dir / "config"
    config.write_text("git config")
    assert index_with_gitignore._should_index(config) is False


def test_should_index_binary_file(index_with_gitignore, tmp_path):
    """Binary files should not be indexed."""
    binary = tmp_path / "compiled.pyc"
    binary.write_bytes(b"\x00\x01\x02")
    assert index_with_gitignore._should_index(binary) is False


def test_should_index_gitignore_pattern(index_with_gitignore, tmp_path):
    """Files matching .gitignore patterns should not be indexed."""
    # node_modules should be ignored
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    pkg = node_modules / "some_pkg" / "index.js"
    pkg.parent.mkdir()
    pkg.write_text("module.exports = {}")
    assert index_with_gitignore._should_index(pkg) is False

    # .log files should be ignored
    log_file = tmp_path / "debug.log"
    log_file.write_text("log content")
    assert index_with_gitignore._should_index(log_file) is False


def test_should_index_respects_pycache(index_with_gitignore, tmp_path):
    """__pycache__ directories should be ignored per .gitignore."""
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    cache_file = pycache / "module.cpython-312.pyc"
    cache_file.write_bytes(b"\x00\x01\x02")
    assert index_with_gitignore._should_index(cache_file) is False


def test_should_index_multipart_extension(index_with_gitignore, tmp_path):
    """Multi-part extensions like .min.js should not be indexed."""
    minified_js = tmp_path / "bundle.min.js"
    minified_js.write_text("!function(){console.log('minified')}();")
    assert index_with_gitignore._should_index(minified_js) is False

    minified_css = tmp_path / "styles.min.css"
    minified_css.write_text("body{margin:0}")
    assert index_with_gitignore._should_index(minified_css) is False

    # But normal .js files should be indexed
    normal_js = tmp_path / "app.js"
    normal_js.write_text("console.log('hello');")
    assert index_with_gitignore._should_index(normal_js) is True


# ─────────────────────────────────────────────────────────────────────────────
# _chunk_file Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_index(tmp_path, mock_client):
    """Create a simple CodebaseIndex without .gitignore."""
    with patch("gpal.index.chromadb.PersistentClient"):
        index = CodebaseIndex(tmp_path, mock_client)
    return index


def test_chunk_file_small(simple_index, tmp_path):
    """Small files produce a single chunk."""
    small_file = tmp_path / "small.py"
    small_file.write_text("line1\nline2\nline3")

    chunks = simple_index._chunk_file(small_file)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["file"] == "small.py"
    assert chunks[0]["metadata"]["start_line"] == 1
    assert chunks[0]["metadata"]["end_line"] == 3
    assert "line1" in chunks[0]["text"]


def test_chunk_file_large(simple_index, tmp_path):
    """Large files are split into overlapping chunks."""
    # Create a file with 100 lines
    lines = [f"line {i}" for i in range(1, 101)]
    large_file = tmp_path / "large.py"
    large_file.write_text("\n".join(lines))

    chunks = simple_index._chunk_file(large_file)

    # With CHUNK_SIZE=50 and CHUNK_OVERLAP=10, 100 lines should produce:
    # Chunk 1: lines 1-50
    # Chunk 2: lines 41-90 (starting at 40, which is 50-10)
    # Chunk 3: lines 81-100 (starting at 80, which is 80)
    assert len(chunks) >= 2

    # First chunk starts at line 1
    assert chunks[0]["metadata"]["start_line"] == 1

    # Check overlap - second chunk should start before first chunk ends
    if len(chunks) > 1:
        first_end = chunks[0]["metadata"]["end_line"]
        second_start = chunks[1]["metadata"]["start_line"]
        assert second_start < first_end  # Overlap exists


def test_chunk_file_empty(simple_index, tmp_path):
    """Empty files produce no chunks."""
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")

    chunks = simple_index._chunk_file(empty_file)
    assert chunks == []


def test_chunk_file_binary_content(simple_index, tmp_path):
    """Files with binary content (non-UTF8) produce no chunks."""
    binary_file = tmp_path / "data.bin"
    binary_file.write_bytes(b"\x80\x81\x82\x83")

    chunks = simple_index._chunk_file(binary_file)
    assert chunks == []


def test_chunk_file_id_format(simple_index, tmp_path):
    """Chunk IDs follow the expected format."""
    test_file = tmp_path / "test.py"
    test_file.write_text("content")

    chunks = simple_index._chunk_file(test_file)
    assert len(chunks) == 1
    assert chunks[0]["id"] == "test.py:1-1"


# ─────────────────────────────────────────────────────────────────────────────
# Constants Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_binary_extensions_coverage():
    """Verify common binary extensions are covered."""
    expected = {".pyc", ".so", ".exe", ".png", ".jpg", ".mp3", ".zip", ".pdf"}
    for ext in expected:
        assert ext in BINARY_EXTENSIONS, f"Missing extension: {ext}"


def test_chunk_constants():
    """Verify chunking constants are sensible."""
    assert CHUNK_SIZE > CHUNK_OVERLAP, "Overlap must be smaller than chunk size"
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP >= 0


def test_max_file_size():
    """Verify MAX_FILE_SIZE matches server.py."""
    from gpal.server import MAX_FILE_SIZE as SERVER_MAX
    assert MAX_FILE_SIZE == SERVER_MAX


def test_embedding_batch_size():
    """Verify EMBEDDING_BATCH_SIZE is reasonable for API limits."""
    assert 50 <= EMBEDDING_BATCH_SIZE <= 250, "Batch size should be within typical API limits"
