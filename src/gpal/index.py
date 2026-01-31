"""
gpal semantic search index using Gemini embeddings + chromadb.

Provides vector-based code search that finds code by meaning rather than
exact text matching.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import chromadb
import pathspec
from google import genai

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB - matches server.py
CHUNK_SIZE = 50  # lines per chunk
CHUNK_OVERLAP = 10  # overlapping lines between chunks
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_BATCH_SIZE = 100  # Max chunks per Gemini API call

# Binary/generated file extensions to skip
BINARY_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".o", ".obj", ".bin", ".exe", ".dll",
    ".class", ".jar", ".war", ".ear",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".whl", ".egg",
    ".min.js", ".min.css",  # minified files
}


# ─────────────────────────────────────────────────────────────────────────────
# XDG Path Helper
# ─────────────────────────────────────────────────────────────────────────────


def get_index_path() -> Path:
    """Get XDG-compliant path for index storage."""
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        base = Path(xdg_data)
    else:
        base = Path.home() / ".local" / "share"
    return base / "gpal" / "index"


# ─────────────────────────────────────────────────────────────────────────────
# CodebaseIndex
# ─────────────────────────────────────────────────────────────────────────────


class CodebaseIndex:
    """
    Semantic code search index using Gemini embeddings and chromadb.

    Each project root gets a unique index directory based on a hash of
    the absolute path. The index persists across sessions.
    """

    def __init__(self, root: Path, client: genai.Client):
        """
        Initialize the index for a project root.

        Args:
            root: The project root directory to index.
            client: A configured Gemini API client.
        """
        self.root = root.resolve()
        self.client = client
        self.db_path = get_index_path() / self._path_hash()

        # Ensure directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.chroma = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.chroma.get_or_create_collection(
            name="code",
            metadata={"hnsw:space": "cosine"},
        )
        self._load_gitignore()

    def _path_hash(self) -> str:
        """Generate a unique hash for this project root."""
        return hashlib.md5(str(self.root).encode()).hexdigest()[:12]

    def _load_gitignore(self) -> None:
        """Load .gitignore patterns for filtering files."""
        self.ignore_spec: pathspec.PathSpec | None = None
        gitignore = self.root / ".gitignore"
        if gitignore.exists():
            try:
                patterns = gitignore.read_text(encoding="utf-8").splitlines()
                self.ignore_spec = pathspec.PathSpec.from_lines("gitignore", patterns)
            except (OSError, UnicodeDecodeError):
                pass  # If we can't read .gitignore, just skip it

    def _should_index(self, path: Path) -> bool:
        """
        Check if a file should be indexed.

        Skips:
        - Hidden files/directories (starting with .)
        - Binary files
        - Files over MAX_FILE_SIZE
        - Files matching .gitignore patterns
        """
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            return False

        # Skip hidden files/directories
        if any(part.startswith(".") for part in rel.parts):
            return False

        # Skip binary extensions (check full filename for multi-part like .min.js)
        name = path.name.lower()
        if any(name.endswith(ext) for ext in BINARY_EXTENSIONS):
            return False

        # Skip large files
        try:
            if path.stat().st_size > MAX_FILE_SIZE:
                return False
        except OSError:
            return False

        # Check .gitignore patterns
        if self.ignore_spec and self.ignore_spec.match_file(str(rel)):
            return False

        return True

    def _chunk_file(self, path: Path) -> list[dict]:
        """
        Split a file into overlapping chunks with metadata.

        Returns a list of dicts with id, text, and metadata for each chunk.
        """
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            return []

        if not lines:
            return []

        rel_path = str(path.relative_to(self.root))
        chunks = []

        for i in range(0, len(lines), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_lines = lines[i : i + CHUNK_SIZE]
            if not chunk_lines:
                continue

            start_line = i + 1
            end_line = i + len(chunk_lines)

            chunks.append({
                "id": f"{rel_path}:{start_line}-{end_line}",
                "text": "\n".join(chunk_lines),
                "metadata": {
                    "file": rel_path,
                    "start_line": start_line,
                    "end_line": end_line,
                },
            })

        return chunks

    def _embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """
        Get embeddings from Gemini for a list of texts.

        Automatically batches requests to stay within API limits.

        Args:
            texts: List of text strings to embed.
            task_type: Either "RETRIEVAL_DOCUMENT" for indexing or
                       "RETRIEVAL_QUERY" for searching.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            response = self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config={"task_type": task_type},
            )
            all_embeddings.extend([e.values for e in response.embeddings])

        return all_embeddings

    def index_file(self, path: Path) -> None:
        """
        Index or re-index a single file.

        Removes any existing chunks for the file before adding new ones.
        """
        try:
            rel_path = str(path.relative_to(self.root))
        except ValueError:
            return

        # Delete old chunks for this file
        existing = self.collection.get(where={"file": rel_path})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        if not self._should_index(path):
            return

        chunks = self._chunk_file(path)
        if not chunks:
            return

        # Batch embed all chunks
        texts = [c["text"] for c in chunks]
        embeddings = self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in chunks],
        )

    def rebuild(self) -> int:
        """
        Full rebuild of the index.

        Clears all existing data and re-indexes the entire codebase.

        Returns:
            Number of files indexed.
        """
        # Clear existing collection
        self.chroma.delete_collection("code")
        self.collection = self.chroma.create_collection(
            name="code",
            metadata={"hnsw:space": "cosine"},
        )

        count = 0
        for path in self.root.rglob("*"):
            if path.is_file() and self._should_index(path):
                self.index_file(path)
                count += 1

        return count

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search for code matching a natural language query.

        Args:
            query: Natural language description of what to find.
            limit: Maximum number of results to return.

        Returns:
            List of matches with file, lines, score, and snippet.
        """
        # Embed query with RETRIEVAL_QUERY task type
        embeddings = self._embed([query], task_type="RETRIEVAL_QUERY")
        if not embeddings:
            return []

        query_embedding = embeddings[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
        )

        # Format results
        matches = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        for i, doc in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else None

            # Cosine distance → similarity score
            score = round(1 - dist, 3) if dist is not None else None

            # Truncate snippet for display
            snippet = doc[:200] + "..." if len(doc) > 200 else doc

            matches.append({
                "file": meta.get("file", "unknown"),
                "lines": f"{meta.get('start_line', '?')}-{meta.get('end_line', '?')}",
                "score": score,
                "snippet": snippet,
            })

        return matches
