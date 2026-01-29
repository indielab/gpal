# gpal (Gemini Principal Assistant Layer)

> **Your "Second Brain" for Software Engineering, powered by Gemini 3.**

`gpal` is a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that gives your IDE or agentic workflow access to a Principal Software Engineer persona. It wraps the latest Google Gemini models in a stateful, tool-equipped interface designed for deep code analysis.

## Features

*   **Stateful Consulting:** Maintains conversation history (`session_id`), allowing for iterative debugging and architectural debates.
*   **High-Agency:** Equipped with internal tools (`list_directory`, `read_file`, `search_project`) to autonomously explore your codebase. It doesn't just guess; it checks the files.
*   **Massive Context:** Leverages Gemini's 1M+ token context window to ingest entire modules or documentation sets.
*   **Open Standard:** Built on MCP, making it compatible with Claude Desktop, Cursor, VS Code, and other MCP clients.

## Installation

### Prerequisites

*   Python 3.12+
*   [`uv`](https://github.com/astral-sh/uv) (Recommended)
*   Google Gemini API Key (Get one at [AI Studio](https://aistudio.google.com/))

### Running Standalone

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/yourusername/gpal.git
    cd gpal
    ```

2.  **Run with `uv`:**
    ```bash
    export GEMINI_API_KEY="your_key_here"
    uv run gpal
    ```

## Usage

### with Claude Desktop

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gpal": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/gpal",
        "run",
        "gpal"
      ],
      "env": {
        "GEMINI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

Once connected, simply ask Claude:
> *"Ask Gemini to review `src/main.py` for security vulnerabilities."*
> *"Consult Gemini: why is my build failing? (It will autonomously search for errors)"*

### Programmatic Usage

You can use `gpal` as a library in your own Python agents:

```python
from gpal.server import consult_gemini

# Ask a question, Gemini will search the current directory to answer
response = consult_gemini.fn(
    "What license does this project use?", 
    session_id="dev-session-1"
)
print(response)
```

## Development

*   **Test:** `uv run pytest`
*   **Lint:** `uv run ruff check .` (if configured)

## License

MIT License. See [LICENSE](LICENSE) for details.
