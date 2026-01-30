# gpal

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io/)

An MCP server that gives your IDE or agent access to Google Gemini with autonomous codebase exploration. Your pal Gemini.

## Why gpal?

When you ask gpal a question, Gemini doesn't just guess — it **explores your codebase itself**. It lists directories, reads files, and searches for patterns before answering. This makes it ideal for:

- 🔍 **Deep code analysis** — "Find all error handling patterns in this codebase"
- 🏗️ **Architectural reviews** — "How is authentication implemented?"
- 🐛 **Bug hunting** — "Why might this function return null?"
- 📚 **Codebase onboarding** — "Explain how the request pipeline works"
- 🖼️ **Visual review** — Analyze screenshots, diagrams, video via `media_paths`
- 📋 **Structured extraction** — "List all API endpoints as JSON"

## Features

| Feature | Description |
|---------|-------------|
| **Stateful sessions** | Maintains conversation history via `session_id` |
| **Autonomous exploration** | Gemini has tools to list, read, and search files |
| **2M token context** | Leverages Gemini 3's massive context window |
| **Two-tier consultation** | Flash for speed, Pro for depth |
| **Seamless switching** | History preserved when switching between Flash and Pro |
| **Multimodal** | Analyze images, audio, video, PDFs |
| **File uploads** | Upload large files to Gemini's File API |
| **Structured output** | JSON mode with optional schema constraints |
| **Nested agency** | Claude can delegate entire tasks to Gemini |

**Limits:** 10MB file reads, 20MB inline media, 20 search matches max.

### Flash vs Pro

| Model | Use Case | Strengths |
|-------|----------|-----------|
| `consult_gemini_flash` | **Scout** — exploration first | Fast, efficient, great for searching and mapping |
| `consult_gemini_pro` | **Architect** — analysis second | Deep reasoning, synthesis, complex reviews |

**Workflow:** Start with Flash to gather context, then switch to Pro for analysis. Both share the same session history.

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended)
- [Gemini API key](https://aistudio.google.com/)

### Quick Start

```bash
git clone https://github.com/tobert/gpal.git
cd gpal
export GEMINI_API_KEY="your_key_here"  # or GOOGLE_API_KEY
uv run gpal
```

## Usage

### Claude Desktop / Cursor / VS Code

Add to your MCP config (e.g., `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "gpal": {
      "command": "uv",
      "args": ["--directory", "/path/to/gpal", "run", "gpal"],
      "env": {
        "GEMINI_API_KEY": "your_key_here"
      }
    }
  }
}
```

Then ask your AI assistant:

> "Ask Gemini to analyze the authentication flow in this codebase"

> "Use `consult_gemini_flash` to find where errors are handled"

## Development

```bash
uv run pytest              # Run tests
uv run pytest -v           # Verbose output
```

## License

MIT — see [LICENSE](LICENSE)
