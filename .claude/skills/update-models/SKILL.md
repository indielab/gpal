---
name: update-models
description: >-
  This skill should be used when the user asks to "update models",
  "check model freshness", "update Gemini models", "what models are available",
  "switch to a newer model", "upgrade models", "deprecated model",
  "model not found", or discusses model version maintenance for the gpal
  MCP server.
---

# Update Models

Maintain and update Gemini model IDs configured in the gpal MCP server.

## Overview

gpal configures model IDs as constants at the top of `src/gpal/server.py`. Google
regularly releases new model versions and deprecates old ones. This skill provides
the workflow for checking freshness, discovering new models, updating constants,
and verifying the changes.

## Workflow

### Step 1: Check Current Model Freshness

Read the `gpal://models/check` MCP resource. It compares every configured model
against the Google API's available models list and reports status:

- **ok** — Model exists in the API
- **alias** — Uses a `-latest` alias (resolved server-side, always valid)
- **not_listed** — Model not found in API (may be deprecated or renamed)

### Step 2: Discover Available Models

Call the `list_models` MCP tool to see all models grouped by capability
(generateContent, generateImages, embedContent, etc.).

Compare available models against configured constants. Look for:
- Newer versions of existing models (e.g., `gemini-3-flash-001` replacing `gemini-3-flash-preview`)
- New model families worth adopting
- Deprecated models that need replacement

### Step 3: Decide on Updates

Apply these rules when choosing model updates:

| Slot | Strategy | Rationale |
|------|----------|-----------|
| Flash/Pro consult | Pin to specific version | Capabilities matter; test before switching |
| Search/Code exec | Use `-latest` alias | Stateless utility calls; safe to auto-update |
| Image generation | Pin to specific version | Output quality varies between versions |
| Speech | Pin to specific version | Voice behavior changes between versions |

Prefer GA models (dated suffix like `-001`) over preview models when available.

### Step 4: Apply Changes

Update all locations listed in `references/model-update-checklist.md`. The key files:

1. **`src/gpal/server.py`** — Model constants, `MODEL_ALIASES`, `NANO_BANANA_MODELS`
2. **`CLAUDE.md`** — Model Strategy table
3. **`pyproject.toml` + `src/gpal/__init__.py`** — Version bump (both must match)

### Step 5: Verify

1. Run tests: `uv run pytest tests/test_server.py tests/test_tools.py tests/test_index.py -v`
2. Dogfood with Gemini Pro: ask it to review the changes
3. Reinstall: `uv cache clean --force gpal && uv tool install --force /home/atobey/src/gpal`
4. Reconnect MCP (`/mcp`) and read `gpal://models/check` to confirm

## Additional Resources

### Reference Files

- **`references/model-update-checklist.md`** — Complete list of files to modify,
  naming conventions, Nano Banana vs Imagen details, and verification steps
