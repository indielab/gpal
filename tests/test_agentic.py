"""
Manual smoke test: autonomous exploration via consult_gemini.

Run directly (requires GEMINI_API_KEY):
    uv run python tests/test_agentic.py

pytest collects this file safely — no test_* functions and no module-level execution.
"""

import asyncio
import os

from fastmcp import Client
from gpal.server import mcp


async def main():
    async with Client(mcp) as c:
        print("Testing Agentic Capabilities (Flash).")
        print("Query: autonomous discovery of the project license without file_paths hint.")

        result = await c.call_tool(
            "consult_gemini",
            {
                "query": (
                    "What license does this project use? "
                    "You MUST list the directory to find the license file, "
                    "READ the file content, and ONLY THEN answer. Do not guess. "
                    "Verify by reading the file."
                ),
                "model": "flash",
            },
        )

        print("\n--- Response from Gemini ---\n")
        print(result.content)
        print("\n----------------------------\n")


if __name__ == "__main__":
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY must be set")
        raise SystemExit(1)
    asyncio.run(main())
