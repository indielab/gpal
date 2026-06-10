"""
Manual smoke test: basic connectivity to flash and pro models.

Run directly (requires GEMINI_API_KEY):
    uv run python tests/test_connectivity.py

pytest collects this file safely — no test_* functions and no module-level execution.
"""

import asyncio
import os

from fastmcp import Client
from gpal.server import mcp


async def main():
    async with Client(mcp) as c:
        print("Testing Connectivity...")

        print("Ping Flash...")
        r1 = await c.call_tool(
            "consult_gemini",
            {"query": "Ping", "model": "flash"},
        )
        print(f"Flash: {r1.content}")

        print("Ping Pro...")
        r2 = await c.call_tool(
            "consult_gemini",
            {"query": "Ping", "model": "pro"},
        )
        print(f"Pro: {r2.content}")


if __name__ == "__main__":
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY must be set")
        raise SystemExit(1)
    asyncio.run(main())
