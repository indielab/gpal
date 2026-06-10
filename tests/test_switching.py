"""
Manual smoke test: history migration across model switch (flash -> pro).

Sends a question to flash, then asks pro to recall the answer — verifying
that conversation history is preserved when the model changes mid-session.

Run directly (requires GEMINI_API_KEY):
    uv run python tests/test_switching.py

pytest collects this file safely — no test_* functions and no module-level execution.
"""

import asyncio
import os

from fastmcp import Client
from gpal.server import mcp


async def main():
    print("Testing Model Switching (Flash -> Pro)...")

    # Both calls share the same MCP Client session — history carries over automatically.
    async with Client(mcp) as c:
        print("\n[Step 1] Asking Flash (2+2)...")
        r1 = await c.call_tool(
            "consult_gemini",
            {
                "query": "What is 2+2? Only answer with the number.",
                "model": "flash",
            },
        )
        print(f"Flash Answer: {r1.content}")

        print("\n[Step 2] Asking Pro (recall + multiply)...")
        r2 = await c.call_tool(
            "consult_gemini",
            {
                "query": "Multiply that number by 10. Answer with the number only.",
                "model": "pro",
            },
        )
        answer = str(r2.content)
        print(f"Pro Answer: {answer}")

        if "40" in answer:
            print("\nSUCCESS: Context preserved across model switch!")
        else:
            print("\nFAILURE: Context lost.")


if __name__ == "__main__":
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY must be set")
        raise SystemExit(1)
    asyncio.run(main())
