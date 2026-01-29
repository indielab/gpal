import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath("src"))

from gpal.server import consult_gemini

print("Testing Agentic Capabilities...")
print("Query: 'What license does this project use? Please verify by reading the file.'")

try:
    # Note: We do NOT pass file_paths. We expect Gemini to find it.
    response = consult_gemini.fn(
        "What license does this project use? Please verify by reading the file.", 
        session_id="agentic-test-1"
    )
    print("\n--- Response from Gemini ---\n")
    print(response)
    print("\n----------------------------\n")
except Exception as e:
    print(f"Error: {e}")
