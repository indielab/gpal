import sys
import os

sys.path.insert(0, os.path.abspath("src"))

from gpal.server import consult_gemini

print(f"Type: {type(consult_gemini)}")
print(f"Dir: {dir(consult_gemini)}")

if hasattr(consult_gemini, "fn"):
    print("Found .fn attribute")
    try:
         # Try calling the underlying function
        response = consult_gemini.fn("Hello! Confirm you are working.", session_id="test-init")
        print("Response:", response)
    except Exception as e:
        print(f"Error calling .fn: {e}")
elif hasattr(consult_gemini, "__wrapped__"):
     print("Found __wrapped__ attribute")
     response = consult_gemini.__wrapped__("Hello!", session_id="test-1")
     print("Response:", response)