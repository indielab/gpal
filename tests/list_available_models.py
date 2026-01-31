
import os
from google import genai
from google.genai import types

def list_models():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    print("Listing models...")
    for model in client.models.list():
        print(f"Name: {model.name}")
        # print(f"Model details: {model}") 

if __name__ == "__main__":
    list_models()
