import os
import google.generativeai as genai
from dotenv import load_dotenv

try:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file.")
    else:
        genai.configure(api_key=api_key)

        print("Finding available models for your API key...")

        # List all available models and filter for the ones that support generateContent
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"- {model.name}")

except Exception as e:
    print(f"An error occurred: {e}")
