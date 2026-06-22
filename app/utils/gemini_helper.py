import os
from google import genai
from google.genai import types

# ✅ New Client-based API (google-genai SDK)
client = genai.Client(api_key="AIzaSyCnISStgJQPPCcFgUSLH69mrwGbyPQnveQ")

def get_gemini_response(prompt: str) -> str:
    """
    Get AI medical explanation from Gemini 2.5 Flash.
    Uses the new google-genai SDK (google-generativeai is deprecated).
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a helpful medical assistant. "
                    "Always respond in clear, simple language. "
                    "Never diagnose — only explain symptoms and suggest consulting a doctor."
                ),
                max_output_tokens=800,
                temperature=0.4,       # Lower = more factual, less creative
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0  # 0 = thinking OFF (faster, cheaper for healthcare info)
                )
            )
        )
        return response.text

    except Exception as e:
        return f"⚠️ AI explanation unavailable: {str(e)}"