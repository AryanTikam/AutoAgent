from openai import OpenAI
import os
import json

def call_llm(prompt, max_retries=3):
    """
    Calls GPT-4o-Mini with retry logic and better error handling.
    Uses the new OpenAI client syntax.
    """
    api_key = os.getenv("AIPROXY_TOKEN")
    
    if not api_key:
        raise ValueError("AIPROXY_TOKEN environment variable is required")

    client = OpenAI(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant focused on data processing tasks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # Lower temperature for more deterministic responses
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to get LLM response after {max_retries} attempts: {str(e)}")
            continue

def validate_file_path(path):
    """
    Validates that the file path is within allowed boundaries.
    """
    if not path.startswith("/data/"):
        raise ValueError("Access denied: Can only access files in /data directory")
    
    # Prevent directory traversal attacks
    normalized_path = os.path.normpath(path)
    if not normalized_path.startswith("/data/"):
        raise ValueError("Access denied: Invalid path")
    
    return normalized_path