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
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to get LLM response after {max_retries} attempts: {str(e)}")
            continue

def validate_file_path(path, allow_write=False):
    """
    Validates that the file path is within /data/ and prevents unauthorized access.
    `allow_write` controls whether the path can be written to.
    """
    base_dir = "/data/"
    
    # Ensure the path is within /data/
    normalized_path = os.path.normpath(path)
    if not normalized_path.startswith(base_dir):
        raise ValueError(f"Access denied: {path} is outside the allowed directory")

    # Prevent deletions by ensuring no file is being removed
    if not allow_write and not os.path.exists(normalized_path):
        raise ValueError(f"Access denied: {path} does not exist")
    
    return normalized_path
