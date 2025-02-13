import os
import json
import sqlite3
from datetime import datetime, timedelta
import random
import numpy as np
from PIL import Image
import wave
import struct
from pydub import AudioSegment

def ensure_directory_permissions(directory):
    """Ensure directory exists and has correct permissions"""
    os.makedirs(directory, exist_ok=True)
    os.chmod(directory, 0o755)  # rwxr-xr-x
    
def create_test_files(base_dir):
    """Create valid test image and audio files with proper permissions"""
    ensure_directory_permissions(base_dir)
    
    try:
        # Create a simple test image
        img_path = os.path.join(base_dir, 'input.jpg')
        img_size = (800, 600)
        img_array = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        img_array[:, :] = [255, 255, 255]  # White background
        img_array[100:300, 200:400] = [255, 0, 0]  # Red rectangle
        img_array[250:450, 350:550] = [0, 255, 0]  # Green rectangle
        
        img = Image.fromarray(img_array)
        img.save(img_path, "JPEG")
        os.chmod(img_path, 0o644)  # rw-r--r--
        print("Created valid test image: input.jpg")
        
    except Exception as e:
        print(f"Warning: Failed to create test image: {str(e)}")
    
    try:
        # Create a simple audio file (1 second of silence)
        mp3_path = os.path.join(base_dir, 'audio.mp3')
        audio = AudioSegment.silent(duration=1000)
        audio.export(mp3_path, format="mp3")
        os.chmod(mp3_path, 0o644)  # rw-r--r--
        print("Created valid test audio: audio.mp3")
            
    except Exception as e:
        print(f"Warning: Failed to create test audio: {str(e)}")
        # Create empty audio file as placeholder
        try:
            with open(mp3_path, 'wb') as f:
                f.write(b'')
            os.chmod(mp3_path, 0o644)
            print("Created empty placeholder audio file")
        except Exception as inner_e:
            print(f"Warning: Failed to create placeholder audio file: {str(inner_e)}")


def setup_test_environment():
    """Creates necessary directories and test data files"""
    base_dir = '/data'
    print(f"Creating test environment at: {base_dir}")

    # Create required directories
    for dir_path in ['logs', 'docs', 'repo']:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

    try:
        # Generate dates.txt
        start_date = datetime(2024, 1, 1)
        dates = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(100)]
        with open(os.path.join(base_dir, 'dates.txt'), 'w') as f:
            f.write('\n'.join(dates))
        print("Created dates.txt")

        # Generate contacts.json
        contacts = [
            {"first_name": "John", "last_name": "Doe", "email": "john@example.com"},
            {"first_name": "Alice", "last_name": "Smith", "email": "alice@example.com"},
            {"first_name": "Bob", "last_name": "Johnson", "email": "bob@example.com"},
            {"first_name": "Emma", "last_name": "Brown", "email": "emma@example.com"}
        ]
        with open(os.path.join(base_dir, 'contacts.json'), 'w') as f:
            json.dump(contacts, f, indent=2)
        print("Created contacts.json")

        # Generate sample logs
        for i in range(15):
            timestamp = datetime.now() - timedelta(hours=i)
            log_content = f"[{timestamp.isoformat()}] Sample log entry #{i}\nMore details here\n"
            with open(os.path.join(base_dir, 'logs', f'app_{i}.log'), 'w') as f:
                f.write(log_content)
        print("Created 15 log files")

        # Generate sample Markdown file
        md_content = """# Sample Document
## Introduction
This is a test markdown file.
"""
        with open(os.path.join(base_dir, 'docs', 'sample1.md'), 'w') as f:
            f.write(md_content)
        print("Created markdown file")

        # Generate a test SQLite database
        db_path = os.path.join(base_dir, 'analytics.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                category TEXT,
                amount INTEGER
            )
        """)
        cursor.executemany("INSERT INTO sales (category, amount) VALUES (?, ?)", [
            ("Electronics", 500),
            ("Clothing", 200),
            ("Books", 150)
        ])
        conn.commit()
        conn.close()
        print("Created analytics.db with sample data")

        # Create valid test files (image and audio)
        create_test_files(base_dir)

        # Generate a sample Markdown file
        markdown_path = os.path.join(base_dir, 'input.md')
        with open(markdown_path, 'w') as f:
            f.write("# Sample Markdown\n\nThis is a test document.")
        print("Created input.md")

        # Setup a local empty Git repo for testing
        repo_path = os.path.join(base_dir, 'repo')
        os.system(f"git init {repo_path}")
        with open(os.path.join(repo_path, "README.md"), "w") as f:
            f.write("# Test Repo\n\nThis is a test repository.")
        os.system(f"cd {repo_path} && git add README.md && git commit -m 'Initial commit'")
        print("Created test Git repository")

        # Verify existence of critical files
        print("\n✅ Verification:")
        for file_name in ["dates.txt", "contacts.json", "analytics.db", "audio.mp3", "input.jpg", "input.md"]:
            print(f"{file_name} exists: {os.path.exists(os.path.join(base_dir, file_name))}")

    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        raise

if __name__ == "__main__":
    setup_test_environment()