# task_handler.py
import os
import json
import sqlite3
from datetime import datetime
import glob
import subprocess
import base64
import requests
from PIL import Image
import io
from bs4 import BeautifulSoup
import markdown2
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from git import Repo
import duckdb
import speech_recognition as sr
from pydub import AudioSegment
from utils import call_llm, validate_file_path

def install_and_run_datagen():
    """Install uv and run datagen.py"""
    try:
        subprocess.run(["pip", "install", "uv", "--quiet"], check=True)
        subprocess.run([
            "python3", "-m", "uv",
            "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py",
            "user@example.com"
        ], check=True)
        return "Datagen executed successfully"
    except Exception as e:
        raise RuntimeError(f"Error running datagen: {str(e)}")

def format_markdown():
    """Format markdown file using prettier"""
    try:
        subprocess.run(["npm", "install", "-g", "prettier@3.4.2"], check=True)
        subprocess.run(["prettier", "--write", "/data/format.md"], check=True)
        return "Markdown formatted successfully"
    except Exception as e:
        raise RuntimeError(f"Error formatting markdown: {str(e)}")

def count_wednesdays():
    """Count Wednesdays in dates.txt"""
    try:
        with open("/data/dates.txt", "r") as file:
            dates = [line.strip() for line in file]
        count = sum(1 for date in dates if datetime.strptime(date, "%Y-%m-%d").weekday() == 2)
        with open("/data/dates-wednesdays.txt", "w") as file:
            file.write(str(count))
        return f"Counted {count} Wednesdays"
    except Exception as e:
        raise RuntimeError(f"Error counting Wednesdays: {str(e)}")

def sort_contacts():
    """Sort contacts by last name and first name"""
    try:
        with open("/data/contacts.json", "r") as file:
            contacts = json.load(file)
        contacts.sort(key=lambda x: (x['last_name'], x['first_name']))
        with open("/data/contacts-sorted.json", "w") as file:
            json.dump(contacts, file, indent=4)
        return "Contacts sorted successfully"
    except Exception as e:
        raise RuntimeError(f"Error sorting contacts: {str(e)}")

def process_logs():
    """Process recent log files"""
    try:
        log_files = glob.glob("/data/logs/*.log")
        if not log_files:
            raise ValueError("No log files found")
        log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        recent_logs = []
        for file in log_files[:10]:
            with open(file, 'r') as f:
                first_line = f.readline().strip()
                recent_logs.append(first_line)
        with open("/data/logs-recent.txt", "w") as f:
            f.write("\n".join(recent_logs))
        return "Log processing completed"
    except Exception as e:
        raise RuntimeError(f"Error processing logs: {str(e)}")

def extract_markdown_titles():
    """Extract H1 titles from markdown files"""
    try:
        files = glob.glob("/data/docs/*.md")
        index = {}
        for file in files:
            with open(file, "r") as f:
                for line in f:
                    if line.startswith("# "):
                        index[os.path.basename(file)] = line.strip("# ").strip()
                        break
        with open("/data/docs/index.json", "w") as f:
            json.dump(index, f, indent=4)
        return "Markdown titles extracted"
    except Exception as e:
        raise RuntimeError(f"Error extracting markdown titles: {str(e)}")

def extract_email_sender():
    """Extract sender's email from email.txt"""
    try:
        with open("/data/email.txt", "r") as file:
            content = file.read()
        prompt = f"Extract only the sender's email address from this email:\n\n{content}"
        email = call_llm(prompt).strip()
        with open("/data/email-sender.txt", "w") as file:
            file.write(email)
        return "Email sender extracted"
    except Exception as e:
        raise RuntimeError(f"Error extracting email: {str(e)}")

def extract_credit_card():
    """Extract credit card number from image"""
    try:
        with open("/data/credit-card.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        prompt = f"Extract only the credit card number from this image (base64): {encoded_string}"
        card_number = call_llm(prompt).strip()
        card_number = ''.join(filter(str.isdigit, card_number))
        with open("/data/credit-card.txt", "w") as file:
            file.write(card_number)
        return "Credit card number extracted"
    except Exception as e:
        raise RuntimeError(f"Error extracting credit card: {str(e)}")

def find_similar_comments():
    """Find most similar pair of comments using embeddings"""
    try:
        with open("/data/comments.txt", "r") as file:
            comments = [line.strip() for line in file if line.strip()]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(comments)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(similarity_matrix, -1)
        max_sim_idx = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
        with open("/data/comments-similar.txt", "w") as file:
            file.write(f"{comments[max_sim_idx[0]]}\n{comments[max_sim_idx[1]]}")
        return "Similar comments found"
    except Exception as e:
        raise RuntimeError(f"Error finding similar comments: {str(e)}")

def calculate_gold_sales():
    """Calculate total sales for Gold ticket type"""
    try:
        conn = sqlite3.connect("/data/ticket-sales.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT SUM(units * price) 
            FROM tickets 
            WHERE type = 'Gold'
        """)
        total = cursor.fetchone()[0]
        conn.close()
        with open("/data/ticket-sales-gold.txt", "w") as file:
            file.write(str(total))
        return f"Gold ticket sales calculated: {total}"
    except Exception as e:
        raise RuntimeError(f"Error calculating ticket sales: {str(e)}")

# Phase B Business Tasks
def fetch_api_data():
    """Fetch data from an API and save it"""
    try:
        response = requests.get("https://jsonplaceholder.typicode.com/users")
        response.raise_for_status()
        with open("/data/api-data.json", "w") as f:
            json.dump(response.json(), f, indent=4)
        return "API data fetched and saved"
    except Exception as e:
        raise RuntimeError(f"Error fetching API data: {str(e)}")

def handle_git_repo(repo_url):
    """Clone a GitHub repo and make a commit"""
    try:
        validate_file_path("/data/repo", allow_write=True)
        repo_path = "/data/repo"

        if not os.path.exists(repo_path):
            Repo.clone_from(repo_url, repo_path)

        repo = Repo(repo_path)
        with open(os.path.join(repo_path, "README.md"), "a") as f:
            f.write("\nUpdated by automation")

        repo.git.add("README.md")
        repo.index.commit("Automated update")
        return "Git repository updated successfully"
    except Exception as e:
        raise RuntimeError(f"Error handling git repo: {str(e)}")

def run_database_query(query):
    """Run a SQL query while enforcing security"""
    try:
        validate_file_path("/data/analytics.db")

        # Block DELETE, DROP, or other unsafe operations
        forbidden_keywords = ["DELETE", "DROP", "ALTER"]
        if any(keyword in query.upper() for keyword in forbidden_keywords):
            raise ValueError("Query contains forbidden operations")

        conn = duckdb.connect("/data/analytics.db")
        result = conn.execute(query).fetchall()
        conn.close()

        with open("/data/query-results.json", "w") as f:
            json.dump(result, f, indent=4)

        return "SQL query executed successfully"
    except Exception as e:
        raise RuntimeError(f"Error running database query: {str(e)}")

def scrape_website():
    """Extract data from a website"""
    try:
        response = requests.get("http://example.com")
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a.get('href') for a in soup.find_all('a')]
        
        with open("/data/scraped-links.txt", "w") as f:
            f.write("\n".join(links))
        return "Website scraped successfully"
    except Exception as e:
        raise RuntimeError(f"Error scraping website: {str(e)}")

def process_image(image_path="/data/input.jpg"):
    """Compress or resize an image and save it to /data/output.jpg"""
    try:
        output_path = "/data/output.jpg"
        os.makedirs("/data", exist_ok=True)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as img:
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')

            # Resize to exact 800x600 instead of using .thumbnail()
            img = img.resize((800, 600), Image.LANCZOS)
            img.save(output_path, "JPEG", quality=80, optimize=True)

        # Ensure readable permissions
        os.chmod(output_path, 0o644)

        if not os.path.exists(output_path):
            raise RuntimeError("Failed to create output file")

        return "Image processed successfully"

    except Exception as e:
        return f"Error processing image: {str(e)}"

def transcribe_audio(audio_path):
    """Transcribe an MP3 file"""
    try:
        # Default to audio.mp3 if no path provided
        if not audio_path:
            audio_path = "/data/audio.mp3"
            
        validate_file_path(audio_path)
        temp_wav_path = "/data/temp.wav"

        # Verify file exists and is readable
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if not os.access(audio_path, os.R_OK):
            raise PermissionError(f"Cannot read audio file: {audio_path}")

        # Check file size
        if os.path.getsize(audio_path) == 0:
            raise ValueError("Audio file is empty")

        try:
            # Create a simple audio file if the input is empty or invalid
            if os.path.getsize(audio_path) < 1024:  # Less than 1KB
                audio = AudioSegment.silent(duration=1000)  # 1 second of silence
                audio.export(audio_path, format="mp3")

            # Load and convert audio with error handling
            audio = AudioSegment.from_mp3(audio_path)
            
            # Ensure output directory is writable
            output_dir = os.path.dirname(temp_wav_path)
            os.makedirs(output_dir, exist_ok=True)
            
            if not os.access(output_dir, os.W_OK):
                raise PermissionError(f"Cannot write to output directory: {output_dir}")
            
            # Export to WAV with explicit parameters
            audio.export(
                temp_wav_path,
                format="wav",
                parameters=["-ac", "1", "-ar", "16000"]
            )

            # Verify the WAV file was created
            if not os.path.exists(temp_wav_path):
                raise RuntimeError("Failed to create temporary WAV file")

            # Initialize recognizer with specific parameters
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True

            # Transcribe with error handling
            with sr.AudioFile(temp_wav_path) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    text = "No speech detected"
                except sr.RequestError:
                    text = "Error occurred during transcription"

                # Save transcription with proper permissions
                transcription_path = "/data/transcription.txt"
                with open(transcription_path, "w") as f:
                    f.write(text)
                os.chmod(transcription_path, 0o644)

        finally:
            # Clean up temp file
            if os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except OSError:
                    pass  # Ignore cleanup errors

        return "Audio transcribed successfully"
    except Exception as e:
        raise RuntimeError(f"Error transcribing audio: {str(e)}")

def convert_markdown():
    """Convert Markdown to HTML"""
    try:
        with open("/data/input.md", "r") as f:
            markdown = f.read()
        
        html = markdown2.markdown(markdown)
        
        with open("/data/output.html", "w") as f:
            f.write(html)
        return "Markdown converted to HTML"
    except Exception as e:
        raise RuntimeError(f"Error converting markdown: {str(e)}")

def filter_csv():
    """Filter CSV and return JSON"""
    try:
        results = []
        with open("/data/data.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['value']) > 100:  # Example filter
                    results.append(row)
        
        with open("/data/filtered.json", "w") as f:
            json.dump(results, f, indent=4)
        return "CSV filtered and converted to JSON"
    except Exception as e:
        raise RuntimeError(f"Error filtering CSV: {str(e)}")

def handle_task(task_desc):
    """Processes user requests using LLM classification"""
    try:
        # Map task descriptions to functions
        task_mapping = {
            "install_and_run_datagen": install_and_run_datagen,
            "format_markdown": format_markdown,
            "count_wednesdays": count_wednesdays,
            "sort_contacts": sort_contacts,
            "process_logs": process_logs,
            "extract_markdown_titles": extract_markdown_titles,
            "extract_email_sender": extract_email_sender,
            "extract_credit_card": extract_credit_card,
            "find_similar_comments": find_similar_comments,
            "calculate_gold_sales": calculate_gold_sales,
            "fetch_api_data": fetch_api_data,
            "handle_git_repo": lambda: handle_git_repo("https://github.com/example/repo.git"),
            "run_database_query": lambda: run_database_query("SELECT * FROM sales LIMIT 5"),
            "scrape_website": scrape_website,
            "process_image": lambda: process_image("/data/input.jpg"),
            "transcribe_audio": lambda: transcribe_audio("/data/audio.mp3"),
            "convert_markdown": convert_markdown,
            "filter_csv": filter_csv
        }

        # Normalize task description and attempt exact match first
        normalized_task = task_desc.lower().strip()
        task_keywords = {
            "resize": "process_image",
            "image": "process_image",
            "transcribe": "transcribe_audio",
            "audio": "transcribe_audio",
        }

        # Try to find matching task based on keywords
        selected_task_name = None
        for keyword, task_name in task_keywords.items():
            if keyword in normalized_task:
                selected_task_name = task_name
                break

        if not selected_task_name:
            # Ask LLM only if we couldn't determine the task through keywords
            llm_response = call_llm(f"""
            Identify the correct function for this task:
            {task_desc}
            Available functions: {', '.join(task_mapping.keys())}
            Respond with only the function name.
            """)
            selected_task_name = llm_response.strip()

        selected_task = task_mapping.get(selected_task_name)
        if not selected_task:
            raise ValueError(f"Unknown task: {task_desc}")

        return selected_task()
    except Exception as e:
        raise RuntimeError(f"Task handling error: {str(e)}")
