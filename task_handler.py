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

def install_and_run_datagen(task_desc):
    """Install uv (if required), install dependencies, and run datagen.py with user email"""
    try:
        # Extract the email from the task description
        words = task_desc.split()
        user_email = next((word for word in words if "@" in word), "user@example.com")  # Default email

        # ✅ Ensure uv is installed
        subprocess.run(["pip", "install", "--quiet", "uv"], check=True)

        # ✅ Download datagen.py
        script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
        script_path = "/data/datagen.py"
        subprocess.run(["curl", "-s", "-o", script_path, script_url], check=True)

        # ✅ Install missing dependencies (Faker and others)
        subprocess.run(["pip", "install", "--quiet", "-r", script_path], check=False)  # Ignore errors if no requirements file

        # ✅ Make the script executable
        subprocess.run(["chmod", "+x", script_path], check=True)

        # ✅ Run the script using python3
        result = subprocess.run(["python3", script_path, user_email], 
                                capture_output=True, text=True, check=True)

        return f"Datagen executed successfully for {user_email}: {result.stdout}"

    except subprocess.CalledProcessError as e:
        return f"Error running datagen: {e.stderr}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def format_markdown():
    """Format markdown file using prettier"""
    try:
        subprocess.run(["npm", "install", "-g", "prettier@3.4.2"], check=True)
        subprocess.run(["prettier", "--write", "/data/format.md"], check=True)
        return "Markdown formatted successfully"
    except Exception as e:
        raise RuntimeError(f"Error formatting markdown: {str(e)}")

def count_days(task_desc):
    """Count the number of a specific weekday in a date file, supporting multiple formats & languages."""
    try:
        # **1️⃣ Extract the required day from the task description**
        prompt = f"""
        Extract the weekday (e.g., Monday, Tuesday, ...) from this task description:
        "{task_desc}"
        Respond with only the English name of the day.
        """
        day_name = call_llm(prompt).strip().capitalize()

        # **2️⃣ Extract file paths from the task description**
        prompt_paths = f"""
        Extract the source and destination file paths from this task description:
        "{task_desc}"
        Respond in the format: source=source_path, destination=destination_path
        """
        path_response = call_llm(prompt_paths).strip()
        try:
            source_path, dest_path = path_response.split(", ")
            source_path = source_path.replace("source=", "").strip()
            dest_path = dest_path.replace("destination=", "").strip()
        except:
            raise ValueError("Failed to extract file paths from task description")

        # **3️⃣ Validate paths**
        if not source_path.startswith("/data/") or not dest_path.startswith("/data/"):
            raise ValueError("Access outside /data/ is not allowed")

        # **4️⃣ Read the date file**
        with open(source_path, "r") as file:
            dates = [line.strip() for line in file]

        # **5️⃣ Supported date formats**
        valid_formats = [
            "%Y-%m-%d",        # 2024-03-14
            "%d-%b-%Y",        # 14-Mar-2024
            "%b %d, %Y",       # Mar 14, 2024
            "%Y/%m/%d %H:%M:%S"  # 2024/03/14 15:30:45
        ]

        # **6️⃣ Parse each date and count occurrences of the extracted weekday**
        def parse_date(date_str):
            for fmt in valid_formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unknown date format: {date_str}")

        day_index = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_name)
        count = sum(1 for date in dates if parse_date(date).weekday() == day_index)

        # **7️⃣ Write the result to the specified destination file**
        with open(dest_path, "w") as file:
            file.write(str(count))

        return f"Counted {count} occurrences of {day_name} in {source_path} and wrote to {dest_path}"

    except ValueError as e:
        raise RuntimeError(f"Error counting days: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")

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
    """Extract credit card number from image using OCR."""
    try:
        possible_paths = ["/data/credit-card.png", "/data/credit_card.png"]
        image_path = next((path for path in possible_paths if os.path.exists(path)), None)

        if not image_path:
            raise FileNotFoundError(f"Credit card image not found in: {possible_paths}")

        if not os.access(image_path, os.R_OK):
            raise PermissionError(f"Cannot read credit card image: {image_path}")

        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            raise RuntimeError(f"Error opening image file: {str(e)}")

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        prompt = f"Extract only the credit card number from this image (base64): {encoded_string}"
        card_number = call_llm(prompt).strip()

        card_number = "".join(filter(str.isdigit, card_number))
        if not (13 <= len(card_number) <= 19):
            raise ValueError(f"Invalid credit card number extracted: {card_number}")

        output_path = "/data/credit_card.txt"
        with open(output_path, "w") as file:
            file.write(card_number)

        return f"Credit card number extracted successfully and saved to {output_path}"

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
def fetch_api_data(task_desc):
    """Fetch data from a given API and save it to a specified file."""
    try:
        # 🔍 Extract API URL and output file
        prompt = f"""
        Extract only the API URL and output file path from this task description: {task_desc}.
        Respond strictly in the format: <api_url>, <output_path>.
        """
        response = call_llm(prompt).strip()

        if "," not in response:
            raise ValueError("Could not extract API URL and output file path from task description")

        api_url, output_path = map(str.strip, response.split(","))

        # Validate paths
        validate_file_path(output_path, allow_write=True)

        # Fetch data from API
        api_response = requests.get(api_url)
        api_response.raise_for_status()

        # Save data
        with open(output_path, "w") as f:
            json.dump(api_response.json(), f, indent=4)

        return f"API data fetched from {api_url} and saved to {output_path}"
    except Exception as e:
        raise RuntimeError(f"Error fetching API data: {str(e)}")

def handle_git_repo(task_desc):
    """Clone a GitHub repo and make a commit dynamically."""
    try:
        # 🔍 Extract repo URL and file to modify
        prompt = f"""
        Extract only the GitHub repository URL and the file to modify from this task description: {task_desc}.
        Respond in the format: <repo_url>, <file_to_modify>. If the file is not mentioned, return "README.md".
        Do not include any additional text.
        """
        response = call_llm(prompt).strip()

        if "," in response:
            repo_url, file_to_modify = map(str.strip, response.split(","))
        else:
            repo_url = response.strip()
            file_to_modify = "README.md"

        # Ensure URL starts with https://github.com/
        if not repo_url.startswith("https://github.com/"):
            raise ValueError(f"Invalid GitHub URL extracted: {repo_url}")

        repo_path = "/data/repo"

        # Clone repo if not already present
        if not os.path.exists(repo_path):
            Repo.clone_from(repo_url, repo_path)

        # Modify file
        file_path = os.path.join(repo_path, file_to_modify)
        with open(file_path, "a") as f:
            f.write("\nUpdated by automation")

        # Commit changes
        repo = Repo(repo_path)
        repo.git.add(file_to_modify)
        repo.index.commit("Automated update")

        return f"Git repository {repo_url} updated successfully, modified {file_to_modify}"
    except Exception as e:
        raise RuntimeError(f"Error handling git repo: {str(e)}")

def run_database_query(task_desc):
    """Run a SQL query dynamically while ensuring security and correct extraction."""
    try:
        # 🔍 Extract database path and SQL query
        prompt = f"""
        Extract only the database file path and the SQL query from this task description: {task_desc}.
        Respond strictly in the format: <database_path>, <SQL_query>. Do not add any extra text.
        """
        response = call_llm(prompt).strip()

        # Ensure response is correctly formatted
        if "," not in response:
            raise ValueError(f"Could not extract database path and SQL query from task description: {response}")

        db_path, query = map(str.strip, response.split(",", 1))

        # **Validate database path**
        if not db_path.startswith("/data/") or not db_path.endswith(".db"):
            raise ValueError(f"Invalid database path extracted: {db_path}")

        # **Ensure database file exists**
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")

        # **Prevent unsafe queries**
        forbidden_keywords = ["DELETE", "DROP", "ALTER"]
        if any(keyword in query.upper() for keyword in forbidden_keywords):
            raise ValueError("Query contains forbidden operations")

        # **Execute SQL query**
        conn = duckdb.connect(db_path)
        result = conn.execute(query).fetchall()
        conn.close()

        # **Save result**
        output_path = db_path.replace(".db", "-query-results.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)

        return f"SQL query executed on {db_path} and results saved to {output_path}"

    except Exception as e:
        raise RuntimeError(f"Error running database query: {str(e)}")

def scrape_website(task_desc):
    """Scrape a website and extract links."""
    try:
        # 🔍 Extract website URL and output file
        prompt = f"""
        Extract only the website URL and output file path from this task description: {task_desc}.
        Respond strictly in the format: <website_url>, <output_path>. Do not add any extra text.
        """
        response = call_llm(prompt).strip()

        if "," not in response:
            raise ValueError(f"Could not extract website URL and output file path: {response}")

        website_url, output_path = map(str.strip, response.split(","))

        # **Validate URL format**
        if not website_url.startswith("http"):
            raise ValueError(f"Invalid website URL extracted: {website_url}")

        # **Validate file path**
        validate_file_path(output_path, allow_write=True)

        # **Scrape website**
        response = requests.get(website_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # **Extract links, filtering out None values**
        links = [a.get('href') for a in soup.find_all('a') if a.get('href')]

        # **Save links**
        with open(output_path, "w") as f:
            f.write("\n".join(links))

        return f"Website {website_url} scraped and links saved to {output_path}"
    except Exception as e:
        raise RuntimeError(f"Error scraping website: {str(e)}")

def process_image(task_desc):
    """Resize or compress an image dynamically."""
    try:
        # 🔍 Extract input image path, output path, and dimensions
        prompt = f"""
        Extract only the input image path, output image path, and dimensions from this task description: {task_desc}.
        Respond strictly in the format: <input_path>, <output_path>, <width>x<height>.
        """
        response = call_llm(prompt).strip()

        if "," not in response:
            raise ValueError("Could not extract input image path, output path, and dimensions")

        input_path, output_path, dimensions = map(str.strip, response.split(","))
        width, height = map(int, dimensions.split("x"))

        # Validate paths
        validate_file_path(input_path)
        validate_file_path(output_path, allow_write=True)

        # Process image
        with Image.open(input_path) as img:
            img = img.resize((width, height), Image.LANCZOS)
            img.save(output_path, "JPEG", quality=80, optimize=True)

        return f"Image {input_path} resized to {width}x{height} and saved to {output_path}"
    except Exception as e:
        return f"Error processing image: {str(e)}"

def transcribe_audio(task_desc):
    """Transcribe an MP3 file dynamically."""
    try:
        # 🔍 Extract input audio path and output transcription file path
        prompt = f"""
        Extract only the input audio file path and output transcription file path from this task description: {task_desc}.
        Respond strictly in the format: <audio_path>, <output_path>.
        """
        response = call_llm(prompt).strip()

        if "," not in response:
            raise ValueError(f"Could not extract input audio path and output file path: {response}")

        audio_path, output_path = map(str.strip, response.split(","))

        # Validate paths
        validate_file_path(audio_path)
        validate_file_path(output_path, allow_write=True)

        # Check if file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Convert to WAV
        temp_wav_path = audio_path.replace(".mp3", ".wav")
        audio = AudioSegment.from_mp3(audio_path)
        audio.export(temp_wav_path, format="wav")

        # Transcribe
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                text = "No speech detected"
            except sr.RequestError:
                text = "Error occurred during transcription"

        # Save transcription
        with open(output_path, "w") as f:
            f.write(text)

        return f"Audio {audio_path} transcribed and saved to {output_path}"
    except Exception as e:
        raise RuntimeError(f"Error transcribing audio: {str(e)}")

def convert_markdown(task_desc):
    """Convert Markdown to HTML dynamically."""
    try:
        # 🔍 Extract input Markdown file and output HTML file
        prompt = f"""
        Extract only the input Markdown file path and output HTML file path from this task description: {task_desc}.
        Respond strictly in the format: <input_md_path>, <output_html_path>.
        """
        response = call_llm(prompt).strip()

        if "," not in response:
            raise ValueError("Could not extract input and output file paths")

        md_path, html_path = map(str.strip, response.split(","))

        # Validate paths
        validate_file_path(md_path)
        validate_file_path(html_path, allow_write=True)

        # Convert Markdown to HTML
        with open(md_path, "r") as f:
            html = markdown2.markdown(f.read())

        with open(html_path, "w") as f:
            f.write(html)

        return f"Markdown {md_path} converted to HTML and saved to {html_path}"
    except Exception as e:
        raise RuntimeError(f"Error converting markdown: {str(e)}")

def filter_csv(task_desc):
    """Filter a CSV file based on a condition and save the results to a JSON file dynamically."""
    try:
        # 🔍 Extract CSV file path, output file path, column name, and threshold value
        prompt = f"""
        Extract the input CSV file path, output JSON file path, column name, and threshold value from this task: {task_desc}.
        Respond strictly in the format: <csv_path>, <json_path>, <column_name>, <threshold>.
        """
        response = call_llm(prompt).strip()

        if "," not in response or len(response.split(",")) != 4:
            raise ValueError("Could not extract CSV file path, output file path, column name, and threshold value")

        csv_path, json_path, column_name, threshold = map(str.strip, response.split(","))
        threshold = float(threshold)

        # Validate paths
        validate_file_path(csv_path)
        validate_file_path(json_path, allow_write=True)

        # Read and filter CSV
        results = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if column_name in row and float(row[column_name]) > threshold:
                    results.append(row)

        # Save results to JSON
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)

        return f"Filtered data from {csv_path} on column '{column_name}' with threshold {threshold} and saved to {json_path}"
    except Exception as e:
        raise RuntimeError(f"Error filtering CSV: {str(e)}")

def extract_repo_url(task_desc):
    """Extracts the GitHub repository URL from the task description"""
    words = task_desc.split()
    for word in words:
        if word.startswith("https://github.com/"):
            return word
    return "https://github.com/example/repo.git"  # Default repo

def extract_sql_query(task_desc):
    """Extracts the SQL query from the task description"""
    query_start = task_desc.lower().find("run sql query")
    if query_start != -1:
        return task_desc[query_start + len("run sql query "):]
    return "SELECT * FROM sales LIMIT 5"  # Default query

def handle_task(task_desc):
    """Processes user requests using LLM classification"""
    try:
        # Map task descriptions to functions
        task_mapping = {
            "install_and_run_datagen": lambda task_desc: install_and_run_datagen(task_desc),
            "format_markdown": lambda task_desc: format_markdown(),
            "count_days": lambda task_desc: count_days(task_desc),
            "sort_contacts": lambda task_desc: sort_contacts(),
            "process_logs": lambda task_desc: process_logs(),
            "extract_markdown_titles": lambda task_desc: extract_markdown_titles(),
            "extract_email_sender": lambda task_desc: extract_email_sender(),
            "extract_credit_card": lambda task_desc: extract_credit_card(),
            "find_similar_comments": lambda task_desc: find_similar_comments(),
            "calculate_gold_sales": lambda task_desc: calculate_gold_sales(),
            "fetch_api_data": lambda task_desc: fetch_api_data(task_desc),
            "handle_git_repo": lambda task_desc: handle_git_repo(extract_repo_url(task_desc)),
            "run_database_query": lambda task_desc: run_database_query(extract_sql_query(task_desc)),
            "scrape_website": lambda task_desc: scrape_website(task_desc),
            "process_image": lambda task_desc: process_image(task_desc),
            "transcribe_audio": lambda task_desc: transcribe_audio(task_desc),
            "convert_markdown": lambda task_desc: convert_markdown(task_desc),
            "filter_csv": lambda task_desc: filter_csv(task_desc)
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

        return selected_task(task_desc) if "task_desc" in selected_task.__code__.co_varnames else selected_task()

    except Exception as e:
        raise RuntimeError(f"Task handling error: {str(e)}")
