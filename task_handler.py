import os
import json
import sqlite3
from datetime import datetime
import glob
from utils import call_llm, validate_file_path
import subprocess
import base64

def install_and_run_datagen():
    """Install uv and run datagen.py"""
    try:
        # Install uv
        os.system("pip install uv --quiet")
        
        # Run datagen
        os.system("python3 -m uv https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py user@example.com")
        return "Datagen executed successfully"
    except Exception as e:
        raise RuntimeError(f"Error running datagen: {str(e)}")

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
            
        # Sort by modification time, most recent first
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

def handle_task(task_desc):
    """Main task handler function"""
    try:
        # Call the LLM to interpret the task
        llm_response = call_llm(f"Classify this task into one of these categories: install_and_run_datagen, count_wednesdays, sort_contacts, process_logs, extract_markdown_titles. Task: {task_desc}")
        
        # Task mapping
        task_functions = {
            'install_and_run_datagen': install_and_run_datagen,
            'count_wednesdays': count_wednesdays,
            'sort_contacts': sort_contacts,
            'process_logs': process_logs,
            'extract_markdown_titles': extract_markdown_titles
        }
        
        # Find matching task
        selected_task = None
        for task_name in task_functions.keys():
            if task_name.lower() in llm_response.lower():
                selected_task = task_name
                break
        
        if selected_task:
            result = task_functions[selected_task]()
            return result
        else:
            raise ValueError(f"Task not recognized: {task_desc}")
            
    except Exception as e:
        raise RuntimeError(f"Task handling error: {str(e)}")