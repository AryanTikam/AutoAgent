import os
import json
from datetime import datetime, timedelta
import random

def setup_test_environment():
    """Creates necessary directories and test data files"""
    # Use absolute path for Docker
    base_dir = '/data'
    print(f"Creating test environment at: {base_dir}")
    
    # Create base data directory and subdirectories
    for dir_path in ['logs', 'docs']:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

    try:
        # Generate dates.txt with a mix of dates including Wednesdays
        start_date = datetime(2024, 1, 1)
        dates = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(100)]
        dates_path = os.path.join(base_dir, 'dates.txt')
        with open(dates_path, 'w') as f:
            f.write('\n'.join(dates))
        print(f"Created dates.txt at: {dates_path}")

        # Generate contacts.json
        contacts = [
            {"first_name": "John", "last_name": "Doe", "email": "john@example.com"},
            {"first_name": "Alice", "last_name": "Smith", "email": "alice@example.com"},
            {"first_name": "Bob", "last_name": "Johnson", "email": "bob@example.com"},
            {"first_name": "Emma", "last_name": "Brown", "email": "emma@example.com"}
        ]
        contacts_path = os.path.join(base_dir, 'contacts.json')
        with open(contacts_path, 'w') as f:
            json.dump(contacts, f, indent=2)
        print(f"Created contacts.json at: {contacts_path}")

        # Generate sample log files
        for i in range(15):
            timestamp = datetime.now() - timedelta(hours=i)
            log_content = f"[{timestamp.isoformat()}] Sample log entry #{i}\nAdditional log details\n"
            log_path = os.path.join(base_dir, 'logs', f'app_{i}.log')
            with open(log_path, 'w') as f:
                f.write(log_content)
        print(f"Created 15 log files in: {os.path.join(base_dir, 'logs')}")

        # Generate sample markdown files
        md_content = """# Sample Document
## Introduction
This is a test markdown file.
"""
        md_path = os.path.join(base_dir, 'docs', 'sample1.md')
        with open(md_path, 'w') as f:
            f.write(md_content)
        print(f"Created markdown file at: {md_path}")

        # Verify file existence and permissions
        print("\nVerifying created files:")
        print(f"dates.txt exists: {os.path.exists(dates_path)}")
        print(f"contacts.json exists: {os.path.exists(contacts_path)}")
        print(f"logs directory has files: {len(os.listdir(os.path.join(base_dir, 'logs')))}")
        print(f"docs directory has files: {len(os.listdir(os.path.join(base_dir, 'docs')))}")

    except Exception as e:
        print(f"Error during setup: {str(e)}")
        raise

if __name__ == "__main__":
    setup_test_environment()