import os
import json
from datetime import datetime, timedelta
import random

def setup_test_environment():
    """Creates necessary directories and test data files"""
    # Create base data directory
    os.makedirs('/data/logs', exist_ok=True)
    os.makedirs('/data/docs', exist_ok=True)

    # Generate dates.txt with a mix of dates including Wednesdays
    start_date = datetime(2024, 1, 1)
    dates = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(100)]
    with open('/data/dates.txt', 'w') as f:
        f.write('\n'.join(dates))

    # Generate contacts.json
    contacts = [
        {"first_name": "John", "last_name": "Doe", "email": "john@example.com"},
        {"first_name": "Alice", "last_name": "Smith", "email": "alice@example.com"},
        {"first_name": "Bob", "last_name": "Johnson", "email": "bob@example.com"},
        {"first_name": "Emma", "last_name": "Brown", "email": "emma@example.com"}
    ]
    with open('/data/contacts.json', 'w') as f:
        json.dump(contacts, f, indent=2)

    # Generate sample log files
    for i in range(15):
        timestamp = datetime.now() - timedelta(hours=i)
        log_content = f"[{timestamp.isoformat()}] Sample log entry #{i}\nAdditional log details\n"
        with open(f'/data/logs/app_{i}.log', 'w') as f:
            f.write(log_content)

    # Generate sample markdown files
    md_content = """# Sample Document
## Introduction
This is a test markdown file.
"""
    with open('/data/docs/sample1.md', 'w') as f:
        f.write(md_content)

if __name__ == "__main__":
    setup_test_environment()