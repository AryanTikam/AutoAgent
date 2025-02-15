import os
import json
import csv
import sqlite3
import random
import datetime
from faker import Faker
from PIL import Image, ImageDraw
from pydub.generators import Sine
from pydub import AudioSegment

# Set the data directory
DATA_DIR = "/data"
os.makedirs(DATA_DIR, exist_ok=True)

fake = Faker()

# 📅 **Generate dates.txt (Random Dates)**
with open(os.path.join(DATA_DIR, "dates.txt"), "w") as f:
    for _ in range(100):
        f.write(fake.date() + "\n")

# 📋 **Generate contacts.json (Random Contacts)**
contacts = [{"first_name": fake.first_name(), "last_name": fake.last_name(), "email": fake.email()} for _ in range(50)]
with open(os.path.join(DATA_DIR, "contacts.json"), "w") as f:
    json.dump(contacts, f, indent=4)

# 📜 **Generate markdown file**
markdown_content = """# Sample Title
This is a sample markdown file with multiple headings.

## Heading 1
Some content.

## Heading 2
More content.
"""
with open(os.path.join(DATA_DIR, "input.md"), "w") as f:
    f.write(markdown_content)

# 📝 **Generate comments.txt (Random Comments)**
with open(os.path.join(DATA_DIR, "comments.txt"), "w") as f:
    for _ in range(30):
        f.write(fake.sentence() + "\n")

# 📧 **Generate email.txt**
email_content = f"""From: {fake.email()}
To: {fake.email()}
Subject: Test Email

Hello, this is a sample email.
"""
with open(os.path.join(DATA_DIR, "email.txt"), "w") as f:
    f.write(email_content)

# 📊 **Generate CSV file**
csv_path = os.path.join(DATA_DIR, "data.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "value"])
    writer.writeheader()
    for _ in range(20):
        writer.writerow({"name": fake.name(), "value": random.randint(50, 200)})

# 📸 **Generate input.jpg (Random Image)**
img = Image.new("RGB", (800, 600), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
draw = ImageDraw.Draw(img)
draw.text((10, 10), "Sample Image", fill="white")
img.save(os.path.join(DATA_DIR, "input.jpg"))

# 🎵 **Generate audio.mp3 (Silent 3s Audio)**
sine_wave = Sine(440).to_audio_segment(duration=3000)
sine_wave.export(os.path.join(DATA_DIR, "audio.mp3"), format="mp3")

# 🎟️ **Generate SQLite database (ticket-sales.db)**
db_path = os.path.join(DATA_DIR, "ticket-sales.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS tickets (type TEXT, units INTEGER, price REAL)")
ticket_types = ["Gold", "Silver", "Bronze"]
for _ in range(50):
    cursor.execute("INSERT INTO tickets VALUES (?, ?, ?)", (random.choice(ticket_types), random.randint(1, 5), round(random.uniform(50, 500), 2)))
conn.commit()
conn.close()

print("✅ Test data generated successfully in /data/")
