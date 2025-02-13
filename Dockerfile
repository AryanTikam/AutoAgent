# Use Python 3.12.3 as base image
FROM python:3.12.3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create data directory and ensure proper permissions
RUN mkdir -p /data /data/logs /data/docs && \
    chmod 777 /data /data/logs /data/docs

# Generate initial test data
COPY setup_data.py .
RUN python setup_data.py

# Expose the port the app runs on
EXPOSE 8000

# Command to run the app
CMD ["python", "app.py"]