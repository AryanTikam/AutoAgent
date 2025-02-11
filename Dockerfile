# Use the official Python image as the base
FROM python:3.12.3

# Set the working directory in the container
WORKDIR /app

# Copy all files from the project directory to /app in the container
COPY . /app

# Create the /data directory and set permissions
RUN mkdir -p /data/logs /data/docs && chmod -R 755 /data

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for Flask API
EXPOSE 8000

# Run the Flask app with unbuffered output
CMD ["python", "-u", "app.py"]
