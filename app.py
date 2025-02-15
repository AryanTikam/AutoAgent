from flask import Flask, request, jsonify
import os
from task_handler import handle_task
from utils import validate_file_path
import mimetypes

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask API is running!", 200

@app.route('/run', methods=['POST'])
def run_task():
    task_desc = request.args.get('task')
    if not task_desc:
        return jsonify({"error": "Task description is required"}), 400

    try:
        result = handle_task(task_desc)
        return jsonify({"message": "Task completed", "result": result}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/read', methods=['GET'])
def read_file():
    try:
        file_path = request.args.get('path')
        if not file_path:
            return jsonify({"error": "File path is required"}), 400

        validated_path = validate_file_path(file_path)

        if not os.path.exists(validated_path):
            return jsonify({"error": "File not found"}), 404

        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(validated_path)

        # Read as binary if it's an image or unknown format
        read_mode = "rb" if mime_type and mime_type.startswith("image") else "r"

        with open(validated_path, read_mode) as file:
            file_content = file.read()

        # Return binary data if it's an image
        if read_mode == "rb":
            return file_content, 200, {'Content-Type': mime_type or 'application/octet-stream'}

        return file_content, 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)