import requests
import json
import time

def test_single_task(base_url, task, output_path=None, expected_status=200):
    """Test a single task and its output"""
    print(f"\nTesting task: {task}")
    
    # Run task
    try:
        response = requests.post(f"{base_url}/run", params={"task": task})
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code != expected_status:
            print(f"❌ Task failed with status {response.status_code}")
            return False
            
        # Check output file if specified
        if output_path:
            time.sleep(1)  # Give time for file operations to complete
            read_response = requests.get(f"{base_url}/read", params={"path": output_path})
            print(f"Output file status: {read_response.status_code}")
            if read_response.status_code == 200:
                print(f"Output content: {read_response.text[:100]}...")
                return True
            else:
                print(f"❌ Failed to read output file")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        return False

def run_tests():
    base_url = "http://localhost:8000"
    
    tests = [
        {
            "name": "Clone Git Repo & Commit",
            "task": "Clone the repo and make a commit",
            "output": None
        },
        {
            "name": "Run Secure SQL Query",
            "task": "Run SQL query on /data/analytics.db",
            "output": "/data/query-results.json"
        },
        {
            "name": "Compress & Resize Image",
            "task": "Resize image to 800x600",
            "output": "/data/output.jpg"
        },
        {
            "name": "Transcribe MP3 Audio",
            "task": "Transcribe audio file to text",
            "output": "/data/transcription.txt"
        }
    ]

    for test in tests:
        success = test_single_task(
            base_url, 
            test["task"], 
            test.get("output"),
            test.get("expected_status", 200)
        )
        print(f"✅ {test['name']}" if success else f"❌ {test['name']} failed")

if __name__ == "__main__":
    run_tests()