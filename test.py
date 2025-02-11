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
            "name": "Count Wednesdays",
            "task": "Count the number of Wednesdays in /data/dates.txt",
            "output": "/data/dates-wednesdays.txt"
        },
        {
            "name": "Sort Contacts",
            "task": "Sort the contacts in /data/contacts.json by last name",
            "output": "/data/contacts-sorted.json"
        },
        {
            "name": "Process Logs",
            "task": "Get first lines of 10 most recent logs in /data/logs/",
            "output": "/data/logs-recent.txt"
        }
    ]
    
    results = []
    for test in tests:
        success = test_single_task(base_url, test["task"], test["output"])
        results.append({
            "test": test["name"],
            "success": success
        })
    
    # Print summary
    print("\n=== Test Summary ===")
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"{status} {result['test']}")

if __name__ == "__main__":
    run_tests()