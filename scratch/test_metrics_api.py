import urllib.request
import json

BASE_URL = "http://127.0.0.1:5000/api"

def test_endpoint(path, params=None):
    url = f"{BASE_URL}{path}"
    if params:
        query = urllib.parse.urlencode(params)
        url = f"{url}?{query}"
    
    print(f"Testing GET {url}...")
    try:
        with urllib.request.urlopen(url) as response:
            status = response.status
            body = response.read().decode("utf-8")
            data = json.loads(body)
            print(f"  -> SUCCESS (Status {status})")
            return data
    except Exception as e:
        print(f"  -> FAILED: {e}")
        if hasattr(e, "read"):
            try:
                error_body = e.read().decode("utf-8")
                print(f"     Error response body: {error_body}")
            except Exception:
                pass
        return None

def run_tests():
    print("=== Testing Flask Metrics APIs ===")
    
    # 1. Health check
    test_endpoint("/health")
    
    # 2. Experiments Catalog
    catalog = test_endpoint("/metrics/experiments")
    
    # 3. Dataset Profiles
    print("\n--- Testing Dataset Profiles ---")
    test_endpoint("/metrics/dataset-profile", {"dataset": "raw_original", "version": "v1"})
    test_endpoint("/metrics/dataset-profile", {"dataset": "raw_original", "version": "v2"})
    test_endpoint("/metrics/dataset-profile", {"dataset": "raw_cleaning", "version": "v1"})
    test_endpoint("/metrics/dataset-profile", {"dataset": "raw_cleaning", "version": "v2"})
    
    # 4. Individual Experiment
    print("\n--- Testing Load Experiment ---")
    test_endpoint("/metrics/experiment", {"id": "resnet50__raw_original__v1"})
    test_endpoint("/metrics/experiment", {"id": "yolo__raw_original__v1"})
    test_endpoint("/metrics/experiment", {"id": "vit__raw_original__v1"})
    
    # 5. Compare Experiments
    print("\n--- Testing Compare Experiments ---")
    test_endpoint("/metrics/compare", {
        "first": "resnet50__raw_original__v1",
        "second": "resnet50__raw_original__v2"
    })

if __name__ == "__main__":
    run_tests()
