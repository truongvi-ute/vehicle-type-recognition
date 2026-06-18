import requests
import io
from PIL import Image

def test_api():
    # Create a simple red test image
    img = Image.new('RGB', (300, 300), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    pipelines = ["normal", "rain", "sun", "night", "gaussian", "motion", "unsharp"]
    
    print("Testing /api/preprocess endpoint:")
    for pipeline in pipelines:
        img_byte_arr.seek(0)
        files = {'image': ('test.jpg', img_byte_arr, 'image/jpeg')}
        data = {'pipeline': pipeline}
        
        response = requests.post("http://127.0.0.1:5000/api/preprocess", files=files, data=data)
        if response.status_code == 200:
            res = response.json()
            print(f"  - Pipeline '{pipeline}': SUCCESS. Returned pipeline: {res.get('pipeline')}")
        else:
            print(f"  - Pipeline '{pipeline}': FAILED. Status: {response.status_code}, Error: {response.text}")
            
    print("\nTesting /api/predict endpoint:")
    for pipeline in pipelines:
        img_byte_arr.seek(0)
        files = {'image': ('test.jpg', img_byte_arr, 'image/jpeg')}
        # Using resnet50_raw_original_v1_best as a dummy model if it exists
        data = {'pipeline': pipeline, 'model_name': 'resnet50_raw_original_v1_best'}
        
        response = requests.post("http://127.0.0.1:5000/api/predict", files=files, data=data)
        if response.status_code == 200:
            res = response.json()
            print(f"  - Pipeline '{pipeline}': SUCCESS. Model: {res.get('model_name')}, Pipeline: {res.get('pipeline')}, Top prediction: {res.get('predictions')[0] if res.get('predictions') else 'None'}")
        else:
            print(f"  - Pipeline '{pipeline}': FAILED. Status: {response.status_code}, Error: {response.text}")

if __name__ == "__main__":
    test_api()
