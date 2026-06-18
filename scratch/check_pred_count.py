import requests
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition")
url = 'http://127.0.0.1:5000/api/predict'

# Find any jpg image in data/raw to test
image_path = None
for p in (PROJECT_ROOT / "data").rglob("*.jpg"):
    if p.is_file():
        image_path = p
        break

if not image_path:
    # fallback to sample.jpg
    image_path = PROJECT_ROOT / "src" / "image_pipelines" / "sample.jpg"

print(f"Testing prediction on image: {image_path}")
files = {'image': open(image_path, 'rb')}
data = {'model_name': 'yolo_raw_original_v1', 'pipeline': 'normal'}

r = requests.post(url, files=files, data=data)
print('Status code:', r.status_code)
if r.status_code == 200:
    preds = r.json().get('predictions', [])
    print('Number of predictions returned:', len(preds))
    for i, p in enumerate(preds):
        print(f"  #{i+1} {p['class_name']}: {p['confidence'] * 100:.2f}%")
else:
    print('Error response:', r.text)
