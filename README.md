# Vehicle Type Recognition

VehicleTypeRecognition is an image classification project for 10 vehicle classes using ResNet-50, Vision Transformer, and YOLO-cls.

The project architecture is defined in [docs/ARCHITECTURE_FINAL.md](docs/ARCHITECTURE_FINAL.md). That file is the Single Source of Truth for dataset policy, split strategy, augmentation, training, evaluation, deployment, and project structure.

## Dataset

Vehicle-10 contains **36,006 RGB images** in 10 classes:

| Class | Data group |
|-------|------------|
| `bicycle` | Minority |
| `boat` | Large |
| `bus` | Medium |
| `car` | Large |
| `helicopter` | Minority |
| `minibus` | Minority |
| `motorcycle` | Medium |
| `taxi` | Minority |
| `train` | Minority |
| `truck` | Medium |

## Split Strategy

Independent original split:

| Split | Ratio | Approx. count | Role |
|-------|-------|---------------|------|
| `train` | 85% | ~30,605 | Training source |
| `valid_unseen` | 5% | ~1,800 | Independent validation |
| `test` | 10% | ~3,601 | Final evaluation |

Auxiliary validation:

- `valid_traincopy`: copied from `train`, about the same working scale as validation unless configured otherwise.
- It is not an independent split and does not change the 85/5/10 split math.
- It exists only for instructor-required reference checks.

Official reporting priority:

1. `test`
2. `valid_unseen`
3. `valid_traincopy` only as auxiliary reference

## Data Pipeline

Base Pipeline:

1. Resize while preserving aspect ratio.
2. Zero-pad to **224x224**.

Class balancing and offline augmentation are applied only to `train`.

Default class quota:

- About **7,000 images/class**.
- Final dataset size depends on configured class quotas.

Default environment quota:

| Bucket | Ratio | Example for 7,000 images/class |
|--------|-------|--------------------------------|
| `normal` | 70% | 4,900 |
| `rain` | 10% | 700 |
| `sun` | 10% | 700 |
| `night` | 10% | 700 |

Large classes such as `boat` and `car` are not over-generated. Minority classes such as `helicopter`, `taxi`, `train`, `bicycle`, and `minibus` use additional transforms to fill quotas.

## Project Structure

```text
VehicleTypeRecognition/
├── backend/
│   ├── app.py
│   ├── routes/
│   ├── services/
│   ├── utils/
│   └── requirements.txt
├── frontend/
│   ├── package.json
│   ├── src/
│   └── public/
├── models/
├── outputs/
├── data/
│   ├── raw/<class>/
│   ├── splits/
│   │   ├── train/<class>/
│   │   ├── valid_unseen/<class>/
│   │   ├── valid_traincopy/<class>/
│   │   └── test/<class>/
│   ├── balanced/<class>/
│   └── augmented/
│       ├── train/<class>/
│       │   ├── normal/
│       │   ├── rain/
│       │   ├── sun/
│       │   └── night/
│       ├── valid_unseen/<class>/
│       ├── valid_traincopy/<class>/
│       └── test/<class>/
├── src/
├── docs/
├── requirements.txt
└── README.md
```

`data/augmented/` is the training-ready dataset root. Only `data/augmented/train/` contains offline-augmented images; validation and test folders are Base Pipeline outputs without augmentation.

`backend/app.py` is the official Flask API entry point. A root-level Streamlit `app.py`, if still present, is legacy/prototype only and is not the production demo.

## Quick Start

```bash
pip install -r requirements.txt
```

```bash
python src/data_split.py --raw_dir data/raw --output_dir data/splits --seed 42
python src/balance_classes.py --input_dir data/splits/train --output_dir data/balanced
python src/augment_offline.py --input_dir data/balanced --output_dir data/augmented
```

```bash
python src/train.py --model resnet50 --data_dir data/augmented --max_epochs 100 --patience 10
python src/train_vit.py --model vit_base --data_dir data/augmented --max_epochs 100 --patience 10
python src/train_yolo.py --model yolov8n-cls --data_dir data/augmented --epochs 100 --patience 10
```

## Deployment Demo

Flask backend:

```bash
cd backend
pip install -r requirements.txt
python app.py
```

React frontend:

```bash
cd frontend
npm install
npm run dev
```

Deployment flow:

```text
React UI -> Flask API -> Base Pipeline preprocessing -> Model inference -> JSON response -> React visualization
```

For detailed training hyperparameters, see [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md).

## Model Roles

| Model | Role |
|-------|------|
| ResNet-50 | Stable CNN baseline |
| Vision Transformer | Global-context comparison model |
| YOLO-cls | Fast inference comparison model |

## Documentation

- [docs/ARCHITECTURE_FINAL.md](docs/ARCHITECTURE_FINAL.md): Single Source of Truth.
- [docs/DATASET_INFO.md](docs/DATASET_INFO.md): Dataset overview.
- [docs/DATA_STRATEGY.md](docs/DATA_STRATEGY.md): Split, balancing, and leakage rules.
- [docs/AUGMENTATION.md](docs/AUGMENTATION.md): Offline and online augmentation.
- [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md): Training and evaluation.
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): Flask + React deployment.
- [docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md): Model comparison.
- [docs/AUDIT_REPORT.md](docs/AUDIT_REPORT.md): Documentation audit status.
