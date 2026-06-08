# Deployment Guide

This document describes the official Flask + React deployment direction. The authoritative design is [ARCHITECTURE_FINAL.md](ARCHITECTURE_FINAL.md).

Streamlit is no longer the primary deployment path. If a legacy Streamlit `app.py` still exists, it is treated only as a prototype, not the production demo.

## Run the Demo

Backend:

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Deployment Flow

```text
React UI -> Flask API -> Base Pipeline preprocessing -> Model inference -> JSON response -> React visualization
```

## Backend Responsibilities

The backend uses Flask as a REST API server.

Flask responsibilities:

- Load trained model checkpoints from `models/`.
- Receive uploaded images from the React frontend.
- Apply the Base Pipeline exactly as used in evaluation.
- Run inference with the selected model.
- Return JSON containing top-3 predicted classes, confidence scores, model name, and processing time when available.

Example JSON shape:

```json
{
  "model_name": "resnet50",
  "processing_time_ms": 42.5,
  "predictions": [
    {"class": "car", "confidence": 0.91},
    {"class": "taxi", "confidence": 0.06},
    {"class": "minibus", "confidence": 0.02}
  ],
  "preprocessed_image": "optional-base64-or-url"
}
```

## Frontend Responsibilities

The frontend uses React.

React responsibilities:

- Let users choose a model.
- Upload an image.
- Preview the original image.
- Display the preprocessed image if the backend returns it.
- Display top-3 predictions.
- Display a confidence chart.
- Provide a more flexible UI than the legacy Streamlit prototype.

## Deployment Preprocessing

Deployment input must use the same Base Pipeline as evaluation:

1. Resize while preserving aspect ratio.
2. Zero-pad to **224x224**.

No offline or online augmentation is applied during inference.

Demo predictions are inference outputs, not official evaluation metrics.

## Proposed Deployment Structure

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
├── src/
└── docs/
```

## Metric Reporting Policy

Official model quality must be reported from:

1. `test`
2. `valid_unseen`

`valid_traincopy` is auxiliary only and must not be used as final deployment-quality evidence.

## Common Checks

- Model weights should exist in `models/`.
- Class names should match the 10 Vehicle-10 classes.
- Flask preprocessing should match the Base Pipeline.
- React should visualize the JSON response without changing inference results.
