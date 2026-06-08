# Code Audit Report

Audit goal: compare the current codebase against the latest Markdown architecture, especially [ARCHITECTURE_FINAL.md](ARCHITECTURE_FINAL.md) and [DEPLOYMENT.md](DEPLOYMENT.md).

Scope reviewed:

- `app.py`
- `requirements.txt`
- `src/data_prep.py`
- `src/dataset.py`
- `src/train.py`
- `src/model.py`

No code was changed during this audit.

## 1. Code Files That Diverge From The New Documentation

### `app.py`

Status: significantly outdated.

Main differences:

- Still implements a Streamlit application instead of Flask + React.
- Uses `streamlit run app.py` in the header instructions (`app.py:15`).
- Imports and uses Streamlit directly (`app.py:40`, many `st.*` calls throughout).
- Displays Top-K/Top-5 predictions in the Streamlit UI, not a Flask JSON API (`app.py:10`, `app.py:426`, `app.py:457`).
- Footer still says "Built with PyTorch & Streamlit" (`app.py:1185`).
- Supports only ResNet-50 and ViT in the UI mapping, not YOLO-cls (`app.py:90-99`).

Expected by docs:

- Flask backend REST API.
- React frontend.
- JSON response with top-3 predictions, confidence scores, model name, and processing time.

### `src/data_prep.py`

Status: significantly outdated.

Main differences:

- Split ratio uses `TRAIN_RATIO = 0.80`, not documented `85%` train (`src/data_prep.py:51`).
- Validation copy is merged into `valid/`, not separated as `valid_traincopy/` (`src/data_prep.py:306-308`, `src/data_prep.py:595`).
- Does not create the documented `valid_unseen/` and `valid_traincopy/` directories.
- Base Pipeline still uses morphology (`src/data_prep.py:89`, `src/data_prep.py:106-108`).
- Offline augmentation still uses 4 physical outputs per image (`_base`, `_night`, `_rain`, `_sun`) (`src/data_prep.py:542`, `src/data_prep.py:559-562`, `src/data_prep.py:587-592`).
- Still documents and logs the old `71,170 -> 284,680` strategy (`src/data_prep.py:592`).
- Final log still says `x4 pipelines` (`src/data_prep.py:818`).

Expected by docs:

- Independent split: `train` 85%, `valid_unseen` 5%, `test` 10%.
- Auxiliary `valid_traincopy` copied from train and kept separate.
- Base Pipeline = resize with aspect-ratio preservation + zero padding only.
- Quota-based offline augmentation with environment buckets `normal/rain/sun/night`.

### `src/dataset.py`

Status: outdated folder assumptions.

Main differences:

- Expects `train/valid/test`, not `train/valid_unseen/valid_traincopy/test` (`src/dataset.py:181-183`, `src/dataset.py:211`).
- Loads validation from `data_dir/valid` (`src/dataset.py:242-244`).
- Docstring still says train has physical augmentation `x4 pipelines` (`src/dataset.py:9`, `src/dataset.py:181`).
- Does not expose separate loaders for `valid_unseen` and `valid_traincopy`.

Expected by docs:

- Primary validation signal should be `valid_unseen`.
- `valid_traincopy` should be optional/auxiliary and reported separately.
- Dataset root should support `data/augmented/valid_unseen/<class>/` and `data/augmented/valid_traincopy/<class>/`.

### `src/train.py`

Status: partially aligned, but still tied to old validation loader.

Main differences:

- Uses `create_dataloaders`, which currently returns `train_loader`, `valid_loader`, `test_loader` based on old `valid/` folder.
- Early stopping is based on generic `valid`, not explicitly `valid_unseen`.
- Test loader is created but not used for final official test reporting in the training script.
- Only supports ResNet-50 and ViT through `MODEL_NAME_MAP`; YOLO-cls is documented as a project model family but has no training path here (`src/train.py:78-81`).
- CLI help still describes `train/valid/test` layout (`src/train.py:617`).

Expected by docs:

- Early stopping should prioritize `valid_unseen`.
- `valid_traincopy` should be auxiliary and separately reported if used.
- Official test evaluation should be a separate final evaluation step.
- YOLO-cls should either have a documented code path or remain explicitly unsupported in code until implemented.

### `src/model.py`

Status: model support is narrower than docs.

Main differences:

- Supports only `resnet50` and `vit_base_patch16_224` (`src/model.py:47`).
- Does not support YOLO-cls, although docs list YOLO-cls as a model family.

Expected by docs:

- ResNet-50, Vision Transformer, and YOLO-cls are the documented model families.
- If YOLO-cls is not implemented yet, code/docs should clearly mark it as pending or provide a separate training/inference path.

### `requirements.txt`

Status: mixed old/new deployment dependencies.

Main differences:

- Includes Flask (`requirements.txt:19`), which aligns with the new backend direction.
- Still includes Streamlit (`requirements.txt:85`), which is now legacy/prototype only.
- There is no `backend/requirements.txt` yet.
- There is no `frontend/package.json` yet.

Expected by docs:

- Official deployment structure should include `backend/requirements.txt` and `frontend/package.json`.
- Root Streamlit dependency should be moved to a legacy/prototype dependency set if kept.

## 2. Parts Still Using Old Streamlit

Files:

- `app.py`
- `requirements.txt`

Evidence:

- `app.py` header says to run `streamlit run app.py` (`app.py:15`).
- `app.py` imports Streamlit (`app.py:40`).
- Streamlit UI primitives are used throughout: `st.set_page_config`, `st.markdown`, `st.sidebar`, `st.file_uploader`, `st.plotly_chart`, `st.tabs`, etc.
- Footer says "Built with PyTorch & Streamlit" (`app.py:1185`).
- `requirements.txt` still includes `streamlit==1.57.0` (`requirements.txt:85`).

Docs say:

- Streamlit is no longer the main deployment path.
- Flask + React is the official deployment direction.
- A root-level Streamlit `app.py`, if retained, is legacy/prototype only.

## 3. Augmentation Still Following Old `train x 4 pipelines`

File:

- `src/data_prep.py`

Evidence:

- `_augment_one_image` produces four files per image: `_base`, `_night`, `_rain`, `_sun` (`src/data_prep.py:542`, `src/data_prep.py:559-562`).
- `augment_data` describes every balanced train image producing four physical files (`src/data_prep.py:587-592`).
- Old hard count remains: `71,170 -> 284,680` (`src/data_prep.py:592`).
- Log message says `balanced x 4` (`src/data_prep.py:661`).
- Final output summary still says `train/ (x4 pipelines)` (`src/data_prep.py:818`).

Docs say:

- Offline augmentation must be quota-based.
- Target default is about 7,000 images/class.
- Environment distribution should be 70% `normal`, 10% `rain`, 10% `sun`, 10% `night`.
- Large classes such as `boat` and `car` should not be over-generated.
- Minority classes should use geometric/content transforms plus environment simulation to fill quotas.

## 4. Base Pipeline Still Uses Morphology

File:

- `src/data_prep.py`

Evidence:

- Docstring says Base Pipeline applies Morphological Closing and Opening (`src/data_prep.py:89`).
- Code creates a morphology kernel (`src/data_prep.py:106`).
- Code applies `cv2.MORPH_CLOSE` and `cv2.MORPH_OPEN` (`src/data_prep.py:107-108`).

Additional deployment mismatch:

- `app.py` inference transform uses `v2.Resize((224, 224))`, which distorts aspect ratio instead of zero-padding (`app.py:378-386`).

Docs say:

- Base Pipeline must only resize while preserving aspect ratio and zero-pad to 224x224.
- No additional cleanup filter or image transformation belongs to the Base Pipeline.

## 5. Split / Validation / Test Issues

### Current split logic

File:

- `src/data_prep.py`

Evidence:

- `TRAIN_RATIO = 0.80`, but docs require 85% train (`src/data_prep.py:51`).
- `VALID_RATIO = 0.05` and `VALID_COPY_RATIO = 0.05` exist (`src/data_prep.py:52`, `src/data_prep.py:55`).
- Split docstring says:
  - `train/<class>` = 80%.
  - `valid/<class>` = 5% unseen + 5% copy from train.
  - `test/<class>` = 10%.
  (`src/data_prep.py:306-308`)
- Valid copy is mixed into `valid/`, not separated into `valid_traincopy/`.

Docs require:

- `test`: 10% independent.
- `valid_unseen`: 5% independent.
- `valid_traincopy`: 5% copied from train, auxiliary only, separate from `valid_unseen`.
- Independent original split must be 85/5/10.

### Current dataset loader logic

File:

- `src/dataset.py`

Evidence:

- Expects `train/<class>`, `valid/<class>`, and `test/<class>` (`src/dataset.py:181-183`).
- Checks only `("train", "valid", "test")`.
- Loads validation from `data_dir/valid`.

Docs require:

- `data/augmented/train/<class>/...`
- `data/augmented/valid_unseen/<class>/`
- `data/augmented/valid_traincopy/<class>/`
- `data/augmented/test/<class>/`

### Current training logic

File:

- `src/train.py`

Evidence:

- Uses only a generic `valid_loader` returned by `create_dataloaders`.
- Early stopping is based on generic `val_loss`.
- Does not distinguish `valid_unseen` from `valid_traincopy`.

Docs require:

- Early stopping should prioritize `valid_unseen`.
- `valid_traincopy` should be auxiliary and separate.
- Official evaluation should prioritize `test`.

## 6. Work Needed For Flask + React Deployment

Current state:

- No `backend/` directory.
- No `frontend/` directory.
- No `backend/app.py`.
- No `backend/routes/`, `backend/services/`, or `backend/utils/`.
- No `backend/requirements.txt`.
- No `frontend/package.json`, `frontend/src/`, or `frontend/public/`.
- Root `app.py` is Streamlit and should become legacy/prototype only.

Backend work needed:

- Create `backend/app.py` as Flask entry point.
- Create REST routes, likely:
  - `GET /health`
  - `GET /models`
  - `POST /predict`
- Move reusable inference logic out of Streamlit UI into backend services:
  - model loading
  - checkpoint discovery
  - Base Pipeline preprocessing
  - inference
  - top-3 formatting
  - optional processing-time measurement
- Return JSON in the documented shape:
  - model name
  - processing time if available
  - top-3 classes
  - confidence scores
  - preprocessed image if included
- Ensure Base Pipeline preserves aspect ratio and zero-pads.

Frontend work needed:

- Create React project under `frontend/`.
- Add model selector.
- Add image upload and original preview.
- Call Flask `POST /predict`.
- Render optional preprocessed image returned by backend.
- Render top-3 predictions.
- Render confidence chart.
- Handle loading/error states.

Dependency work needed:

- Split backend dependencies into `backend/requirements.txt`.
- Add Flask CORS support if React dev server runs on a different port.
- Move Streamlit dependency into a legacy/prototype requirements file if keeping root `app.py`.
- Add `frontend/package.json`.

## 7. Recommended Safe Fix Order

1. **Freeze docs/code contract**
   - Keep [ARCHITECTURE_FINAL.md](ARCHITECTURE_FINAL.md) as the target.
   - Decide whether YOLO-cls is implemented now or explicitly pending.

2. **Fix Base Pipeline first**
   - Remove morphology from `src/data_prep.py`.
   - Implement reusable aspect-ratio resize + zero padding.
   - Reuse the same logic for future Flask inference.

3. **Fix split structure**
   - Change `TRAIN_RATIO` to 0.85.
   - Create separate folders:
     - `train/`
     - `valid_unseen/`
     - `valid_traincopy/`
     - `test/`
   - Ensure `valid_traincopy` is copied from train and not mixed into `valid_unseen`.

4. **Update dataset loaders**
   - Make `src/dataset.py` load `valid_unseen` as primary validation.
   - Add optional `valid_traincopy` loader or separate helper.
   - Preserve test loader for final evaluation.

5. **Update training**
   - Use `valid_unseen` for early stopping.
   - Record `valid_traincopy` only as auxiliary if configured.
   - Add or document a separate final test evaluation command.

6. **Replace offline augmentation**
   - Remove the forced four-output-per-image strategy.
   - Add quota-based bucket filling.
   - Implement `normal/rain/sun/night` folder structure.
   - Add large-class cap/bucket policy.
   - Add minority-class transforms before environment bucket filling.

7. **Extract inference services**
   - Move model loading, preprocessing, and prediction from Streamlit `app.py` into reusable functions.
   - Ensure output is top-3 by default.

8. **Build Flask backend**
   - Create `backend/` structure.
   - Implement `/health`, `/models`, and `/predict`.
   - Return documented JSON.

9. **Build React frontend**
   - Create `frontend/` structure.
   - Implement model selection, upload, preview, predictions, and confidence chart.

10. **Mark Streamlit as legacy**
   - Keep root `app.py` only if useful as a prototype.
   - Remove Streamlit from primary requirements.
   - Update any code comments or README references after code migration.

11. **End-to-end verification**
   - Regenerate a small dataset sample through data prep.
   - Train or smoke-test loaders.
   - Run Flask prediction on one image.
   - Run React UI against Flask.
   - Confirm output matches the documented flow:

```text
React UI -> Flask API -> Base Pipeline preprocessing -> Model inference -> JSON response -> React visualization
```
