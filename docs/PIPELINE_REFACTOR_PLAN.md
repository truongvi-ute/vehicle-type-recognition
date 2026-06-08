# Ke hoach tach 4 pipeline tien xu ly anh

## Muc tieu

Tach logic xu ly anh thanh 4 pipeline rieng de dung chung cho:

- Offline augmentation khi tao du lieu train.
- Backend Flask khi preview pipeline va demo tren web.

Huong uu tien: lay hanh vi hien tai cua backend lam chuan cho 4 pipeline, sau do cap nhat `src/augment_offline.py` de su dung lai cung logic nay. Cach nay giup anh hien thi tren web va anh sinh ra khi augment train khong bi lech cong thuc.

## Hien trang

Backend hien co logic pipeline trong:

- `backend/utils/preprocessing.py`

Bao gom:

- `normal`: base pipeline, resize giu ti le va zero-pad ve 224x224.
- `rain`: them vach mua, giam sang, lam mo nhe.
- `sun`: them flare/vung sang, tang sang.
- `night`: giam sang theo gamma, chinh kenh mau, them noise.

Offline augmentation hien co logic rieng trong:

- `src/augment_offline.py`

Van co 4 nhom `normal`, `rain`, `sun`, `night`, nhung cong thuc khac backend. Vi vay hien tai web demo va train augmentation chua dung chung mot pipeline.

## Cau truc de xuat

Tao package pipeline dung chung:

```text
src/image_pipelines/
  __init__.py
  base.py
  rain.py
  sun.py
  night.py
  registry.py
```

Vai tro tung file:

- `base.py`: resize giu ti le, zero-pad 224x224.
- `rain.py`: pipeline mua, lay logic tu backend hien tai.
- `sun.py`: pipeline nang, lay logic tu backend hien tai.
- `night.py`: pipeline dem, lay logic tu backend hien tai.
- `registry.py`: danh sach pipeline ho tro va ham `apply_environment_pipeline`.

## Chuan du lieu noi bo

Nen chon mot chuan noi bo duy nhat de tranh sai mau:

- Input/output noi bo: `PIL.Image.Image` o mode `RGB`.
- Backend dang dung PIL nen gan nhu khong can doi.
- `src/augment_offline.py` dang dung OpenCV nen se chuyen anh tu `cv2 BGR` sang `PIL RGB` truoc khi apply pipeline, sau do chuyen nguoc ve `cv2 BGR` khi ghi file.

Ly do chon PIL/RGB:

- Giong backend hien tai.
- De tao preview base64 cho web.
- Giam nguy co lech mau do BGR/RGB khi dung chung voi frontend/backend.

## API de xuat

Trong `src/image_pipelines/base.py`:

```python
TARGET_SIZE = 224

def apply_base_pipeline(image: Image.Image, target_size: int = TARGET_SIZE) -> Image.Image:
    ...
```

Trong tung file moi truong:

```python
def apply_rain_pipeline(image: Image.Image, seed: int | None = 42) -> Image.Image:
    ...

def apply_sun_pipeline(image: Image.Image, seed: int | None = 42) -> Image.Image:
    ...

def apply_night_pipeline(image: Image.Image, seed: int | None = 42) -> Image.Image:
    ...
```

Trong `src/image_pipelines/registry.py`:

```python
SUPPORTED_PIPELINES = ["normal", "rain", "sun", "night"]

def apply_environment_pipeline(
    image: Image.Image,
    pipeline: str | None,
    seed: int | None = 42,
) -> tuple[Image.Image, str]:
    ...
```

## Cach dung trong backend

Cap nhat `backend/utils/preprocessing.py`:

- Giu lai phan doc file upload, normalize tensor, encode data URL.
- Xoa/cat bot logic rieng cua `apply_base_pipeline`, `apply_rain_pipeline`, `apply_sun_pipeline`, `apply_night_pipeline`.
- Import pipeline dung chung tu `src/image_pipelines`.

Vi du:

```python
from src.image_pipelines.base import apply_base_pipeline
from src.image_pipelines.registry import apply_environment_pipeline, SUPPORTED_PIPELINES
```

Ket qua mong muon:

- `/api/preprocess` dung pipeline chung de render preview.
- `/api/predict` dung pipeline chung neu van cho chon `normal/rain/sun/night`.
- Neu muon bam sat kien truc chinh thuc, `/api/predict` co the ep `normal`, con `/api/preprocess` van cho preview 4 pipeline.

## Cach dung trong offline augmentation

Cap nhat `src/augment_offline.py`:

- Bo cac ham rieng: `apply_rain`, `apply_sun`, `apply_night`, `apply_environment`.
- Import pipeline chung.
- Khi doc anh bang OpenCV:
  - `cv2.imread` tra ve BGR.
  - Chuyen sang RGB/PIL.
  - Apply base + environment pipeline chung.
  - Chuyen ve BGR de `cv2.imwrite`.

Luong xu ly de xuat:

```text
cv2.imread -> BGR numpy
BGR -> RGB PIL
apply_base_pipeline
optional geometric augmentation
apply_base_pipeline lai neu geometric lam bien dang
apply_environment_pipeline
RGB PIL -> BGR numpy
cv2.imwrite
```

Ghi chu: geometric augmentation hien dang dung OpenCV trong `src/augment_offline.py`. Co the giu lai rieng cho train vi day la augmentation bo sung chi dung de fill quota, khong phai pipeline moi truong web demo.

## Chinh sach inference

Co 2 lua chon can quyet dinh truoc khi sua code:

### Lua chon A: Demo robustness

Web cho predict voi ca 4 pipeline:

- `normal`
- `rain`
- `sun`
- `night`

Uu diem:

- Nguoi dung thay duoc model phan ung ra sao voi tung dieu kien anh.
- Preview va inference dung cung anh da xu ly.

Nhuoc diem:

- Khac voi tai lieu kien truc hien tai, vi deployment chinh thuc dang yeu cau Base Pipeline only.

### Lua chon B: Deployment chuan

Web van cho preview 4 pipeline, nhung predict chi dung `normal`.

Uu diem:

- Bam sat `docs/ARCHITECTURE_FINAL.md`.
- Prediction dai dien cho inference/evaluation chuan.

Nhuoc diem:

- Neu nguoi dung chon `rain/sun/night`, anh preview khac anh that su dua vao model. Can hien thi ro trang thai nay tren UI hoac an pipeline selector khoi predict.

Khuyen nghi: neu muc tieu la demo khoa hoc/bao cao ve augmentation, chon A va ghi ro day la "simulation mode". Neu muc tieu la deployment chuan, chon B.

## Thu tu thuc hien

1. Tao package `src/image_pipelines`.
2. Chuyen logic backend hien tai sang 4 file pipeline moi.
3. Cap nhat backend import pipeline chung.
4. Cap nhat `src/augment_offline.py` dung pipeline chung.
5. Chay smoke test cho pipeline:
   - Anh bat ky qua `normal` co kich thuoc 224x224.
   - Anh qua `rain/sun/night` co kich thuoc 224x224.
   - Tensor backend co shape `(1, 3, 224, 224)`.
6. Neu co san anh mau, tao preview 4 pipeline va so sanh truc quan.
7. Cap nhat tai lieu neu quyet dinh cho web predict voi `rain/sun/night`.

## Rui ro can luu y

- Sai mau do chuyen BGR/RGB khi offline augmentation ghi file.
- Ket qua train moi co the khac train cu vi cong thuc `rain/sun/night` doi theo backend.
- Seed co the lam preview lap lai qua moi lan, trong khi train nen co bien thien theo anh/index.
- Neu dung seed co dinh `42` cho moi anh, cac flare/noise co the qua giong nhau. Khi augment train nen truyen seed thay doi theo class, bucket, index hoac dung RNG rieng.

## Ket qua mong muon

Sau khi tach:

- Chỉ con mot bo cong thuc cho `normal/rain/sun/night`.
- Backend preview, backend predict va offline train augmentation co the dung chung logic.
- Viec bao tri de hon: sua pipeline mot lan, ca train va web demo cung thay doi.
