# Hướng dẫn Tích hợp Mô hình Phân loại YOLO

Tài liệu này mô tả chi tiết cách tích hợp chính xác mô hình YOLO-cls (cụ thể là các mô hình phân loại YOLOv8/v11 của Ultralytics) vào dự án VehicleTypeRecognition cho cả quá trình huấn luyện (training) và suy luận (inference). Tài liệu giải quyết các khoảng cách hiện tại trong mã nguồn giữa các mô hình PyTorch tiêu chuẩn (ResNet-50, ViT) và YOLO-cls.

---

## 1. Kiến trúc YOLO-cls và Các Yêu cầu Khác biệt

Không giống như ResNet-50 và Vision Transformer (được định nghĩa dưới dạng các class `nn.Module` PyTorch tiêu chuẩn trong `src/model.py`), các mô hình YOLO-cls từ Ultralytics được quản lý thông qua thư viện bậc cao `ultralytics`.

### Các điểm khác biệt chính

| Đặc trưng | PyTorch Tiêu chuẩn (ResNet-50, ViT) | YOLO-cls (Ultralytics) |
|---|---|---|
| **Định dạng Checkpoint** | `.pth` chỉ chứa `state_dict` | `.pt` chứa toàn bộ mô hình + siêu tham số |
| **Cơ chế Nạp Mô hình** | `torch.load` + `model.load_state_dict` | `from ultralytics import YOLO; YOLO("path.pt")` |
| **Đầu vào Suy luận** | Tensor PyTorch đã chuẩn hóa dạng `(1, 3, 224, 224)` | Ảnh PIL, mảng NumPy, đường dẫn file hoặc Tensor PyTorch |
| **Đầu ra Suy luận** | `logits` thô (cần đi qua hàm `softmax`) | Danh sách các đối tượng `Results` chứa thuộc tính `.probs` |
| **Quản lý Thiết bị (GPU/CPU)**| Thủ công bằng `.to(device)` | Tự động hoặc cấu hình qua `.to("cuda" / "cpu")` / tham số gọi |

---

## 2. Các Thay đổi Mã nguồn Khuyến nghị

Để hỗ trợ YOLO-cls chính xác mà không làm ảnh hưởng đến luồng xử lý hiện tại của ResNet-50 và ViT, chúng ta cần điều chỉnh trình nạp mô hình và dịch vụ suy luận ở backend.

### Bước 2.1: Cập nhật cơ chế tìm kiếm đường dẫn mô hình
Trong file [backend/services/model_loader.py](file:///d:/Xu%20Ly%20Anh%20So/ProjectNhom/vehicle-type-recognition/backend/services/model_loader.py), chúng ta cần cho phép tìm kiếm thêm các file định dạng `.pt` (sử dụng bởi YOLO) bên cạnh các file `.pth`.

Chỉnh sửa hàm `_candidate_paths` như sau:
```python
def _candidate_paths(models_dir: Path, model_name: Optional[str]) -> list[Path]:
    if model_name:
        raw = Path(model_name)
        names = [raw.name]
        # Cho phép cả hậu tố .pth và .pt
        if raw.suffix not in (".pth", ".pt"):
            names.extend([
                f"{raw.name}.pth", f"{raw.name}_best.pth",
                f"{raw.name}.pt", f"{raw.name}_best.pt"
            ])
        return [models_dir / name for name in names]

    # Quét cả checkpoint .pth và .pt
    paths = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt"))
    return sorted(
        paths,
        key=lambda path: (0 if path.name.endswith("_best.pth") or path.name.endswith("_best.pt") else 1, path.name.lower()),
    )
```

### Bước 2.2: Triển khai Trình nạp mô hình YOLO
Trong file [backend/services/model_loader.py](file:///d:/Xu%20Ly%20Anh%20So/ProjectNhom/vehicle-type-recognition/backend/services/model_loader.py), khi một checkpoint được xác định là YOLO (dựa trên đuôi file `.pt` hoặc tên mô hình chứa chữ `yolo`), chúng ta sẽ bỏ qua cơ chế nạp thông thường của PyTorch và sử dụng API của `ultralytics`.

```python
def get_model(models_dir: str | Path, model_name: Optional[str] = None) -> LoadedModel:
    model_path = resolve_model_path(models_dir, model_name)
    cache_key = str(model_path.resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Kiểm tra xem đây có phải là mô hình YOLO không
    is_yolo = model_path.suffix == ".pt" or "yolo" in model_path.name.lower()
    
    if is_yolo:
        from ultralytics import YOLO
        # Nạp mô hình YOLO (Ultralytics tự quản lý việc đưa lên thiết bị tính toán)
        yolo_model = YOLO(str(model_path))
        yolo_model.to(device)
        
        # Đóng gói mô hình vào cấu trúc LoadedModel để backend sử dụng thống nhất
        loaded = LoadedModel(
            name=model_path.stem,
            path=model_path,
            model=yolo_model, # Lưu đối tượng YOLO
            device=device,
        )
    else:
        # Nạp mô hình PyTorch tiêu chuẩn (ResNet, ViT)
        model = load_for_inference(
            checkpoint_path=str(model_path),
            num_classes=len(CLASS_NAMES),
            device=device,
        )
        loaded = LoadedModel(
            name=model_path.stem,
            path=model_path,
            model=model,
            device=device,
        )
        
    _MODEL_CACHE[cache_key] = loaded
    return loaded
```

### Bước 2.3: Cập nhật Luồng xử lý Dự đoán
Trong file [backend/services/inference_service.py](file:///d:/Xu%20Ly%20Anh%20So/ProjectNhom/vehicle-type-recognition/backend/services/inference_service.py), xử lý riêng biệt đối tượng YOLO trong quá trình lan truyền xuôi (forward pass):

```python
    started = time.perf_counter()
    loaded = get_model(models_dir=models_dir, model_name=model_name)
    
    # 1. Kiểm tra mô hình có phải là thực thể YOLO hay không
    from ultralytics import YOLO
    is_yolo = isinstance(loaded.model, YOLO)

    # 2. Tiền xử lý ảnh đầu vào
    # Áp dụng Base Pipeline: thay đổi kích thước giữ tỷ lệ + đệm viền đen về 224x224
    image_tensor, processed_image, selected_pipeline = preprocess_image(
        image_stream,
        pipeline=pipeline,
    )
    
    predictions: List[Dict[str, object]] = []
    
    if is_yolo:
        # YOLO chấp nhận ảnh PIL trực tiếp và tự xử lý kích thước bên trong.
        # Tuy nhiên, để đảm bảo tính nhất quán của Base Pipeline, chúng ta truyền ảnh PIL đã qua xử lý.
        results = loaded.model(processed_image, verbose=False)
        probs = results[0].probs
        
        # Trích xuất chỉ số phân loại và độ tự tin của Top-K kết quả
        topk_conf = probs.top5conf.tolist() if hasattr(probs, 'top5conf') else []
        topk_idx = probs.top5.tolist() if hasattr(probs, 'top5') else []
        
        # Phương án dự phòng nếu không lấy được top5conf trực tiếp
        if not topk_idx:
            probs_tensor = probs.data
            k = min(top_k, probs_tensor.numel())
            topk_conf, topk_idx = torch.topk(probs_tensor, k=k)
            topk_conf = topk_conf.tolist()
            topk_idx = topk_idx.tolist()
            
        for confidence, index in zip(topk_conf, topk_idx):
            predictions.append({
                "class_name": CLASS_NAMES[index],
                "confidence": round(float(confidence), 6),
            })
    else:
        # Xử lý forward pass cho ResNet-50 / ViT tiêu chuẩn
        image_tensor = image_tensor.to(loaded.device)
        with torch.no_grad():
            logits = loaded.model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        
        k = min(top_k, probabilities.numel(), len(CLASS_NAMES))
        confidences, indices = torch.topk(probabilities, k=k)
        
        for confidence, index in zip(confidences.tolist(), indices.tolist()):
            predictions.append({
                "class_name": CLASS_NAMES[index],
                "confidence": round(float(confidence), 6),
            })
            
    # Giới hạn số lượng trả về đúng bằng top_k
    predictions = predictions[:top_k]
```

---

## 3. Cấu hình Huấn luyện YOLO

Script [src/train_yolo.py](file:///d:/Xu%20Ly%20Anh%20So/ProjectNhom/vehicle-type-recognition/src/train_yolo.py) thực hiện bao bọc và ánh xạ cấu trúc dữ liệu phù hợp với định dạng đầu vào của YOLO:

1. **Bộ chuyển đổi thư mục dữ liệu (Dataset Directory Adapter):** YOLO yêu cầu thư mục dữ liệu trực tiếp chứa các nhánh `train/val/test`, bên trong là thư mục của từng lớp phương tiện. Script `train_yolo.py` tự động giải quyết việc này bằng cách tạo symlink liên kết các thư mục từ `data/augmented/` sang `outputs/yolo_dataset_adapter/`.
2. **Cấu hình thiết bị:** Thư viện Ultralytics tự động phát hiện GPU khả dụng. Bạn có thể thiết lập GPU cụ thể thông qua đối số:
   ```bash
   python src/train_yolo.py --model yolov8n-cls.pt --device 0
   ```
3. **Đầu ra trọng số (Weights Output):** Sau khi huấn luyện hoàn tất, trọng số tốt nhất được lưu tại:
   `outputs/yolo/yolo_cls/weights/best.pt`
   Để sử dụng mô hình này trên ứng dụng Flask, hãy tạo thư mục `models` (nếu chưa có) và sao chép file trọng số về đây với tên định dạng chuẩn:
   ```bash
   mkdir -p models
   cp outputs/yolo/yolo_cls/weights/best.pt models/yolo_cls_best.pt
   ```
