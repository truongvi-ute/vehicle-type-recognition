import { Cpu } from "lucide-react";

function ModelSelector({ models = [], selectedModel, onChange, disabled }) {
  // Group the models by backbone architecture for display
  const groupedModels = models.reduce((acc, model) => {
    let groupName = "Khác";
    const val = model.value.toLowerCase();
    if (val.startsWith("resnet50")) {
      groupName = "Kiến trúc ResNet-50 (CNN)";
    } else if (val.startsWith("yolo")) {
      groupName = "Kiến trúc YOLOv8-cls (Siêu nhanh)";
    } else if (val.startsWith("vit")) {
      groupName = "Kiến trúc ViT-B/16 (Vision Transformer)";
    }
    
    if (!acc[groupName]) {
      acc[groupName] = [];
    }
    acc[groupName].push(model);
    return acc;
  }, {});

  return (
    <section className="panelBlock">
      <div className="panelHeader">
        <Cpu size={18} />
        <h2>Chọn Mô hình (Model)</h2>
      </div>

      <div className="selectorForm">
        <div className="selectorField">
          <label htmlFor="model-select-dropdown">Danh sách mô hình khả dụng ({models.length})</label>
          <select
            id="model-select-dropdown"
            value={selectedModel}
            onChange={(e) => onChange(e.target.value)}
            disabled={disabled}
            className="modelSelect"
          >
            {Object.entries(groupedModels).map(([groupName, items]) => (
              <optgroup label={groupName} key={groupName}>
                {items.map((item) => (
                  <option key={item.value} value={item.value} title={item.description}>
                    {item.label}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
          <p className="modelHelpText" style={{ fontSize: "12px", color: "#647269", marginTop: "6px", marginBottom: 0 }}>
            Mỗi mô hình được cấu hình sẵn theo Backbone, tập dữ liệu huấn luyện và phiên bản tăng cường tương ứng.
          </p>
        </div>
      </div>
    </section>
  );
}

export default ModelSelector;
