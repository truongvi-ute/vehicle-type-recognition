import { useState, useRef, useEffect } from "react";
import { Cpu, ChevronDown, Check, Zap, Sparkles } from "lucide-react";

const ARCH_ICONS = {
  resnet: Cpu,
  yolo: Zap,
  vit: Sparkles,
};

function ModelSelector({ models = [], selectedModel, onChange, disabled }) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const selectedItem = models.find((m) => m.value === selectedModel) || models[0];

  // Group models by backbone architecture for display
  const groupedModels = models.reduce((acc, model) => {
    let groupKey = "other";
    let groupTitle = "Kiến trúc khác";
    const val = model.value.toLowerCase();
    
    if (val.startsWith("resnet50")) {
      groupKey = "resnet";
      groupTitle = "Kiến trúc ResNet-50 (CNN)";
    } else if (val.startsWith("yolo")) {
      groupKey = "yolo";
      groupTitle = "Kiến trúc YOLOv8-cls (Siêu nhanh)";
    } else if (val.startsWith("vit")) {
      groupKey = "vit";
      groupTitle = "Kiến trúc ViT-B/16 (Vision Transformer)";
    }
    
    if (!acc[groupKey]) {
      acc[groupKey] = { title: groupTitle, items: [] };
    }
    acc[groupKey].items.push(model);
    return acc;
  }, {});

  const handleSelect = (val) => {
    onChange(val);
    setIsOpen(false);
  };

  const getArchIcon = (value) => {
    const val = value.toLowerCase();
    if (val.startsWith("resnet50")) return Cpu;
    if (val.startsWith("yolo")) return Zap;
    if (val.startsWith("vit")) return Sparkles;
    return Cpu;
  };

  const SelectedIcon = getArchIcon(selectedItem.value);

  return (
    <section className="panelBlock">
      <div className="panelHeader">
        <Cpu size={18} />
        <h2>Chọn Mô hình (Model)</h2>
      </div>

      <div className="customDropdownContainer" ref={dropdownRef}>
        <label className="dropdownLabel">Danh sách mô hình khả dụng ({models.length})</label>
        
        <button
          type="button"
          className={`customDropdownTrigger ${isOpen ? "active" : ""}`}
          onClick={() => !disabled && setIsOpen(!isOpen)}
          disabled={disabled}
        >
          <div className="selectedInfo">
            <SelectedIcon size={18} className="modelIcon" />
            <div className="selectedTexts">
              <span className="selectedLabel">{selectedItem.label}</span>
              <span className="selectedDesc">{selectedItem.description}</span>
            </div>
          </div>
          <ChevronDown size={18} className={`chevronIcon ${isOpen ? "open" : ""}`} />
        </button>

        {isOpen && (
          <div className="customDropdownMenu">
            {Object.entries(groupedModels).map(([groupKey, group]) => {
              const GroupIcon = ARCH_ICONS[groupKey] || Cpu;
              return (
                <div key={groupKey} className="dropdownGroup">
                  <div className="dropdownGroupHeader">
                    <GroupIcon size={14} />
                    <span>{group.title}</span>
                  </div>
                  <div className="dropdownGroupItems">
                    {group.items.map((item) => {
                      const isSelected = item.value === selectedModel;
                      return (
                        <button
                          key={item.value}
                          type="button"
                          className={`dropdownItem ${isSelected ? "selected" : ""}`}
                          onClick={() => handleSelect(item.value)}
                        >
                          <div className="itemTexts">
                            <span className="itemLabel">{item.label}</span>
                            <span className="itemDesc">{item.description}</span>
                          </div>
                          {isSelected && <Check size={16} className="checkIcon" />}
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}

export default ModelSelector;
