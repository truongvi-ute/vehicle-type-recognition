import { SlidersHorizontal, Image, CloudRain, Sun, Moon, Sparkles, Wind, Zap, AlertCircle } from "lucide-react";

const ICON_MAP = {
  Image,
  CloudRain,
  Sun,
  Moon,
  Sparkles,
  Wind,
  Zap
};

function PipelineSelector({ selectedPipeline, onChange, disabled, selectedModel }) {
  // Determine model versions based on its name convention
  const isV1Model = selectedModel ? selectedModel.includes("_v1_") : false;
  const isV2Model = selectedModel ? (selectedModel.includes("_v2_") || selectedModel.includes("_v3_")) : false;

  const groups = [
    {
      title: "Mặc định",
      version: "both",
      items: [
        {
          label: "Normal",
          value: "normal",
          description: "Không áp dụng thuật toán tiền xử lý",
          icon: "Image",
        }
      ]
    },
    {
      title: "Giả lập Thời tiết (Weather - V1)",
      version: "v1",
      items: [
        {
          label: "Rain",
          value: "rain",
          description: "Giả lập hiệu ứng trời mưa",
          icon: "CloudRain",
        },
        {
          label: "Sun",
          value: "sun",
          description: "Giả lập chói nắng / flare",
          icon: "Sun",
        },
        {
          label: "Night",
          value: "night",
          description: "Giả lập ánh sáng ban đêm",
          icon: "Moon",
        },
      ]
    },
    {
      title: "Làm mờ & Làm nét (Blur/Sharpen - V2)",
      version: "v2",
      items: [
        {
          label: "Gaussian Blur",
          value: "gaussian",
          description: "Làm mờ Gaussian (Nhiễu mờ)",
          icon: "Sparkles",
        },
        {
          label: "Motion Blur",
          value: "motion",
          description: "Làm mờ chuyển động (Motion blur)",
          icon: "Wind",
        },
        {
          label: "Unsharp Masking",
          value: "unsharp",
          description: "Làm sắc nét ảnh (Unsharp mask)",
          icon: "Zap",
        },
      ]
    }
  ];

  return (
    <section className="panelBlock">
      <div className="panelHeader">
        <SlidersHorizontal size={18} />
        <h2>Thuật toán Tiền xử lý</h2>
      </div>

      <div className="pipelineGroupsContainer">
        {groups.map((group) => {
          const isCompatible = group.version === "both" || 
            (group.version === "v1" && isV1Model) || 
            (group.version === "v2" && isV2Model);

          return (
            <div key={group.title} className={`pipelineGroupSection ${!isCompatible ? "incompatibleGroup" : ""}`}>
              <div className="pipelineGroupTitle">
                <span>{group.title}</span>
                {!isCompatible && (
                  <span className="incompatibleBadge" title="Nhóm thuật toán này không thuộc tập huấn luyện của mô hình hiện tại. Kết quả dự đoán có thể bị ảnh hưởng.">
                    <AlertCircle size={12} />
                    Lệch tập Train
                  </span>
                )}
              </div>

              <div className="pipelineGrid" role="radiogroup" aria-label={group.title}>
                {group.items.map((item) => {
                  const IconComponent = ICON_MAP[item.icon] || Image;
                  return (
                    <button
                      key={item.value}
                      className={`pipelineOption ${selectedPipeline === item.value ? "active" : ""} ${!isCompatible ? "incompatibleOption" : ""}`}
                      type="button"
                      onClick={() => onChange(item.value)}
                      disabled={disabled || !isCompatible}
                      role="radio"
                      aria-checked={selectedPipeline === item.value}
                      title={item.description}
                    >
                      <IconComponent size={14} className="pipelineOptionIcon" />
                      <span>{item.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default PipelineSelector;
