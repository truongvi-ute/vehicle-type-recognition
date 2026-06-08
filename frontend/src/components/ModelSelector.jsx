import { Cpu } from "lucide-react";

function ModelSelector({ models, selectedModel, onChange, disabled }) {
  return (
    <section className="panelBlock">
      <div className="panelHeader">
        <Cpu size={18} />
        <h2>Model</h2>
      </div>

      <div className="modelGrid" role="radiogroup" aria-label="Model selector">
        {models.map((model) => (
          <button
            key={model.value}
            className={`modelOption ${selectedModel === model.value ? "active" : ""}`}
            type="button"
            onClick={() => onChange(model.value)}
            disabled={disabled}
            role="radio"
            aria-checked={selectedModel === model.value}
          >
            <strong>{model.label}</strong>
            <span>{model.description}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

export default ModelSelector;
