import { SlidersHorizontal } from "lucide-react";

function PipelineSelector({ pipelines, selectedPipeline, onChange, disabled }) {
  return (
    <section className="panelBlock">
      <div className="panelHeader">
        <SlidersHorizontal size={18} />
        <h2>Pipeline</h2>
      </div>

      <div className="pipelineGrid" role="radiogroup" aria-label="Input pipeline selector">
        {pipelines.map((pipeline) => (
          <button
            key={pipeline.value}
            className={`pipelineOption ${selectedPipeline === pipeline.value ? "active" : ""}`}
            type="button"
            onClick={() => onChange(pipeline.value)}
            disabled={disabled}
            role="radio"
            aria-checked={selectedPipeline === pipeline.value}
            title={pipeline.description}
          >
            {pipeline.label}
          </button>
        ))}
      </div>
    </section>
  );
}

export default PipelineSelector;
