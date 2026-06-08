import { BarChart3, Clock, Gauge } from "lucide-react";

function formatConfidence(value) {
  return `${Math.round(Number(value || 0) * 1000) / 10}%`;
}

function PredictionResult({ result, pipelinePreview, isLoading, isPreviewLoading }) {
  const predictions = result?.predictions || [];
  const preview = result?.processed_image ? result : pipelinePreview;

  return (
    <section className="resultPanel">
      <div className="resultHeader">
        <div>
          <h2>Prediction</h2>
          <p>{result ? result.model_name : "Waiting for image"}</p>
        </div>
        <BarChart3 size={22} />
      </div>

      <div className="processedPreview">
        <div className="processedMeta">
          <strong>Processed image</strong>
          <span>
            {isPreviewLoading
              ? "Rendering pipeline..."
              : `${preview?.pipeline || "normal"} pipeline`}
          </span>
        </div>
        {preview?.processed_image ? (
          <img src={preview.processed_image} alt="Image after selected preprocessing pipeline" />
        ) : (
          <div className="previewPlaceholder">Choose an image to preview the selected pipeline.</div>
        )}
      </div>

      {isLoading ? (
        <div className="emptyState compact">Running inference...</div>
      ) : predictions.length ? (
        <>
          <div className="predictionList">
            {predictions.map((item, index) => (
              <article className="predictionItem" key={`${item.class_name}-${index}`}>
                <div className="predictionText">
                  <span className="rank">#{index + 1}</span>
                  <strong>{item.class_name}</strong>
                  <span>{formatConfidence(item.confidence)}</span>
                </div>
                <div className="confidenceTrack">
                  <div
                    className="confidenceBar"
                    style={{ width: `${Math.max(2, Number(item.confidence || 0) * 100)}%` }}
                  />
                </div>
              </article>
            ))}
          </div>
        </>
      ) : (
        <div className="emptyState">Top-3 classes will appear here.</div>
      )}

      <div className="metricStrip">
        <div>
          <Gauge size={17} />
          <span>Top-3</span>
        </div>
        <div>
          <Clock size={17} />
          <span>{result ? `${result.processing_time_ms} ms` : "-- ms"}</span>
        </div>
      </div>
    </section>
  );
}

export default PredictionResult;
