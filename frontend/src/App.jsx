import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Loader2, RotateCcw, Send, Play, BarChart4 } from "lucide-react";

import { predictVehicle, previewPipeline } from "./api/predictApi";
import ImageUploader from "./components/ImageUploader";
import ModelSelector from "./components/ModelSelector";
import PipelineSelector from "./components/PipelineSelector";
import PredictionResult from "./components/PredictionResult";
import Dashboard from "./components/Dashboard";
import "./styles/App.css";

const MODEL_OPTIONS = [
  // ResNet-50
  {
    label: "ResNet-50 (Raw Original V1)",
    value: "resnet50_raw_original_v1_best",
    description: "V1 Weather - Raw",
  },
  {
    label: "ResNet-50 (Raw Original V2)",
    value: "resnet50_raw_original_v2_best",
    description: "V2 Blur - Raw",
  },
  {
    label: "ResNet-50 (Raw Cleaning V1)",
    value: "resnet50_raw_cleaning_v1_best",
    description: "V1 Weather - Cleaned",
  },
  {
    label: "ResNet-50 (Raw Cleaning V2)",
    value: "resnet50_raw_cleaning_v2_best",
    description: "V2 Blur - Cleaned",
  },
  // YOLO-cls
  {
    label: "YOLO-cls (Raw Original V1)",
    value: "yolo_raw_original_v1_best",
    description: "V1 Weather - Raw",
  },
  {
    label: "YOLO-cls (Raw Original V2)",
    value: "yolo_raw_original_v2_best",
    description: "V2 Blur - Raw",
  },
  {
    label: "YOLO-cls (Raw Cleaning V1)",
    value: "yolo_raw_cleaning_v1_best",
    description: "V1 Weather - Cleaned",
  },
  {
    label: "YOLO-cls (Raw Cleaning V2)",
    value: "yolo_raw_cleaning_v2_best",
    description: "V2 Blur - Cleaned",
  },
  // ViT
  {
    label: "ViT (Raw Original V1)",
    value: "vit_raw_original_v1_best",
    description: "V1 Weather - Raw ViT",
  },
  {
    label: "ViT (Raw Original V2)",
    value: "vit_raw_original_v2_best",
    description: "V2 Blur - Raw ViT",
  },
  {
    label: "ViT (Raw Cleaning V1)",
    value: "vit_raw_cleaning_v1_best",
    description: "V1 Weather - Cleaned ViT",
  },
  {
    label: "ViT (Raw Cleaning V2)",
    value: "vit_raw_cleaning_v2_best",
    description: "V2 Blur - Cleaned ViT",
  },
];

const PIPELINE_OPTIONS = [
  {
    label: "Normal",
    value: "normal",
    description: "Base pipeline only",
  },
  {
    label: "Rain",
    value: "rain",
    description: "Rain simulation",
  },
  {
    label: "Sun",
    value: "sun",
    description: "Sun flare simulation",
  },
  {
    label: "Night",
    value: "night",
    description: "Night simulation",
  },
  {
    label: "Gaussian Blur",
    value: "gaussian",
    description: "Gaussian noise/blur",
  },
  {
    label: "Motion Blur",
    value: "motion",
    description: "Motion blur effect",
  },
  {
    label: "Unsharp Masking",
    value: "unsharp",
    description: "Unsharp masking sharpening",
  },
];

function App() {
  const [activeTab, setActiveTab] = useState("inference");
  const [selectedModel, setSelectedModel] = useState(MODEL_OPTIONS[0].value);
  const [selectedPipeline, setSelectedPipeline] = useState(PIPELINE_OPTIONS[0].value);
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [processedPreview, setProcessedPreview] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);

  useEffect(() => {
    if (!imageFile) {
      setPreviewUrl("");
      return undefined;
    }

    const url = URL.createObjectURL(imageFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  useEffect(() => {
    let cancelled = false;

    async function loadPipelinePreview() {
      if (!imageFile) {
        setProcessedPreview(null);
        return;
      }

      setIsPreviewLoading(true);
      try {
        const preview = await previewPipeline({
          imageFile,
          pipeline: selectedPipeline,
        });
        if (!cancelled) {
          setProcessedPreview(preview);
        }
      } catch (previewError) {
        if (!cancelled) {
          setProcessedPreview(null);
          setError(previewError.message || "Pipeline preview failed.");
        }
      } finally {
        if (!cancelled) {
          setIsPreviewLoading(false);
        }
      }
    }

    loadPipelinePreview();
    return () => {
      cancelled = true;
    };
  }, [imageFile, selectedPipeline]);

  const handleImageChange = (file) => {
    setImageFile(file);
    setResult(null);
    setProcessedPreview(null);
    setError("");
  };

  const handleSubmit = async () => {
    if (!imageFile) {
      setError("Please choose an image before running inference.");
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await predictVehicle({
        imageFile,
        modelName: selectedModel,
        pipeline: selectedPipeline,
      });
      setResult(data);
    } catch (requestError) {
      setError(requestError.message || "Prediction failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setImageFile(null);
    setPreviewUrl("");
    setResult(null);
    setProcessedPreview(null);
    setError("");
  };

  const handlePipelineChange = (pipeline) => {
    setSelectedPipeline(pipeline);
    setResult(null);
    setError("");
  };

  return (
    <main className="appShell">
      <section className="topBar">
        <div>
          <h1>Vehicle Type Recognition</h1>
          <p>Flask + React inference & comparison demo</p>
        </div>
        
        <div className="tabNavigation">
          <button
            className={`tabButton ${activeTab === "inference" ? "active" : ""}`}
            type="button"
            onClick={() => setActiveTab("inference")}
          >
            <Play size={16} />
            <span>Thử nghiệm Nhận dạng</span>
          </button>
          <button
            className={`tabButton ${activeTab === "dashboard" ? "active" : ""}`}
            type="button"
            onClick={() => setActiveTab("dashboard")}
          >
            <BarChart4 size={16} />
            <span>Dashboard So sánh</span>
          </button>
        </div>

        <div className="statusPill">API: /api/predict</div>
      </section>

      {activeTab === "inference" ? (
        <section className="workspace">
          <aside className="controlPanel">
            <ModelSelector
              models={MODEL_OPTIONS}
              selectedModel={selectedModel}
              onChange={setSelectedModel}
              disabled={isLoading}
            />

            <ImageUploader
              imageFile={imageFile}
              previewUrl={previewUrl}
              onImageChange={handleImageChange}
              disabled={isLoading}
            />

            <PipelineSelector
              pipelines={PIPELINE_OPTIONS}
              selectedPipeline={selectedPipeline}
              onChange={handlePipelineChange}
              disabled={isLoading}
              selectedModel={selectedModel}
            />

            {error ? <div className="errorBox">{error}</div> : null}

            <div className="actions">
              <button
                className="primaryButton"
                type="button"
                onClick={handleSubmit}
                disabled={isLoading || !imageFile}
              >
                {isLoading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
                <span>{isLoading ? "Running" : "Predict"}</span>
              </button>
              <button
                className="iconButton"
                type="button"
                onClick={handleReset}
                disabled={isLoading || (!imageFile && !result && !error)}
                aria-label="Reset"
                title="Reset"
              >
                <RotateCcw size={18} />
              </button>
            </div>
          </aside>

          <PredictionResult
            result={result}
            pipelinePreview={processedPreview}
            isLoading={isLoading}
            isPreviewLoading={isPreviewLoading}
          />
        </section>
      ) : (
        <Dashboard />
      )}
    </main>
  );
}

export default App;

createRoot(document.getElementById("root")).render(<App />);
