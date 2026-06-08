const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

export async function predictVehicle({ imageFile, modelName, pipeline }) {
  const formData = new FormData();
  formData.append("image", imageFile);
  if (modelName) {
    formData.append("model_name", modelName);
  }
  if (pipeline) {
    formData.append("pipeline", pipeline);
  }

  const response = await fetch(`${API_BASE_URL}/api/predict`, {
    method: "POST",
    body: formData,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Prediction failed with status ${response.status}.`);
  }

  return payload;
}

export async function previewPipeline({ imageFile, pipeline }) {
  const formData = new FormData();
  formData.append("image", imageFile);
  if (pipeline) {
    formData.append("pipeline", pipeline);
  }

  const response = await fetch(`${API_BASE_URL}/api/preprocess`, {
    method: "POST",
    body: formData,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Pipeline preview failed with status ${response.status}.`);
  }

  return payload;
}
