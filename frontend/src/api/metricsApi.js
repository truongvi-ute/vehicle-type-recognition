const API_BASE_URL =
  (typeof window !== "undefined" && window.__API_BASE_URL__) ||
  "http://localhost:5000";

async function requestJson(path, fallbackMessage) {
  const response = await fetch(`${API_BASE_URL}${path}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || fallbackMessage);
  }
  return payload;
}

export function fetchExperimentCatalog() {
  return requestJson("/api/metrics/experiments", "Không thể tải danh mục thí nghiệm.");
}

export function fetchExperiment(experimentId) {
  return requestJson(
    `/api/metrics/experiment?id=${encodeURIComponent(experimentId)}`,
    "Không thể tải kết quả thí nghiệm.",
  );
}

export function fetchDatasetProfile(datasetId, versionId) {
  return requestJson(
    `/api/metrics/dataset-profile?dataset=${encodeURIComponent(datasetId)}&version=${encodeURIComponent(versionId)}`,
    "Không thể tải thống kê dataset.",
  );
}

export function fetchExperimentComparison(firstId, secondId) {
  return requestJson(
    `/api/metrics/compare?first=${encodeURIComponent(firstId)}&second=${encodeURIComponent(secondId)}`,
    "Không thể so sánh hai thí nghiệm.",
  );
}

export async function fetchRuns() {
  const response = await fetch(`${API_BASE_URL}/api/metrics/runs`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Failed to fetch runs.");
  }
  return payload.runs || [];
}

export async function fetchCurves(path) {
  const response = await fetch(`${API_BASE_URL}/api/metrics/curves?path=${encodeURIComponent(path)}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Failed to fetch training curves.");
  }
  return payload.curves || [];
}

export async function fetchReport(path) {
  const response = await fetch(`${API_BASE_URL}/api/metrics/report?path=${encodeURIComponent(path)}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Failed to fetch classification report.");
  }
  return payload;
}

export async function fetchConfusionMatrix(path) {
  const response = await fetch(`${API_BASE_URL}/api/metrics/confusion-matrix?path=${encodeURIComponent(path)}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Failed to fetch confusion matrix.");
  }
  return payload;
}

export async function fetchSummary() {
  const response = await fetch(`${API_BASE_URL}/api/metrics/summary`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Failed to fetch metrics summary.");
  }
  return payload.summary || [];
}

export async function fetchDataprep() {
  const response = await fetch(`${API_BASE_URL}/api/metrics/dataprep`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Failed to fetch dataprep counts.");
  }
  return payload;
}

export async function fetchSplitMetrics(path) {
  const response = await fetch(`${API_BASE_URL}/api/metrics/split-metrics?path=${encodeURIComponent(path)}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Failed to fetch split metrics.");
  }
  return payload.splits || [];
}

export async function fetchTopErrors(path) {
  const response = await fetch(`${API_BASE_URL}/api/metrics/top-errors?path=${encodeURIComponent(path)}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Failed to fetch top error pairs.");
  }
  return payload.errors || [];
}

