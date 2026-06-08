const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

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

