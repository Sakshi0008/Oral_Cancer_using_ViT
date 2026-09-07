/**
 * API Service for communicating with the FastAPI Oral Cancer Screening backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error(`Health check failed (${res.status})`);
    return await res.json();
  } catch (err) {
    return {
      status: 'offline',
      device: 'unavailable',
      cuda_available: false,
      model_loaded: false,
      checkpoint_exists: false,
      error: err.message,
    };
  }
}

export async function getModelInfo() {
  try {
    const res = await fetch(`${API_BASE_URL}/model-info`);
    if (!res.ok) throw new Error(`Failed to fetch model info (${res.status})`);
    return await res.json();
  } catch (err) {
    console.error('getModelInfo error:', err);
    return null;
  }
}

export async function predictOralImage(file, mcPasses = 15, alpha = 0.5) {
  const formData = new FormData();
  formData.append('file', file);

  const url = `${API_BASE_URL}/predict?mc_passes=${mcPasses}&alpha=${alpha}`;
  const res = await fetch(url, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Prediction failed with status ${res.status}`);
  }

  return await res.json();
}

export async function getMetrics() {
  try {
    const res = await fetch(`${API_BASE_URL}/metrics`);
    if (!res.ok) throw new Error(`Failed to fetch metrics (${res.status})`);
    return await res.json();
  } catch (err) {
    return {
      available: false,
      message: 'Cannot reach backend to fetch evaluation metrics.',
      disclaimer: 'Academic prototype metrics.',
    };
  }
}

export function getStaticOutputUrl(path) {
  if (!path) return null;
  if (path.startsWith('http')) return path;
  return `${API_BASE_URL}${path}`;
}
