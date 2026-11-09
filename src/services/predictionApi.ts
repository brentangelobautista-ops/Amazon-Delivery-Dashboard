export type DeliveryScenario = {
  Agent_Age: number;
  Agent_Rating: number;
  Distance_Model_km: number;
  Order_Hour: number;
  Day_of_Week: number;
  Weather: string;
  Traffic: string;
  Vehicle: string;
  Area: string;
  Category: string;
};

export type CandidateModel = {
  model: string;
  mae: number;
  rmse: number;
  r2: number;
};

export type FeatureImportance = {
  feature: string;
  importance: number;
};

export type EvaluationDiagnostics = {
  actual_vs_predicted_sample: Array<{ actual: number; predicted: number }>;
  residual_histogram: Array<{ range: string; count: number }>;
};

export type PredictionMetadata = {
  model: string;
  test_rows: number;
  held_out_metrics: { mae: number; rmse: number; r2: number };
  candidate_models: CandidateModel[];
  feature_importance: FeatureImportance[];
  evaluation_diagnostics: EvaluationDiagnostics;
};

export type PredictionResponse = {
  predicted_delivery_minutes: number;
  model: string;
  distance_input_capped: boolean;
  disclaimer: string;
};

export type PredictionHealth = {
  ready: boolean;
  model: string | null;
  detail: string | null;
};

export class PredictionApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "PredictionApiError";
  }
}

const configuredApiUrl = import.meta.env.VITE_PREDICTION_API_URL?.trim();
export const predictionApiBaseUrl = (configuredApiUrl || "/api").replace(
  /\/+$/,
  "",
);

function endpoint(path: string) {
  return `${predictionApiBaseUrl}${path}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 7000);

  try {
    const response = await fetch(endpoint(path), {
      ...init,
      signal: controller.signal,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      const detail =
        typeof payload?.detail === "string"
          ? payload.detail
          : `Request failed (${response.status})`;
      throw new PredictionApiError(detail, response.status);
    }
    return response.json() as Promise<T>;
  } catch (error) {
    if (error instanceof PredictionApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new PredictionApiError(
        "The prediction service did not respond in time.",
      );
    }
    throw new PredictionApiError(
      "The prediction service could not be reached.",
    );
  } finally {
    window.clearTimeout(timeout);
  }
}

export async function getPredictionHealth() {
  return request<PredictionHealth>("/health");
}

export async function getPredictionMetadata() {
  return request<PredictionMetadata>("/metadata");
}

export async function connectPredictionService() {
  const health = await getPredictionHealth();
  if (!health.ready) {
    throw new PredictionApiError(
      health.detail || "The prediction service is not ready.",
      503,
    );
  }
  return getPredictionMetadata();
}

export async function predictDelivery(scenario: DeliveryScenario) {
  return request<PredictionResponse>("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(scenario),
  });
}
