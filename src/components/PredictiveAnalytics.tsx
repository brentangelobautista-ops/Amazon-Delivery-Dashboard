import { useEffect, useMemo, useState } from "react";
import { Activity, RefreshCw, Wifi, WifiOff } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  connectPredictionService,
  predictDelivery,
  PredictionApiError,
  type DeliveryScenario,
  type PredictionMetadata,
  type PredictionResponse,
} from "../services/predictionApi";
import { staticModelMetadata } from "../services/staticModelMetadata";

type DatasetOptions = {
  weather: string[];
  traffic: string[];
  vehicle: string[];
  area: string[];
  category: string[];
};

type ConnectionState = "checking" | "online" | "offline";

type Props = {
  options: DatasetOptions;
};

const chartTooltip = {
  borderRadius: 7,
  border: "1px solid var(--line)",
  background: "var(--panel)",
};

function pickOption(options: string[], preferred: string) {
  return options.includes(preferred) ? preferred : (options[0] ?? preferred);
}

function createScenario(
  options: DatasetOptions,
  traffic: string,
): DeliveryScenario {
  return {
    Agent_Age: 30,
    Agent_Rating: 4.5,
    Distance_Model_km: 8,
    Order_Hour: 12,
    Day_of_Week: 2,
    Weather: pickOption(options.weather, "Sunny"),
    Traffic: pickOption(options.traffic, traffic),
    Vehicle: pickOption(options.vehicle, "motorcycle"),
    Area: pickOption(options.area, "Urban"),
    Category: pickOption(options.category, "Electronics"),
  };
}

function readableError(error: unknown) {
  return error instanceof Error
    ? error.message
    : "The prediction service could not be reached.";
}

function mergeLiveMetadata(
  liveMetadata: Partial<PredictionMetadata>,
): PredictionMetadata {
  const diagnostics = liveMetadata.evaluation_diagnostics;
  return {
    ...staticModelMetadata,
    ...liveMetadata,
    held_out_metrics: {
      ...staticModelMetadata.held_out_metrics,
      ...liveMetadata.held_out_metrics,
    },
    candidate_models: liveMetadata.candidate_models?.length
      ? liveMetadata.candidate_models
      : staticModelMetadata.candidate_models,
    feature_importance: liveMetadata.feature_importance?.length
      ? liveMetadata.feature_importance
      : staticModelMetadata.feature_importance,
    evaluation_diagnostics: {
      ...staticModelMetadata.evaluation_diagnostics,
      ...diagnostics,
      actual_vs_predicted_sample:
        diagnostics?.actual_vs_predicted_sample ??
        staticModelMetadata.evaluation_diagnostics.actual_vs_predicted_sample,
      residual_histogram:
        diagnostics?.residual_histogram ??
        staticModelMetadata.evaluation_diagnostics.residual_histogram,
    },
  };
}

function ModelMetric({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note: string;
}) {
  return (
    <section className="model-metric">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </section>
  );
}

function ConnectionStatus({
  state,
  message,
  onRetry,
}: {
  state: ConnectionState;
  message: string;
  onRetry: () => void;
}) {
  const icon =
    state === "online" ? (
      <Wifi />
    ) : state === "offline" ? (
      <WifiOff />
    ) : (
      <Activity />
    );
  const label =
    state === "online"
      ? "Prediction service connected"
      : state === "checking"
        ? "Checking prediction service"
        : "Prediction service offline";

  return (
    <section className={`prediction-connection ${state}`} aria-live="polite">
      {icon}
      <div>
        <b>{label}</b>
        <p>{message}</p>
      </div>
      {state !== "online" && (
        <button type="button" onClick={onRetry} disabled={state === "checking"}>
          <RefreshCw />
          Retry connection
        </button>
      )}
    </section>
  );
}

function ModelEvaluation({ metadata }: { metadata: PredictionMetadata }) {
  const diagnostics = metadata.evaluation_diagnostics;
  const scatterData = diagnostics.actual_vs_predicted_sample;
  const scatterDomain = useMemo(() => {
    const values = scatterData.flatMap(({ actual, predicted }) => [
      actual,
      predicted,
    ]);
    const min = Math.floor(Math.min(...values, 0) / 10) * 10;
    const max = Math.ceil(Math.max(...values, 1) / 10) * 10;
    return { min, max: Math.max(max, min + 10) };
  }, [scatterData]);

  return (
    <>
      <section className="panel model-evaluation">
        <div className="panel-head">
          <div>
            <h3>Model comparison</h3>
            <p>Chronological held-out evaluation. Lower error is better.</p>
          </div>
        </div>
        <div className="model-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>R²</th>
              </tr>
            </thead>
            <tbody>
              {metadata.candidate_models.map((model) => (
                <tr key={model.model}>
                  <td>{model.model}</td>
                  <td>{model.mae} min</td>
                  <td>{model.rmse} min</td>
                  <td>{model.r2}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <h3 className="importance-title">Feature importance</h3>
        <p className="importance-note">
          Model importance summarizes predictive contribution. It does not prove
          causality.
        </p>
        <div className="importance-list">
          {metadata.feature_importance.map((feature) => (
            <div key={feature.feature}>
              <span>{feature.feature}</span>
              <i>
                <b
                  style={{ width: `${Math.round(feature.importance * 100)}%` }}
                />
              </i>
              <em>{(feature.importance * 100).toFixed(1)}%</em>
            </div>
          ))}
        </div>
      </section>

      <div className="evaluation-charts">
        <section className="panel evaluation-chart">
          <div className="panel-head">
            <div>
              <h3>Actual vs. predicted</h3>
              <p>Held-out delivery durations from the training evaluation.</p>
            </div>
          </div>
          <div className="chart-body">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 8, right: 14, bottom: 12, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="actual"
                  name="Actual"
                  type="number"
                  domain={[scatterDomain.min, scatterDomain.max]}
                  tick={{ fontSize: 11 }}
                  label={{
                    value: "Actual minutes",
                    position: "insideBottom",
                    offset: -4,
                  }}
                />
                <YAxis
                  dataKey="predicted"
                  name="Predicted"
                  type="number"
                  domain={[scatterDomain.min, scatterDomain.max]}
                  tick={{ fontSize: 11 }}
                  label={{
                    value: "Predicted minutes",
                    angle: -90,
                    position: "insideLeft",
                  }}
                />
                <Tooltip
                  contentStyle={chartTooltip}
                  formatter={(value) => Number(value).toFixed(1)}
                />
                <ReferenceLine
                  segment={[
                    { x: scatterDomain.min, y: scatterDomain.min },
                    { x: scatterDomain.max, y: scatterDomain.max },
                  ]}
                  stroke="var(--muted)"
                  strokeDasharray="4 4"
                />
                <Scatter data={scatterData} fill="#e97911" fillOpacity={0.28} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="panel evaluation-chart">
          <div className="panel-head">
            <div>
              <h3>Prediction residuals</h3>
              <p>Predicted minus actual delivery time on the held-out set.</p>
            </div>
          </div>
          <div className="chart-body">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={diagnostics.residual_histogram}
                margin={{ top: 8, right: 8, bottom: 20, left: -8 }}
              >
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="range"
                  tick={{ fontSize: 9 }}
                  interval="preserveStartEnd"
                  angle={-22}
                  textAnchor="end"
                  height={52}
                />
                <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                <Tooltip
                  contentStyle={chartTooltip}
                  formatter={(value) => Number(value).toLocaleString()}
                />
                <Bar
                  dataKey="count"
                  name="Held-out deliveries"
                  fill="#e97911"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>
    </>
  );
}

export default function PredictiveAnalytics({ options }: Props) {
  const [metadata, setMetadata] =
    useState<PredictionMetadata>(staticModelMetadata);
  const [connection, setConnection] = useState<ConnectionState>("checking");
  const [connectionMessage, setConnectionMessage] = useState(
    "Connecting to the local FastAPI inference service…",
  );
  const [connectionAttempt, setConnectionAttempt] = useState(0);
  const [predicting, setPredicting] = useState<"single" | "compare" | null>(
    null,
  );
  const [requestError, setRequestError] = useState("");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [comparison, setComparison] = useState<{
    a: PredictionResponse;
    b: PredictionResponse;
  } | null>(null);
  const [scenario, setScenario] = useState<DeliveryScenario>(() =>
    createScenario(options, "Medium"),
  );
  const [scenarioB, setScenarioB] = useState<DeliveryScenario>(() =>
    createScenario(options, "High"),
  );

  useEffect(() => {
    let active = true;
    connectPredictionService()
      .then((liveMetadata) => {
        if (!active) return;
        setMetadata(mergeLiveMetadata(liveMetadata));
        setConnection("online");
        setConnectionMessage(
          "Live model inference is available. Historical evaluation data has been verified from the loaded model.",
        );
      })
      .catch((error) => {
        if (!active) return;
        setMetadata(staticModelMetadata);
        setConnection("offline");
        setConnectionMessage(
          `${readableError(error)} Historical evaluation data below is bundled from the verified training artifact; new estimates are disabled.`,
        );
      });

    return () => {
      active = false;
    };
  }, [connectionAttempt]);

  const retryConnection = () => {
    setConnection("checking");
    setConnectionMessage("Connecting to the local FastAPI inference service…");
    setConnectionAttempt((attempt) => attempt + 1);
  };
  const isOnline = connection === "online";

  const handlePredictionFailure = (error: unknown) => {
    const message = readableError(error);
    setRequestError(message);
    if (
      !(error instanceof PredictionApiError) ||
      !error.status ||
      error.status >= 500
    ) {
      setConnection("offline");
      setConnectionMessage(
        `${message} Retry the connection after the API is available again.`,
      );
    }
  };

  const requestPrediction = async () => {
    if (!isOnline) return;
    setPredicting("single");
    setPrediction(null);
    setRequestError("");
    try {
      setPrediction(await predictDelivery(scenario));
    } catch (error) {
      handlePredictionFailure(error);
    } finally {
      setPredicting(null);
    }
  };

  const compareScenarios = async () => {
    if (!isOnline) return;
    setPredicting("compare");
    setComparison(null);
    setRequestError("");
    try {
      const [a, b] = await Promise.all([
        predictDelivery(scenario),
        predictDelivery(scenarioB),
      ]);
      setComparison({ a, b });
    } catch (error) {
      handlePredictionFailure(error);
    } finally {
      setPredicting(null);
    }
  };

  return (
    <div className="predictive">
      <ConnectionStatus
        state={connection}
        message={connectionMessage}
        onRetry={retryConnection}
      />

      <div className="model-kpis">
        <ModelMetric
          label="Selected model"
          value={metadata.model}
          note="Chosen by held-out MAE"
        />
        <ModelMetric
          label="Held-out MAE"
          value={`${metadata.held_out_metrics.mae} min`}
          note="Mean absolute error"
        />
        <ModelMetric
          label="Held-out RMSE"
          value={`${metadata.held_out_metrics.rmse} min`}
          note="Root mean squared error"
        />
        <ModelMetric
          label="Held-out R²"
          value={String(metadata.held_out_metrics.r2)}
          note={`${metadata.test_rows.toLocaleString()} chronological test rows`}
        />
      </div>

      <div className="predict-grid">
        <section className="panel scenario">
          <div className="panel-head">
            <div>
              <h3>Delivery time estimate</h3>
              <p>Model-based estimate from the historical dataset.</p>
            </div>
          </div>
          <div className="scenario-fields">
            {(
              ["Traffic", "Weather", "Vehicle", "Area", "Category"] as const
            ).map((field) => {
              const key = field.toLowerCase() as keyof DatasetOptions;
              return (
                <label key={field}>
                  {field}
                  <select
                    value={scenario[field]}
                    onChange={(event) =>
                      setScenario((current) => ({
                        ...current,
                        [field]: event.target.value,
                      }))
                    }
                  >
                    {options[key].map((option) => (
                      <option key={option}>{option}</option>
                    ))}
                  </select>
                </label>
              );
            })}
            <label>
              Route distance (km)
              <input
                type="number"
                min="0"
                step="0.1"
                value={scenario.Distance_Model_km}
                onChange={(event) =>
                  setScenario((current) => ({
                    ...current,
                    Distance_Model_km: Number(event.target.value),
                  }))
                }
              />
            </label>
            <label>
              Agent rating
              <input
                type="number"
                min="1"
                max="6"
                step="0.1"
                value={scenario.Agent_Rating}
                onChange={(event) =>
                  setScenario((current) => ({
                    ...current,
                    Agent_Rating: Number(event.target.value),
                  }))
                }
              />
            </label>
            <label>
              Agent age
              <input
                type="number"
                min="15"
                max="50"
                value={scenario.Agent_Age}
                onChange={(event) =>
                  setScenario((current) => ({
                    ...current,
                    Agent_Age: Number(event.target.value),
                  }))
                }
              />
            </label>
            <label>
              Order hour
              <input
                type="number"
                min="0"
                max="23"
                value={scenario.Order_Hour}
                onChange={(event) =>
                  setScenario((current) => ({
                    ...current,
                    Order_Hour: Number(event.target.value),
                  }))
                }
              />
            </label>
          </div>
          <button
            className="predict-button"
            type="button"
            onClick={requestPrediction}
            disabled={!isOnline || predicting !== null}
            title={
              !isOnline
                ? "Start the local prediction API to estimate delivery time."
                : undefined
            }
          >
            {predicting === "single" ? "Estimating…" : "Estimate delivery time"}
          </button>
          {!isOnline && (
            <p className="live-inference-note">
              Live estimation is disabled until the prediction service
              reconnects.
            </p>
          )}
          {requestError && (
            <p className="prediction-inline-error" role="alert">
              {requestError}
            </p>
          )}
          {prediction && (
            <div className="prediction-result">
              <span>Estimated duration</span>
              <strong>{prediction.predicted_delivery_minutes} min</strong>
              <small>
                {Math.floor(prediction.predicted_delivery_minutes / 60)}h{" "}
                {Math.round(prediction.predicted_delivery_minutes % 60)}m
              </small>
              <p>{prediction.disclaimer}</p>
              {prediction.distance_input_capped && (
                <p>
                  The supplied distance exceeded the training P99 and was capped
                  for robust inference.
                </p>
              )}
            </div>
          )}
        </section>

        <ModelEvaluation metadata={metadata} />
      </div>

      <section className="scenario-compare panel">
        <div className="panel-head">
          <div>
            <h3>What-if scenario comparison</h3>
            <p>
              Compare two model-based estimates. Differences reflect learned
              associations, not causal effects.
            </p>
          </div>
        </div>
        <div className="compare-inputs">
          {(
            [
              ["A", scenario, setScenario],
              ["B", scenarioB, setScenarioB],
            ] as const
          ).map(([name, state, setState]) => (
            <div key={name}>
              <b>Scenario {name}</b>
              {(["Traffic", "Weather", "Vehicle"] as const).map((field) => {
                const key = field.toLowerCase() as keyof DatasetOptions;
                return (
                  <label key={field}>
                    {field}
                    <select
                      value={state[field]}
                      onChange={(event) =>
                        setState((current) => ({
                          ...current,
                          [field]: event.target.value,
                        }))
                      }
                    >
                      {options[key].map((option) => (
                        <option key={option}>{option}</option>
                      ))}
                    </select>
                  </label>
                );
              })}
            </div>
          ))}
        </div>
        <button
          className="predict-button compare-button"
          type="button"
          onClick={compareScenarios}
          disabled={!isOnline || predicting !== null}
          title={
            !isOnline
              ? "Start the local prediction API to compare scenarios."
              : undefined
          }
        >
          {predicting === "compare" ? "Comparing…" : "Compare scenarios"}
        </button>
        {comparison && (
          <div className="compare-results">
            <div>
              <span>Scenario A</span>
              <strong>{comparison.a.predicted_delivery_minutes} min</strong>
            </div>
            <div>
              <span>Scenario B</span>
              <strong>{comparison.b.predicted_delivery_minutes} min</strong>
            </div>
            <div>
              <span>Difference</span>
              <strong>
                {(
                  comparison.b.predicted_delivery_minutes -
                  comparison.a.predicted_delivery_minutes
                ).toFixed(1)}{" "}
                min
              </strong>
              <small>
                {comparison.a.predicted_delivery_minutes
                  ? `${(((comparison.b.predicted_delivery_minutes - comparison.a.predicted_delivery_minutes) / comparison.a.predicted_delivery_minutes) * 100).toFixed(1)}% vs. A`
                  : ""}
              </small>
            </div>
          </div>
        )}
      </section>

      <section className="predict-limitations">
        <h3>Model limitations</h3>
        <p>
          Predictions reflect relationships in this historical dataset. They are
          estimates, not actual delivery outcomes. The model uses a
          training-only distance cap at its 99th percentile for robustness, and
          it does not establish that changing a single input will cause a
          delivery-time change. The dataset contains only 44 observed order
          dates, so demand forecasting is not implemented.
        </p>
      </section>
    </div>
  );
}
