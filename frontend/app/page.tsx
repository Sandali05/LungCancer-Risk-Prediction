"use client";
import React, { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

type YesNo = "yes" | "no";
type RadonLevel = "low" | "medium" | "high";
type AlcoholLevel = "none" | "moderate" | "heavy";

type UiInputs = {
  GENDER: 0 | 1;
  RADON_EXPOSURE: RadonLevel;
  ALCOHOL_CONSUMPTION: AlcoholLevel;
  AGE: number;
  PACK_YEARS: number;
  ASBESTOS_EXPOSURE: YesNo;
  SECONDHAND_SMOKE_EXPOSURE: YesNo;
  COPD_DIAGNOSIS: YesNo;
  FAMILY_HISTORY: YesNo;
};

type PredictPayload = {
  age: number;
  pack_years: number;
  gender: UiInputs["GENDER"];
  radon_exposure: UiInputs["RADON_EXPOSURE"];
  asbestos_exposure: UiInputs["ASBESTOS_EXPOSURE"];
  secondhand_smoke_exposure: UiInputs["SECONDHAND_SMOKE_EXPOSURE"];
  copd_diagnosis: UiInputs["COPD_DIAGNOSIS"];
  alcohol_consumption: UiInputs["ALCOHOL_CONSUMPTION"];
  family_history: UiInputs["FAMILY_HISTORY"];
};

type PredictResponse = {
  model: string;
  risk_percentage: number;
  raw_risk_percentage?: number | null;
  adjusted_risk_percentage?: number | null;
  adjusted_for_prevalence: boolean;
  pi_train?: number | null;
  pi_deploy?: number | null;
  inputs_used?: Record<string, string | number | boolean | null>;
};

type ModelInfo = {
  feature_order?: string[];
  numeric_cols?: string[];
  binary_cols?: string[];
  one_hot_cols?: string[];
  radon_levels?: string[];
  alcohol_levels?: string[];
  pi_train?: number | null;
  pi_deploy?: number | null;
};

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-white p-3 shadow">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-lg font-semibold">{value}</div>
    </div>
  );
}

const toNumber = (value: unknown): number => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
};

const toBinary = (value: unknown): 0 | 1 => (toNumber(value) >= 1 ? 1 : 0);

const uiToApiPayload = (inputs: UiInputs): PredictPayload => ({
  age: toNumber(inputs.AGE),
  pack_years: toNumber(inputs.PACK_YEARS),
  gender: inputs.GENDER,
  radon_exposure: inputs.RADON_EXPOSURE,
  asbestos_exposure: inputs.ASBESTOS_EXPOSURE,
  secondhand_smoke_exposure: inputs.SECONDHAND_SMOKE_EXPOSURE,
  copd_diagnosis: inputs.COPD_DIAGNOSIS,
  alcohol_consumption: inputs.ALCOHOL_CONSUMPTION,
  family_history: inputs.FAMILY_HISTORY,
});

const fetchPredict = async (
  inputs: UiInputs,
  baselinePct?: number
): Promise<PredictResponse> => {
  const payload = uiToApiPayload(inputs);
  const query = baselinePct != null ? `?pi_deploy=${(baselinePct / 100).toFixed(4)}` : "";
  const response = await fetch(`${API_BASE}/predict${query}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`POST /predict failed: ${response.status} ${await response.text()}`);
  }
  return (await response.json()) as PredictResponse;
};

const fetchModelInfo = async (): Promise<ModelInfo> => {
  const response = await fetch(`${API_BASE}/model-info`);
  if (!response.ok) {
    throw new Error(`GET /model-info failed: ${response.status}`);
  }
  return (await response.json()) as ModelInfo;
};

const describeError = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }
  return typeof error === "string" ? error : "Prediction failed";
};

const binaryFields: Array<[
  keyof Pick<UiInputs, "ASBESTOS_EXPOSURE" | "SECONDHAND_SMOKE_EXPOSURE" | "COPD_DIAGNOSIS" | "FAMILY_HISTORY">,
  string
]> = [
  ["ASBESTOS_EXPOSURE", "Asbestos exposure"],
  ["SECONDHAND_SMOKE_EXPOSURE", "Secondhand smoke exposure"],
  ["COPD_DIAGNOSIS", "COPD diagnosis"],
  ["FAMILY_HISTORY", "Family history of lung cancer"],
];

export default function Page() {
  const [inputs, setInputs] = useState<UiInputs>({
    GENDER: 1,
    RADON_EXPOSURE: "low",
    ALCOHOL_CONSUMPTION: "none",
    AGE: 60,
    PACK_YEARS: 20,
    ASBESTOS_EXPOSURE: "no",
    SECONDHAND_SMOKE_EXPOSURE: "no",
    COPD_DIAGNOSIS: "no",
    FAMILY_HISTORY: "no",
  });
  const [baseline, setBaseline] = useState<number>(50);
  const [pct, setPct] = useState<string>("—");
  const [details, setDetails] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    fetchModelInfo().then(setModelInfo).catch(() => {});
  }, []);

  const onChange = <K extends keyof UiInputs>(key: K, value: UiInputs[K]) => {
    setInputs((prev) => ({ ...prev, [key]: value }));
  };

  const onPredict = async () => {
    setError(null);
    setLoading(true);
    try {
      const data = await fetchPredict(inputs, baseline);
      const main = Number(data.risk_percentage) || 0;
      setPct(`${main.toFixed(1)}%`);
      setDetails(data);
    } catch (err) {
      setError(describeError(err));
      setPct("—");
      setDetails(null);
    } finally {
      setLoading(false);
    }
  };

  const barWidth = pct.endsWith("%") ? pct : "0%";

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <div className="mx-auto max-w-5xl p-6">
        <header className="mb-6 flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold">Lung Cancer Risk Predictor</h1>
            <p className="mt-1 text-gray-600">
              Enter patient factors to estimate the model&apos;s predicted probability of lung cancer.
              <span className="ml-2 text-xs text-gray-500">(Prototype; not medical advice)</span>
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Stat label="AUC (test)" value={"0.737"} />
            <Stat label="Accuracy (test)" value={"0.840"} />
          </div>
        </header>

        <main className="grid grid-cols-1 gap-6 md:grid-cols-3">
          <section className="md:col-span-2 space-y-6">
            <div className="rounded-2xl bg-white p-6 shadow">
              <h2 className="mb-4 text-xl font-semibold">Demographics & Exposure</h2>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div>
                  <label className="mb-1 block text-sm font-medium">Gender (0=female, 1=male)</label>
                  <input
                    type="number"
                    className="w-full rounded-xl border px-3 py-2"
                    value={inputs.GENDER}
                    min={0}
                    max={1}
                    step={1}
                    onChange={(event) => onChange("GENDER", toBinary(event.target.value))}
                  />
                </div>

                <div>
                  <label className="mb-1 block text-sm font-medium">Radon Exposure</label>
                  <select
                    className="w-full rounded-xl border px-3 py-2"
                    value={inputs.RADON_EXPOSURE}
                    onChange={(event) => onChange("RADON_EXPOSURE", event.target.value as RadonLevel)}
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                </div>

                <div>
                  <label className="mb-1 block text-sm font-medium">Alcohol Consumption</label>
                  <select
                    className="w-full rounded-xl border px-3 py-2"
                    value={inputs.ALCOHOL_CONSUMPTION}
                    onChange={(event) => onChange("ALCOHOL_CONSUMPTION", event.target.value as AlcoholLevel)}
                  >
                    <option value="none">None</option>
                    <option value="moderate">Moderate</option>
                    <option value="heavy">Heavy</option>
                  </select>
                </div>

                <div>
                  <label className="mb-1 block text-sm font-medium">Age (years)</label>
                  <input
                    type="number"
                    className="w-full rounded-xl border px-3 py-2"
                    value={inputs.AGE}
                    min={0}
                    onChange={(event) => onChange("AGE", toNumber(event.target.value))}
                  />
                </div>

                <div>
                  <label className="mb-1 block text-sm font-medium">Pack-years (smoking)</label>
                  <input
                    type="number"
                    className="w-full rounded-xl border px-3 py-2"
                    value={inputs.PACK_YEARS}
                    min={0}
                    onChange={(event) => onChange("PACK_YEARS", toNumber(event.target.value))}
                  />
                </div>

                {binaryFields.map(([key, label]) => (
                  <div key={key}>
                    <label className="mb-1 block text-sm font-medium">{label}</label>
                    <select
                      className="w-full rounded-xl border px-3 py-2"
                      value={inputs[key]}
                      onChange={(event) => onChange(key, event.target.value as YesNo)}
                    >
                      <option value="no">No</option>
                      <option value="yes">Yes</option>
                    </select>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-2xl bg-white p-6 shadow">
              <h2 className="mb-4 text-xl font-semibold">Model Details</h2>
              <p className="text-sm text-gray-600">
                This <b>calibrated XGBoost</b> backend returns a probability. Age &amp; pack-years are standardized
                on the server; categories like Radon and Alcohol are parsed as text and one-hot encoded server-side.
              </p>
              {modelInfo && (
                <details className="mt-3">
                  <summary className="cursor-pointer text-sm text-gray-700">See feature order</summary>
                  <div className="mt-2 text-xs text-gray-600">
                    {modelInfo.feature_order?.join(", ")}
                  </div>
                </details>
              )}
            </div>
          </section>

          <aside className="md:col-span-1">
            <div className="sticky top-6 rounded-2xl bg-white p-6 shadow">
              <h2 className="mb-2 text-xl font-semibold">Predicted Risk</h2>
              <div className="mb-2 flex items-center gap-2">
                {details?.adjusted_for_prevalence ? (
                  <span className="rounded-full bg-emerald-100 px-2 py-1 text-xs text-emerald-700">
                    Adjusted to π<sub>deploy</sub>
                    {details?.pi_deploy != null ? ` = ${(details.pi_deploy * 100).toFixed(2)}%` : ""}
                  </span>
                ) : (
                  <span className="rounded-full bg-amber-100 px-2 py-1 text-xs text-amber-700">
                    Using training prior (raw)
                  </span>
                )}
              </div>

              <div className="mb-4 text-5xl font-bold">{pct}</div>
              {details && (
                <div className="mb-3 text-xs text-gray-600 space-y-1">
                  {details.raw_risk_percentage != null && (
                    <div>Raw (training prior): <b>{details.raw_risk_percentage.toFixed(2)}%</b></div>
                  )}
                  {details.adjusted_risk_percentage != null && (
                    <div>Adjusted: <b>{details.adjusted_risk_percentage.toFixed(2)}%</b></div>
                  )}
                  {details.pi_train != null && (
                    <div>π<sub>train</sub>: {(details.pi_train * 100).toFixed(2)}%</div>
                  )}
                </div>
              )}

              <div className="mt-2 h-3 w-full overflow-hidden rounded-full bg-gray-200">
                <div className="h-full rounded-full bg-black" style={{ width: barWidth }} />
              </div>

              <div className="mt-6">
                <label className="mb-1 block text-sm font-medium">
                  Assumed deployment prevalence (baseline)
                </label>
                <input
                  type="range"
                  className="w-full"
                  min={1}
                  max={70}
                  step={1}
                  value={baseline}
                  onChange={(event) => setBaseline(Number(event.target.value))}
                />
                <div className="mt-1 text-xs text-gray-600">Baseline: {baseline.toFixed(0)}%</div>

                <button
                  onClick={onPredict}
                  className="mt-4 w-full rounded-xl bg-black px-4 py-2 text-white disabled:opacity-60"
                  disabled={loading}
                >
                  {loading ? "Predicting…" : "Predict"}
                </button>
              </div>

              {error && (
                <div className="mt-4 rounded-xl bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>
              )}
            </div>
          </aside>
        </main>

        <footer className="mt-10 text-xs text-gray-500">
          <p>⚠️ Educational prototype only. Do not use for diagnosis or treatment decisions.</p>
        </footer>
      </div>
    </div>
  );
}
