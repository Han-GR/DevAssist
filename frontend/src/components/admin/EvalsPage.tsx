"use client";

import { useEffect, useMemo, useState } from "react";

export interface EvalResultItem {
  id: string;
  eval_type: string;
  model_key: string;
  metric_name: string;
  scope: string;
  score: number;
  meta: Record<string, unknown> | null;
  created_at: string;
}

function formatDateTime(value: string): string {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) {
    return value;
  }
  return d.toLocaleString();
}

function uniqueSorted(values: string[]): string[] {
  return Array.from(new Set(values)).sort((a, b) => a.localeCompare(b));
}

function pickLatestPerModel(rows: EvalResultItem[]): EvalResultItem[] {
  const map = new Map<string, EvalResultItem>();
  for (const r of rows) {
    const prev = map.get(r.model_key);
    if (!prev || String(r.created_at) > String(prev.created_at)) {
      map.set(r.model_key, r);
    }
  }
  return Array.from(map.values()).sort((a, b) => a.model_key.localeCompare(b.model_key));
}

function scoreLabel(value: number): string {
  if (!Number.isFinite(value)) {
    return String(value);
  }
  if (value >= 0 && value <= 1) {
    return `${(value * 100).toFixed(1)}%`;
  }
  return value.toFixed(4);
}

export function EvalsPage({
  apiUrl,
  initialResults,
}: {
  apiUrl: string;
  initialResults: EvalResultItem[] | null;
}) {
  const [results, setResults] = useState<EvalResultItem[] | null>(initialResults);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const endpoint = useMemo(() => `${apiUrl}/admin/eval-results?limit=200`, [apiUrl]);

  const evalTypes = useMemo(
    () => uniqueSorted((results ?? []).map((r) => r.eval_type)),
    [results],
  );
  const metricNames = useMemo(
    () => uniqueSorted((results ?? []).map((r) => r.metric_name)),
    [results],
  );
  const scopes = useMemo(
    () => uniqueSorted((results ?? []).map((r) => r.scope)),
    [results],
  );

  const [selectedEvalType, setSelectedEvalType] = useState<string>(
    evalTypes[0] ?? "finetune_rubric",
  );
  const [selectedMetric, setSelectedMetric] = useState<string>(
    metricNames.includes("pass_rate") ? "pass_rate" : metricNames[0] ?? "pass_rate",
  );
  const [selectedScope, setSelectedScope] = useState<string>(
    scopes.includes("all") ? "all" : scopes[0] ?? "all",
  );

  useEffect(() => {
    if (evalTypes.length > 0 && !evalTypes.includes(selectedEvalType)) {
      setSelectedEvalType(evalTypes[0]!);
    }
  }, [evalTypes, selectedEvalType]);

  useEffect(() => {
    if (metricNames.length > 0 && !metricNames.includes(selectedMetric)) {
      setSelectedMetric(metricNames.includes("pass_rate") ? "pass_rate" : metricNames[0]!);
    }
  }, [metricNames, selectedMetric]);

  useEffect(() => {
    if (scopes.length > 0 && !scopes.includes(selectedScope)) {
      setSelectedScope(scopes.includes("all") ? "all" : scopes[0]!);
    }
  }, [scopes, selectedScope]);

  async function loadResults() {
    setIsLoading(true);
    setError(null);
    try {
      const resp = await fetch(endpoint, { cache: "no-store" });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }
      const data = (await resp.json()) as EvalResultItem[];
      setResults(data);
    } catch (e) {
      setResults(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsLoading(false);
    }
  }

  const filteredLatest = useMemo(() => {
    const filtered = (results ?? []).filter(
      (r) =>
        r.eval_type === selectedEvalType &&
        r.metric_name === selectedMetric &&
        r.scope === selectedScope,
    );
    return pickLatestPerModel(filtered);
  }, [results, selectedEvalType, selectedMetric, selectedScope]);

  const maxScore = useMemo(() => {
    const values = filteredLatest.map((r) => Number(r.score)).filter((v) => Number.isFinite(v));
    return values.length > 0 ? Math.max(...values) : 1;
  }, [filteredLatest]);

  const latestForList = useMemo(() => {
    const filtered = (results ?? []).filter((r) => r.eval_type === selectedEvalType);
    return filtered.slice(0, 50);
  }, [results, selectedEvalType]);

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 rounded-lg border border-zinc-200 bg-white p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-lg font-semibold tracking-tight text-zinc-900">
            Eval Results
          </h1>
          <div className="text-sm text-zinc-600">{endpoint}</div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={loadResults}
            className="rounded-md bg-zinc-900 px-3 py-2 text-sm font-medium text-white hover:bg-zinc-800 disabled:opacity-60"
            disabled={isLoading}
          >
            Refresh
          </button>
        </div>
      </div>

      {error ? (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          {error}
        </div>
      ) : null}

      {isLoading && !results ? (
        <div className="rounded-lg border border-zinc-200 bg-white p-4 text-sm text-zinc-600">
          Loading...
        </div>
      ) : null}

      {!results || results.length === 0 ? (
        <div className="rounded-lg border border-zinc-200 bg-white p-4 text-sm text-zinc-600">
          No eval results yet. Insert rows into eval_results to enable charts.
        </div>
      ) : null}

      {results && results.length > 0 ? (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
          <div className="rounded-lg border border-zinc-200 bg-white p-4">
            <div className="text-sm font-medium text-zinc-900">Eval Type</div>
            <select
              className="mt-2 w-full rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm"
              value={selectedEvalType}
              onChange={(e) => setSelectedEvalType(e.target.value)}
            >
              {evalTypes.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>
          <div className="rounded-lg border border-zinc-200 bg-white p-4">
            <div className="text-sm font-medium text-zinc-900">Metric</div>
            <select
              className="mt-2 w-full rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm"
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
            >
              {metricNames.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>
          <div className="rounded-lg border border-zinc-200 bg-white p-4">
            <div className="text-sm font-medium text-zinc-900">Scope</div>
            <select
              className="mt-2 w-full rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm"
              value={selectedScope}
              onChange={(e) => setSelectedScope(e.target.value)}
            >
              {scopes.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>
        </div>
      ) : null}

      {filteredLatest.length > 0 ? (
        <div className="rounded-lg border border-zinc-200 bg-white p-4">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
            <div className="space-y-1">
              <div className="text-sm font-medium text-zinc-900">Model Comparison</div>
              <div className="text-xs text-zinc-600">
                {selectedEvalType} · {selectedMetric} · {selectedScope}
              </div>
            </div>
            <div className="text-xs text-zinc-500">latest per model_key</div>
          </div>

          <div className="mt-4 space-y-3">
            {filteredLatest.map((r) => {
              const ratio = maxScore > 0 ? Math.max(0, Math.min(1, r.score / maxScore)) : 0;
              return (
                <div key={r.id} className="space-y-1">
                  <div className="flex items-center justify-between gap-3 text-sm">
                    <div className="font-medium text-zinc-900">{r.model_key}</div>
                    <div className="text-zinc-700">{scoreLabel(r.score)}</div>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded bg-zinc-100">
                    <div
                      className="h-full rounded bg-zinc-900"
                      style={{ width: `${(ratio * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <div className="text-xs text-zinc-500">{formatDateTime(r.created_at)}</div>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}

      {latestForList.length > 0 ? (
        <div className="rounded-lg border border-zinc-200 bg-white">
          <div className="border-b border-zinc-200 p-4">
            <div className="text-sm font-medium text-zinc-900">Latest Rows</div>
            <div className="text-xs text-zinc-600">{selectedEvalType}</div>
          </div>
          <div className="overflow-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-zinc-50 text-xs text-zinc-600">
                <tr>
                  <th className="px-4 py-3 font-medium">time</th>
                  <th className="px-4 py-3 font-medium">model</th>
                  <th className="px-4 py-3 font-medium">metric</th>
                  <th className="px-4 py-3 font-medium">scope</th>
                  <th className="px-4 py-3 font-medium">score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-200">
                {latestForList.map((r) => (
                  <tr key={r.id} className="hover:bg-zinc-50">
                    <td className="px-4 py-3 text-xs text-zinc-600">
                      {formatDateTime(r.created_at)}
                    </td>
                    <td className="px-4 py-3 font-medium text-zinc-900">{r.model_key}</td>
                    <td className="px-4 py-3 text-zinc-700">{r.metric_name}</td>
                    <td className="px-4 py-3 text-zinc-700">{r.scope}</td>
                    <td className="px-4 py-3 text-zinc-700">{scoreLabel(r.score)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </div>
  );
}
