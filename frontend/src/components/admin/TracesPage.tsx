"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

export interface AgentTraceStep {
  step_index?: number;
  thought?: string;
  action_raw?: string;
  tool_name?: string | null;
  tool_args?: Record<string, unknown> | null;
  observation?: unknown;
  error?: string | null;
  started_at_ms?: number;
  finished_at_ms?: number;
  latency_ms?: number;
}

export interface AgentTraceItem {
  run_id: string;
  conversation_id: string | null;
  agent_type: string;
  steps: AgentTraceStep[];
  result: string | null;
  error: string | null;
  created_at: string;
}

function formatDateTime(value: string): string {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) {
    return value;
  }
  return d.toLocaleString();
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function TracesPage({
  apiUrl,
  initialTraces,
}: {
  apiUrl: string;
  initialTraces: AgentTraceItem[] | null;
}) {
  const [traces, setTraces] = useState<AgentTraceItem[] | null>(initialTraces);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const endpoint = useMemo(() => `${apiUrl}/admin/agent-traces?limit=50`, [apiUrl]);

  async function loadTraces() {
    setIsLoading(true);
    setError(null);
    try {
      const resp = await fetch(endpoint, { cache: "no-store" });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }
      const data = (await resp.json()) as AgentTraceItem[];
      setTraces(data);
    } catch (e) {
      setTraces(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 rounded-lg border border-zinc-200 bg-white p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-lg font-semibold tracking-tight text-zinc-900">
            Agent Traces
          </h1>
          <div className="text-sm text-zinc-600">{endpoint}</div>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={loadTraces}
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

      {isLoading && !traces ? (
        <div className="rounded-lg border border-zinc-200 bg-white p-4 text-sm text-zinc-600">
          Loading...
        </div>
      ) : null}

      {traces?.length === 0 ? (
        <div className="rounded-lg border border-zinc-200 bg-white p-4 text-sm text-zinc-600">
          No traces yet. Call POST /agent with conversation_id to generate traces.
        </div>
      ) : null}

      {traces?.map((t) => (
        <details
          key={t.run_id}
          className="rounded-lg border border-zinc-200 bg-white"
        >
          <summary className="flex cursor-pointer flex-col gap-1 p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-1">
              <div className="text-sm font-medium text-zinc-900">{t.run_id}</div>
              <div className="text-xs text-zinc-600">
                {formatDateTime(t.created_at)} · {t.agent_type} · steps{" "}
                {t.steps.length}
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs">
              {t.error ? (
                <span className="rounded bg-red-100 px-2 py-1 text-red-800">
                  error
                </span>
              ) : (
                <span className="rounded bg-emerald-100 px-2 py-1 text-emerald-800">
                  ok
                </span>
              )}
              <Link
                href={`/admin/traces/${t.run_id}`}
                onClick={(e) => e.stopPropagation()}
                className="rounded bg-zinc-900 px-2 py-1 text-white hover:bg-zinc-700"
              >
                Detail →
              </Link>
            </div>
          </summary>

          <div className="space-y-3 border-t border-zinc-200 p-4">
            {t.conversation_id ? (
              <div className="text-xs text-zinc-600">
                conversation_id: {t.conversation_id}
              </div>
            ) : null}
            {t.result ? (
              <div className="rounded-md border border-zinc-200 bg-zinc-50 p-3">
                <div className="text-xs font-medium text-zinc-700">result</div>
                <pre className="mt-2 overflow-auto whitespace-pre-wrap text-xs text-zinc-900">
                  {t.result}
                </pre>
              </div>
            ) : null}
            {t.error ? (
              <div className="rounded-md border border-red-200 bg-red-50 p-3">
                <div className="text-xs font-medium text-red-700">error</div>
                <pre className="mt-2 overflow-auto whitespace-pre-wrap text-xs text-red-900">
                  {t.error}
                </pre>
              </div>
            ) : null}

            <div className="space-y-2">
              <div className="text-sm font-medium text-zinc-900">steps</div>
              {t.steps.map((s, idx) => (
                <details
                  key={`${t.run_id}-${idx}`}
                  className="rounded-md border border-zinc-200 bg-white"
                >
                  <summary className="cursor-pointer px-3 py-2 text-sm text-zinc-900">
                    step {s.step_index ?? idx}
                    {s.tool_name ? ` · tool:${s.tool_name}` : ""}
                    {typeof s.latency_ms === "number"
                      ? ` · ${s.latency_ms}ms`
                      : ""}
                    {s.error ? " · error" : ""}
                  </summary>
                  <pre className="overflow-auto whitespace-pre-wrap border-t border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-900">
                    {safeJson(s)}
                  </pre>
                </details>
              ))}
            </div>
          </div>
        </details>
      ))}
    </div>
  );
}
