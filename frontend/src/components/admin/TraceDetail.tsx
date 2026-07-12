"use client";

import { useEffect, useMemo, useState } from "react";

interface AgentTraceStep {
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

interface AgentTraceItem {
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
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString();
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function StepCard({ step, idx }: { step: AgentTraceStep; idx: number }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-md border border-zinc-200 bg-white">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-4 py-3 text-left text-sm hover:bg-zinc-50"
      >
        <span className="font-medium text-zinc-900">
          step {step.step_index ?? idx}
          {step.tool_name ? (
            <span className="ml-2 rounded bg-blue-100 px-1.5 py-0.5 text-xs text-blue-800">
              tool:{step.tool_name}
            </span>
          ) : null}
          {typeof step.latency_ms === "number" ? (
            <span className="ml-2 text-xs text-zinc-500">{step.latency_ms}ms</span>
          ) : null}
          {step.error ? (
            <span className="ml-2 rounded bg-red-100 px-1.5 py-0.5 text-xs text-red-800">
              error
            </span>
          ) : null}
        </span>
        <span className="text-zinc-400">{open ? "▲" : "▼"}</span>
      </button>

      {open ? (
        <div className="space-y-3 border-t border-zinc-200 p-4">
          {step.thought ? (
            <div>
              <div className="mb-1 text-xs font-medium text-zinc-500">Thought</div>
              <pre className="overflow-auto whitespace-pre-wrap rounded bg-zinc-50 p-3 text-xs text-zinc-900">
                {step.thought}
              </pre>
            </div>
          ) : null}

          {step.action_raw ? (
            <div>
              <div className="mb-1 text-xs font-medium text-zinc-500">Action</div>
              <pre className="overflow-auto whitespace-pre-wrap rounded bg-zinc-50 p-3 text-xs text-zinc-900">
                {step.action_raw}
              </pre>
            </div>
          ) : null}

          {step.tool_args != null ? (
            <div>
              <div className="mb-1 text-xs font-medium text-zinc-500">Tool Inputs</div>
              <pre className="overflow-auto whitespace-pre-wrap rounded bg-blue-50 p-3 text-xs text-blue-900">
                {safeJson(step.tool_args)}
              </pre>
            </div>
          ) : null}

          {step.observation != null ? (
            <div>
              <div className="mb-1 text-xs font-medium text-zinc-500">Tool Outputs</div>
              <pre className="overflow-auto whitespace-pre-wrap rounded bg-emerald-50 p-3 text-xs text-emerald-900">
                {safeJson(step.observation)}
              </pre>
            </div>
          ) : null}

          {step.error ? (
            <div>
              <div className="mb-1 text-xs font-medium text-red-600">Error</div>
              <pre className="overflow-auto whitespace-pre-wrap rounded bg-red-50 p-3 text-xs text-red-900">
                {step.error}
              </pre>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

export function TraceDetail({
  apiUrl,
  runId,
}: {
  apiUrl: string;
  runId: string;
}) {
  const [trace, setTrace] = useState<AgentTraceItem | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const endpoint = useMemo(
    () => `${apiUrl}/admin/agent-traces/${runId}`,
    [apiUrl, runId],
  );

  async function loadTrace() {
    setIsLoading(true);
    setError(null);
    try {
      const resp = await fetch(endpoint, { cache: "no-store" });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }
      const data = (await resp.json()) as AgentTraceItem;
      setTrace(data);
    } catch (e) {
      setTrace(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    void loadTrace();
  }, [endpoint]);

  return (
    <div className="space-y-4">
      {/* 顶部信息栏 */}
      <div className="flex flex-col gap-3 rounded-lg border border-zinc-200 bg-white p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-lg font-semibold tracking-tight text-zinc-900">
            Trace Detail
          </h1>
          <div className="break-all text-xs text-zinc-500">{runId}</div>
        </div>
        <button
          type="button"
          onClick={loadTrace}
          className="rounded-md bg-zinc-900 px-3 py-2 text-sm font-medium text-white hover:bg-zinc-800 disabled:opacity-60"
          disabled={isLoading}
        >
          Refresh
        </button>
      </div>

      {error ? (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          {error}
        </div>
      ) : null}

      {isLoading && !trace ? (
        <div className="rounded-lg border border-zinc-200 bg-white p-4 text-sm text-zinc-600">
          Loading...
        </div>
      ) : null}

      {trace ? (
        <>
          {/* 元信息 */}
          <div className="rounded-lg border border-zinc-200 bg-white p-4">
            <div className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
              <div>
                <div className="text-xs text-zinc-500">agent_type</div>
                <div className="font-medium text-zinc-900">{trace.agent_type}</div>
              </div>
              <div>
                <div className="text-xs text-zinc-500">steps</div>
                <div className="font-medium text-zinc-900">{trace.steps.length}</div>
              </div>
              <div>
                <div className="text-xs text-zinc-500">status</div>
                <div>
                  {trace.error ? (
                    <span className="rounded bg-red-100 px-2 py-0.5 text-xs text-red-800">
                      error
                    </span>
                  ) : (
                    <span className="rounded bg-emerald-100 px-2 py-0.5 text-xs text-emerald-800">
                      ok
                    </span>
                  )}
                </div>
              </div>
              <div>
                <div className="text-xs text-zinc-500">created_at</div>
                <div className="text-xs text-zinc-700">
                  {formatDateTime(trace.created_at)}
                </div>
              </div>
            </div>
            {trace.conversation_id ? (
              <div className="mt-3 text-xs text-zinc-500">
                conversation_id: {trace.conversation_id}
              </div>
            ) : null}
          </div>

          {/* 最终结果 */}
          {trace.result ? (
            <div className="rounded-lg border border-zinc-200 bg-white p-4">
              <div className="mb-2 text-sm font-medium text-zinc-900">Result</div>
              <pre className="overflow-auto whitespace-pre-wrap text-sm text-zinc-800">
                {trace.result}
              </pre>
            </div>
          ) : null}

          {trace.error ? (
            <div className="rounded-lg border border-red-200 bg-red-50 p-4">
              <div className="mb-2 text-sm font-medium text-red-700">Error</div>
              <pre className="overflow-auto whitespace-pre-wrap text-sm text-red-900">
                {trace.error}
              </pre>
            </div>
          ) : null}

          {/* Steps */}
          <div className="space-y-2">
            <div className="text-sm font-medium text-zinc-900">
              Steps ({trace.steps.length})
            </div>
            {trace.steps.map((s, idx) => (
              <StepCard key={`${trace.run_id}-${idx}`} step={s} idx={idx} />
            ))}
          </div>
        </>
      ) : null}
    </div>
  );
}
