"use client";

import { useState } from "react";

import type { ChatStreamStep } from "@/lib/streaming";

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function truncateText(value: string, limit: number): string {
  if (value.length <= limit) {
    return value;
  }
  return `${value.slice(0, limit)}\n...<truncated>`;
}

export function AgentStepBubble({ step }: { step: ChatStreamStep }) {
  const [open, setOpen] = useState(Boolean(step.error));

  return (
    <div className="flex w-full justify-center">
      <div className="w-full max-w-[min(860px,calc(100vw-2rem))] rounded-2xl border border-zinc-200 bg-white shadow-sm">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left text-sm hover:bg-zinc-50"
        >
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <span className="font-medium text-zinc-900">
              Step {step.step_index}
            </span>
            {step.tool_name ? (
              <span className="rounded bg-blue-100 px-2 py-0.5 text-xs text-blue-800">
                tool:{step.tool_name}
              </span>
            ) : (
              <span className="rounded bg-zinc-100 px-2 py-0.5 text-xs text-zinc-700">
                thought
              </span>
            )}
            <span className="rounded bg-zinc-100 px-2 py-0.5 text-xs text-zinc-700">
              {step.latency_ms}ms
            </span>
            {step.error ? (
              <span className="rounded bg-red-100 px-2 py-0.5 text-xs text-red-800">
                error
              </span>
            ) : null}
          </div>
          <span className="shrink-0 text-zinc-400">{open ? "▲" : "▼"}</span>
        </button>

        {open ? (
          <div className="space-y-3 border-t border-zinc-200 p-4 text-sm">
            {step.thought.trim().length > 0 ? (
              <div>
                <div className="mb-1 text-xs font-medium text-zinc-500">
                  Thought
                </div>
                <pre className="overflow-auto whitespace-pre-wrap rounded-xl bg-zinc-50 p-3 text-xs text-zinc-900">
                  {truncateText(step.thought, 8000)}
                </pre>
              </div>
            ) : null}

            {step.action_raw.trim().length > 0 ? (
              <div>
                <div className="mb-1 text-xs font-medium text-zinc-500">
                  Action
                </div>
                <pre className="overflow-auto whitespace-pre-wrap rounded-xl bg-zinc-50 p-3 text-xs text-zinc-900">
                  {truncateText(step.action_raw, 8000)}
                </pre>
              </div>
            ) : null}

            {step.tool_args != null ? (
              <div>
                <div className="mb-1 text-xs font-medium text-zinc-500">
                  Tool Inputs
                </div>
                <pre className="overflow-auto whitespace-pre-wrap rounded-xl bg-blue-50 p-3 text-xs text-blue-900">
                  {truncateText(safeJson(step.tool_args), 12000)}
                </pre>
              </div>
            ) : null}

            {step.observation != null ? (
              <div>
                <div className="mb-1 text-xs font-medium text-zinc-500">
                  Observation
                </div>
                <pre className="overflow-auto whitespace-pre-wrap rounded-xl bg-emerald-50 p-3 text-xs text-emerald-900">
                  {truncateText(safeJson(step.observation), 12000)}
                </pre>
              </div>
            ) : null}

            {step.error ? (
              <div>
                <div className="mb-1 text-xs font-medium text-red-600">
                  Error
                </div>
                <pre className="overflow-auto whitespace-pre-wrap rounded-xl bg-red-50 p-3 text-xs text-red-900">
                  {truncateText(step.error, 8000)}
                </pre>
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}

