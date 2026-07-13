"use client";

import { useMemo, useState } from "react";

export interface CodeExecutionResult {
  stdout: string;
  stderr: string;
  exit_code: number;
  duration_ms: number;
}

export interface CodeBlockProps {
  code: string;
  language: string;
  apiUrl?: string;
  conversationId?: string | null;
}

function normalizeLanguage(language: string): string {
  const value = language.trim().toLowerCase();
  if (value === "py") {
    return "python";
  }
  return value;
}

export function CodeBlock(props: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<CodeExecutionResult | null>(null);

  const lang = useMemo(() => normalizeLanguage(props.language), [props.language]);
  const canRun = Boolean(props.apiUrl) && (lang === "python");

  async function copy() {
    await navigator.clipboard.writeText(props.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 900);
  }

  async function run() {
    if (!props.apiUrl) {
      return;
    }
    setIsRunning(true);
    setRunError(null);
    setResult(null);

    try {
      const resp = await fetch(`${props.apiUrl}/agent`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          message: `你必须调用 execute_code 工具执行下面的 Python 代码，并把工具返回的 observation 原样作为最终答案输出（不要额外解释）。\n\n\`\`\`python\n${props.code}\n\`\`\``,
          tools: ["execute_code"],
          conversation_id: props.conversationId ?? undefined,
        }),
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }

      const data = (await resp.json()) as {
        steps?: Array<{ tool_name?: unknown; observation?: unknown }>;
      };

      const steps = Array.isArray(data.steps) ? data.steps : [];
      const execution = steps.find((s) => s.tool_name === "execute_code")?.observation;
      if (!execution || typeof execution !== "object") {
        throw new Error("未找到 execute_code 的执行结果。");
      }

      const stdout = (execution as { stdout?: unknown }).stdout;
      const stderr = (execution as { stderr?: unknown }).stderr;
      const exitCode = (execution as { exit_code?: unknown }).exit_code;
      const durationMs = (execution as { duration_ms?: unknown }).duration_ms;

      if (
        typeof stdout !== "string" ||
        typeof stderr !== "string" ||
        typeof exitCode !== "number" ||
        typeof durationMs !== "number"
      ) {
        throw new Error("执行结果结构不符合预期。");
      }

      setResult({
        stdout,
        stderr,
        exit_code: exitCode,
        duration_ms: durationMs,
      });
    } catch (e) {
      setRunError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <div className="mt-3 overflow-hidden rounded-lg ring-1 ring-zinc-200">
      <div className="flex items-center justify-between bg-zinc-950 px-3 py-2 text-xs text-zinc-200">
        <div className="font-mono">{lang || "text"}</div>
        <div className="flex items-center gap-2">
          {canRun ? (
            <button
              type="button"
              className="rounded bg-zinc-800 px-2 py-1 text-zinc-100 hover:bg-zinc-700 disabled:opacity-60"
              onClick={run}
              disabled={isRunning}
            >
              {isRunning ? "Running..." : "Run"}
            </button>
          ) : null}
          <button
            type="button"
            className="rounded bg-zinc-800 px-2 py-1 text-zinc-100 hover:bg-zinc-700"
            onClick={copy}
          >
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      </div>

      <pre className="overflow-x-auto bg-zinc-950 p-3 text-sm text-zinc-50">
        <code className="font-mono">{props.code}</code>
      </pre>

      {runError ? (
        <div className="border-t border-zinc-200 bg-white px-3 py-2 text-sm text-red-700">
          {runError}
        </div>
      ) : null}

      {result ? (
        <div className="border-t border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="text-xs font-semibold text-zinc-600">stdout</div>
              <pre className="mt-1 max-h-40 overflow-auto rounded bg-zinc-50 p-2 font-mono text-xs text-zinc-900 ring-1 ring-zinc-200">
                {result.stdout || "(empty)"}
              </pre>
            </div>
            <div>
              <div className="text-xs font-semibold text-zinc-600">stderr</div>
              <pre className="mt-1 max-h-40 overflow-auto rounded bg-zinc-50 p-2 font-mono text-xs text-zinc-900 ring-1 ring-zinc-200">
                {result.stderr || "(empty)"}
              </pre>
            </div>
          </div>
          <div className="mt-2 flex items-center gap-4 text-xs text-zinc-600">
            <div>exit_code: {result.exit_code}</div>
            <div>duration_ms: {result.duration_ms}</div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

