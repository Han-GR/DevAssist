"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { MessageBubble } from "@/components/chat/MessageBubble";
import { ChatStreamStep, streamChat } from "@/lib/streaming";

export interface ChatAppProps {
  apiUrl: string;
}

export interface ChatMessage {
  id: string;
  role: "system" | "user" | "assistant";
  content: string;
}

function makeId(): string {
  return crypto.randomUUID();
}

function formatAgentStep(step: ChatStreamStep): string {
  const parts: string[] = [];
  parts.push(`### Agent Step ${step.step_index}`);
  if (step.thought.trim().length > 0) {
    parts.push("");
    parts.push("**Thought**");
    parts.push(step.thought);
  }

  if (step.action_raw.trim().length > 0) {
    parts.push("");
    parts.push("**Action**");
    parts.push(step.action_raw);
  }

  if (step.tool_name) {
    parts.push("");
    parts.push(`**Tool**: ${step.tool_name}`);
  }

  if (step.tool_args) {
    parts.push("");
    parts.push("**Tool Args**");
    parts.push("```json");
    parts.push(JSON.stringify(step.tool_args, null, 2));
    parts.push("```");
  }

  if (step.observation !== null && step.observation !== undefined) {
    parts.push("");
    parts.push("**Observation**");
    const obs =
      typeof step.observation === "string"
        ? step.observation
        : JSON.stringify(step.observation, null, 2);
    parts.push("```json");
    parts.push(obs.length > 4000 ? `${obs.slice(0, 4000)}\n...<truncated>` : obs);
    parts.push("```");
  }

  if (step.error) {
    parts.push("");
    parts.push("**Error**");
    parts.push(step.error);
  }

  parts.push("");
  parts.push(`latency_ms: ${step.latency_ms}`);
  return parts.join("\n");
}

export interface ToastState {
  message: string;
  variant: "success" | "error";
}

export function Toast(props: {
  toast: ToastState;
  onClose: () => void;
  autoCloseMs?: number;
}) {
  useEffect(() => {
    const timeout = window.setTimeout(
      () => props.onClose(),
      props.autoCloseMs ?? 4000,
    );
    return () => window.clearTimeout(timeout);
  }, [props]);

  const classes =
    props.toast.variant === "error"
      ? "border-red-200 bg-red-50 text-red-900"
      : "border-emerald-200 bg-emerald-50 text-emerald-900";

  return (
    <div
      className={[
        "pointer-events-auto flex max-w-[min(520px,calc(100vw-2rem))] items-start gap-3 rounded-2xl border px-4 py-3 shadow-sm",
        classes,
      ].join(" ")}
      role="status"
    >
      <div className="text-sm leading-6">{props.toast.message}</div>
      <button
        className="ml-auto shrink-0 rounded-lg px-2 py-1 text-xs font-medium opacity-80 hover:opacity-100"
        onClick={props.onClose}
        type="button"
      >
        关闭
      </button>
    </div>
  );
}

export function ChatApp(props: ChatAppProps) {
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>(() => []);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState<
    "idle" | "streaming" | "error" | "done"
  >("idle");
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<ToastState | null>(null);

  const canSend = status !== "streaming" && input.trim().length > 0;
  const isEmpty = messages.length === 0;
  const history = useMemo(() => {
    return messages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({ role: m.role, content: m.content }));
  }, [messages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, status]);

  function showToast(next: ToastState) {
    setToast(next);
  }

  async function send() {
    if (!canSend) {
      return;
    }

    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setError(null);
    setStatus("streaming");

    const userText = input.trim();
    setInput("");

    const userMessageId = makeId();
    const assistantMessageId = makeId();

    setMessages((prev) => [
      ...prev,
      { id: userMessageId, role: "user", content: userText },
      { id: assistantMessageId, role: "assistant", content: "" },
    ]);

    try {
      for await (const ev of streamChat({
        apiUrl: props.apiUrl,
        message: userText,
        conversationId: conversationId ?? undefined,
        history,
        signal: abortRef.current.signal,
      })) {
        if (ev.type === "meta") {
          setConversationId(ev.conversation_id);
          continue;
        }

        if (ev.type === "step") {
          const stepMessage: ChatMessage = {
            id: makeId(),
            role: "system",
            content: formatAgentStep(ev),
          };

          setMessages((prev) => {
            const index = prev.findIndex((m) => m.id === assistantMessageId);
            if (index < 0) {
              return [...prev, stepMessage];
            }
            const next = [...prev];
            next.splice(index, 0, stepMessage);
            return next;
          });
          continue;
        }

        if (ev.type === "delta") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessageId
                ? { ...m, content: m.content + ev.content }
                : m,
            ),
          );
          continue;
        }

        if (ev.type === "done") {
          setStatus("done");
          abortRef.current = null;
          return;
        }

        if (ev.type === "error") {
          setStatus("error");
          setError(ev.message);
          showToast({ variant: "error", message: ev.message });
          abortRef.current = null;
          return;
        }
      }

      setStatus("done");
      abortRef.current = null;
    } catch (err) {
      const maybeAbort = err as { name?: unknown };
      if (maybeAbort?.name === "AbortError") {
        setStatus("idle");
        abortRef.current = null;
        return;
      }
      setStatus("error");
      setError(String(err));
      showToast({ variant: "error", message: String(err) });
      abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setStatus("idle");
    showToast({ variant: "success", message: "已停止生成" });
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key !== "Enter") {
      return;
    }
    if (e.shiftKey) {
      return;
    }
    e.preventDefault();
    void send();
  }

  const showThinking =
    status === "streaming" &&
    messages.at(-1)?.role === "assistant" &&
    (messages.at(-1)?.content ?? "").length === 0;

  return (
    <div className="relative flex h-[75vh] flex-col overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-zinc-200 sm:h-[80vh]">
      <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3">
        <div className="flex flex-col">
          <div className="text-sm font-medium text-zinc-900">DevAssist Chat</div>
          <div className="text-xs text-zinc-500">
            {conversationId ? `conversation_id: ${conversationId}` : "未建立会话"}
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs text-zinc-600">
          <span className="flex items-center gap-2">
            <span>状态：{status}</span>
            {status === "streaming" ? (
              <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-zinc-300 border-t-zinc-700" />
            ) : null}
          </span>
          <button
            className="rounded-lg border border-zinc-200 bg-white px-2 py-1 font-medium text-zinc-900 disabled:opacity-50"
            onClick={stop}
            disabled={status !== "streaming"}
          >
            停止
          </button>
        </div>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto bg-zinc-50 px-4 py-4">
        {isEmpty ? (
          <div className="flex h-full items-center justify-center">
            <div className="max-w-md rounded-2xl border border-zinc-200 bg-white p-5 text-sm text-zinc-700 shadow-sm">
              <div className="text-base font-semibold text-zinc-900">
                先问一个问题试试
              </div>
              <div className="mt-2 leading-7 text-zinc-600">
                Enter 发送，Shift+Enter 换行。支持 Markdown 渲染与安全过滤。
              </div>
              <div className="mt-4 grid gap-2 text-xs text-zinc-600 sm:grid-cols-2">
                <div className="rounded-xl bg-zinc-50 px-3 py-2">
                  “FastAPI 的依赖注入怎么用？”
                </div>
                <div className="rounded-xl bg-zinc-50 px-3 py-2">
                  “帮我写一个健康检查接口”
                </div>
              </div>
            </div>
          </div>
        ) : (
          messages.map((m) => (
            <MessageBubble key={m.id} role={m.role} content={m.content} />
          ))
        )}
        {showThinking ? (
          <MessageBubble role="assistant" content="正在思考..." />
        ) : null}
        {error ? (
          <MessageBubble role="system" content={`错误：${error}`} />
        ) : null}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-zinc-200 bg-white p-4">
        <div className="flex flex-col gap-3">
          <textarea
            className="min-h-20 rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 outline-none focus:border-zinc-400"
            placeholder="输入消息，Enter 发送，Shift+Enter 换行"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={status === "streaming"}
          />

          <div className="flex items-center justify-between">
            <div className="text-xs text-zinc-500">
              apiUrl: {props.apiUrl}
            </div>
            <button
              className="rounded-xl bg-zinc-900 px-4 py-2 text-sm font-medium text-zinc-50 disabled:opacity-50"
              onClick={() => void send()}
              disabled={!canSend}
              type="button"
            >
              {status === "streaming" ? (
                <span className="flex items-center gap-2">
                  <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-zinc-50 border-t-transparent" />
                  发送中
                </span>
              ) : (
                "发送"
              )}
            </button>
          </div>
        </div>
      </div>

      {toast ? (
        <div className="pointer-events-none absolute right-4 top-4 z-10 flex justify-end">
          <Toast toast={toast} onClose={() => setToast(null)} />
        </div>
      ) : null}
    </div>
  );
}
