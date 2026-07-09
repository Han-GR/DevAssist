"use client";

import { useMemo, useRef, useState } from "react";

import { MessageBubble } from "@/components/chat/MessageBubble";
import { streamChat } from "@/lib/streaming";

export interface StreamingDemoProps {
  apiUrl: string;
}

export function StreamingDemo(props: StreamingDemoProps) {
  const abortRef = useRef<AbortController | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [input, setInput] = useState("你好，给我一个 Python 的异步例子。");
  const [assistantText, setAssistantText] = useState("");
  const [status, setStatus] = useState<"idle" | "streaming" | "done" | "error">(
    "idle",
  );

  const canStart = status !== "streaming";

  const systemHint = useMemo(() => {
    return [
      "这是一个最小演示组件：点击开始后，会调用后端 /chat?stream=true。",
      "如果你的后端没启动，你会看到 error 状态。",
      "",
      `apiUrl: ${props.apiUrl}`,
    ].join("\n");
  }, [props.apiUrl]);

  async function start() {
    if (!canStart) {
      return;
    }

    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setStatus("streaming");
    setAssistantText("");

    try {
      for await (const ev of streamChat({
        apiUrl: props.apiUrl,
        message: input,
        conversationId: conversationId ?? undefined,
        signal: abortRef.current.signal,
      })) {
        if (ev.type === "meta") {
          setConversationId(ev.conversation_id);
          continue;
        }
        if (ev.type === "delta") {
          setAssistantText((prev) => prev + ev.content);
          continue;
        }
        if (ev.type === "done") {
          setStatus("done");
          return;
        }
        if (ev.type === "error") {
          setStatus("error");
          setAssistantText((prev) => (prev ? prev : `错误：${ev.message}`));
          return;
        }
      }

      setStatus("done");
    } catch (err) {
      setStatus("error");
      setAssistantText(String(err));
    }
  }

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setStatus("idle");
  }

  return (
    <div className="space-y-4 rounded-2xl bg-white p-4 shadow-sm ring-1 ring-zinc-200">
      <MessageBubble role="system" content={systemHint} />

      <div className="flex flex-col gap-2">
        <label className="text-sm font-medium text-zinc-700">输入</label>
        <textarea
          className="min-h-20 rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 outline-none focus:border-zinc-400"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
      </div>

      <div className="flex items-center gap-2">
        <button
          className="rounded-xl bg-zinc-900 px-3 py-2 text-sm font-medium text-zinc-50 disabled:opacity-50"
          onClick={start}
          disabled={!canStart}
        >
          开始流式
        </button>
        <button
          className="rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm font-medium text-zinc-900"
          onClick={stop}
        >
          停止
        </button>
        <div className="text-sm text-zinc-600">
          状态：{status} {conversationId ? `（conversation_id: ${conversationId}）` : ""}
        </div>
      </div>

      <MessageBubble role="assistant" content={assistantText || "（等待中）"} />
    </div>
  );
}

