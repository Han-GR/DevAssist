"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { MessageBubble } from "@/components/chat/MessageBubble";
import { streamChat } from "@/lib/streaming";

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

export function ChatApp(props: ChatAppProps) {
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>(() => [
    {
      id: makeId(),
      role: "system",
      content:
        "这是一个最小聊天页：会直接调用后端 /chat?stream=true，收到 delta 就更新最后一条 assistant 消息。",
    },
  ]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState<
    "idle" | "streaming" | "error" | "done"
  >("idle");
  const [error, setError] = useState<string | null>(null);

  const canSend = status !== "streaming" && input.trim().length > 0;
  const history = useMemo(() => {
    return messages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({ role: m.role, content: m.content }));
  }, [messages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, status]);

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
          abortRef.current = null;
          return;
        }
      }

      setStatus("done");
      abortRef.current = null;
    } catch (err) {
      setStatus("error");
      setError(String(err));
      abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setStatus("idle");
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

  return (
    <div className="flex min-h-[70vh] flex-col overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-zinc-200">
      <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3">
        <div className="flex flex-col">
          <div className="text-sm font-medium text-zinc-900">DevAssist Chat</div>
          <div className="text-xs text-zinc-500">
            {conversationId ? `conversation_id: ${conversationId}` : "未建立会话"}
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs text-zinc-600">
          <span>状态：{status}</span>
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
        {messages.map((m) => (
          <MessageBubble key={m.id} role={m.role} content={m.content} />
        ))}
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
          />

          <div className="flex items-center justify-between">
            <div className="text-xs text-zinc-500">
              apiUrl: {props.apiUrl}
            </div>
            <button
              className="rounded-xl bg-zinc-900 px-4 py-2 text-sm font-medium text-zinc-50 disabled:opacity-50"
              onClick={() => void send()}
              disabled={!canSend}
            >
              发送
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

