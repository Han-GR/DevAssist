export interface SseMessage {
  event: string | null;
  data: string;
}

function splitLines(chunk: string): string[] {
  return chunk.replaceAll("\r\n", "\n").replaceAll("\r", "\n").split("\n");
}

export async function* parseSseStream(
  stream: ReadableStream<Uint8Array>,
  options?: { signal?: AbortSignal },
): AsyncGenerator<SseMessage> {
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");

  let buffer = "";
  let currentEvent: string | null = null;
  let dataLines: string[] = [];

  const emitIfReady = async function* (): AsyncGenerator<SseMessage> {
    if (dataLines.length === 0) {
      return;
    }
    const data = dataLines.join("\n");
    const event = currentEvent;
    dataLines = [];
    currentEvent = null;
    yield { event, data };
  };

  while (true) {
    if (options?.signal?.aborted) {
      try {
        await reader.cancel();
      } catch {}
      return;
    }

    const { done, value } = await reader.read();
    if (done) {
      if (dataLines.length > 0) {
        yield* emitIfReady();
      }
      return;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = splitLines(buffer);

    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line === "") {
        yield* emitIfReady();
        continue;
      }

      if (line.startsWith("event:")) {
        currentEvent = line.slice("event:".length).trim() || null;
        continue;
      }

      if (line.startsWith("data:")) {
        dataLines.push(line.slice("data:".length).trimStart());
        continue;
      }
    }
  }
}

export interface ChatStreamMeta {
  type: "meta";
  conversation_id: string;
}

export interface ChatStreamDelta {
  type: "delta";
  content: string;
}

export interface ChatStreamDone {
  type: "done";
}

export interface ChatStreamError {
  type: "error";
  message: string;
}

export type ChatStreamEvent =
  | ChatStreamMeta
  | ChatStreamDelta
  | ChatStreamDone
  | ChatStreamError;

function safeJsonParse(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

async function parseErrorMessage(resp: Response): Promise<string> {
  const requestId = resp.headers.get("x-request-id");
  const contentType = resp.headers.get("content-type") ?? "";

  if (contentType.includes("application/json")) {
    try {
      const body = (await resp.json()) as unknown;
      if (body && typeof body === "object") {
        const maybeError = (body as { error?: unknown }).error;
        if (maybeError && typeof maybeError === "object") {
          const message = (maybeError as { message?: unknown }).message;
          if (typeof message === "string" && message.trim().length > 0) {
            return requestId ? `${message} (request_id: ${requestId})` : message;
          }
        }

        const message = (body as { message?: unknown }).message;
        if (typeof message === "string" && message.trim().length > 0) {
          return requestId ? `${message} (request_id: ${requestId})` : message;
        }
      }
    } catch {}
  }

  const fallback = resp.statusText
    ? `HTTP ${resp.status} ${resp.statusText}`
    : `HTTP ${resp.status}`;
  return requestId ? `${fallback} (request_id: ${requestId})` : fallback;
}

export async function* streamChat(options: {
  apiUrl: string;
  message: string;
  conversationId?: string;
  history?: Array<{ role: "system" | "user" | "assistant"; content: string }>;
  signal?: AbortSignal;
}): AsyncGenerator<ChatStreamEvent> {
  const url = new URL("/chat", options.apiUrl);
  url.searchParams.set("stream", "true");

  const resp = await fetch(url.toString(), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      conversation_id: options.conversationId ?? null,
      message: options.message,
      history: options.history ?? [],
    }),
    signal: options.signal,
  });

  if (!resp.ok) {
    const message = await parseErrorMessage(resp);
    yield { type: "error", message };
    return;
  }

  if (!resp.body) {
    yield { type: "error", message: "Response body is empty" };
    return;
  }

  for await (const sse of parseSseStream(resp.body, { signal: options.signal })) {
    const parsed = safeJsonParse(sse.data);
    if (!parsed || typeof parsed !== "object") {
      continue;
    }

    const typeValue = (parsed as { type?: unknown }).type;
    if (typeValue === "meta") {
      const id = (parsed as { conversation_id?: unknown }).conversation_id;
      if (typeof id === "string") {
        yield { type: "meta", conversation_id: id };
      }
      continue;
    }

    if (typeValue === "delta") {
      const content = (parsed as { content?: unknown }).content;
      if (typeof content === "string") {
        yield { type: "delta", content };
      }
      continue;
    }

    if (typeValue === "done") {
      yield { type: "done" };
      return;
    }

    if (typeValue === "error") {
      const message = (parsed as { message?: unknown }).message;
      if (typeof message === "string") {
        yield { type: "error", message };
      } else {
        yield { type: "error", message: "Unknown stream error" };
      }
      return;
    }
  }
}
