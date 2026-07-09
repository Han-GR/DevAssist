export interface ApiClientOptions {
  apiUrl: string;
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
  request_id?: string;
}

export class ApiError extends Error {
  public status: number;
  public code: string | null;
  public requestId: string | null;
  public details: unknown;

  public constructor(options: {
    message: string;
    status: number;
    code?: string | null;
    requestId?: string | null;
    details?: unknown;
  }) {
    super(options.message);
    this.name = "ApiError";
    this.status = options.status;
    this.code = options.code ?? null;
    this.requestId = options.requestId ?? null;
    this.details = options.details;
  }
}

async function parseJsonSafely(resp: Response): Promise<unknown | null> {
  const contentType = resp.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    return null;
  }
  try {
    return await resp.json();
  } catch {
    return null;
  }
}

async function requestJson<T>(
  options: ApiClientOptions,
  input: { path: string; method: "GET" | "POST"; body?: unknown },
): Promise<T> {
  const url = new URL(input.path, options.apiUrl);

  const resp = await fetch(url.toString(), {
    method: input.method,
    headers: {
      "content-type": "application/json",
    },
    body: input.body ? JSON.stringify(input.body) : undefined,
  });

  if (resp.ok) {
    const data = await parseJsonSafely(resp);
    return data as T;
  }

  const maybeBody = (await parseJsonSafely(resp)) as ApiErrorBody | null;
  const requestId = resp.headers.get("x-request-id") ?? maybeBody?.request_id ?? null;

  if (resp.status >= 500) {
    throw new ApiError({
      message: "Service unavailable. Please try again.",
      status: resp.status,
      code: maybeBody?.error?.code ?? null,
      requestId,
      details: maybeBody?.error?.details,
    });
  }

  if (maybeBody?.error?.message) {
    throw new ApiError({
      message: maybeBody.error.message,
      status: resp.status,
      code: maybeBody.error.code ?? null,
      requestId,
      details: maybeBody.error.details,
    });
  }

  throw new ApiError({
    message: `Request failed (HTTP ${resp.status})`,
    status: resp.status,
    requestId,
  });
}

export interface HealthResponse {
  status: string;
}

export interface ChatHistoryMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  conversation_id?: string | null;
  message: string;
  history?: ChatHistoryMessage[];
}

export interface ChatResponse {
  conversation_id: string;
  reply: string;
}

export function createApiClient(options: ApiClientOptions) {
  return {
    async health(): Promise<HealthResponse> {
      return await requestJson<HealthResponse>(options, {
        path: "/health",
        method: "GET",
      });
    },
    async chat(payload: ChatRequest): Promise<ChatResponse> {
      return await requestJson<ChatResponse>(options, {
        path: "/chat",
        method: "POST",
        body: payload,
      });
    },
  };
}

