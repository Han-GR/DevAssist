import Link from "next/link";
import { AgentTraceItem, TraceDetail } from "@/components/admin/TraceDetail";

interface Props {
  params: Promise<{ run_id: string }>;
}

export default async function AdminTraceDetailPage({ params }: Props) {
  const { run_id } = await params;
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  function isAgentTraceItem(value: unknown): value is AgentTraceItem {
    if (!value || typeof value !== "object") {
      return false;
    }
    const v = value as { run_id?: unknown; agent_type?: unknown; steps?: unknown };
    return (
      typeof v.run_id === "string" &&
      typeof v.agent_type === "string" &&
      Array.isArray(v.steps)
    );
  }

  let initialTrace: unknown = null;
  try {
    const resp = await fetch(`${apiUrl}/admin/agent-traces/${run_id}`, {
      cache: "no-store",
    });
    if (resp.ok) {
      initialTrace = await resp.json();
    }
  } catch {
    initialTrace = null;
  }

  return (
    <div className="space-y-4">
      <div>
        <Link
          href="/admin/traces"
          className="text-sm text-zinc-600 hover:text-zinc-900"
        >
          ← Back to Traces
        </Link>
      </div>
      <TraceDetail
        apiUrl={apiUrl}
        runId={run_id}
        initialTrace={isAgentTraceItem(initialTrace) ? initialTrace : null}
      />
    </div>
  );
}
