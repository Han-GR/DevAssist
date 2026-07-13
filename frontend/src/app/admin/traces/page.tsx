import { AgentTraceItem, TracesPage } from "@/components/admin/TracesPage";

export default async function AdminTraces() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  let initialTraces: AgentTraceItem[] | null = null;
  try {
    const resp = await fetch(`${apiUrl}/admin/agent-traces?limit=50`, {
      cache: "no-store",
    });
    if (resp.ok) {
      const data = (await resp.json()) as unknown;
      initialTraces = Array.isArray(data) ? (data as AgentTraceItem[]) : [];
    }
  } catch {
    initialTraces = null;
  }

  return <TracesPage apiUrl={apiUrl} initialTraces={initialTraces} />;
}
