import Link from "next/link";
import { TraceDetail } from "@/components/admin/TraceDetail";

interface Props {
  params: Promise<{ run_id: string }>;
}

export default async function AdminTraceDetailPage({ params }: Props) {
  const { run_id } = await params;
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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
      <TraceDetail apiUrl={apiUrl} runId={run_id} />
    </div>
  );
}
