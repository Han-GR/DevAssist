import Link from "next/link";

export default function AdminHome() {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4">
      <div className="space-y-2">
        <h1 className="text-lg font-semibold tracking-tight text-zinc-900">
          Admin Dashboard
        </h1>
        <p className="text-sm text-zinc-600">
          Use the traces viewer to inspect Agent Thought/Action/Observation steps.
        </p>
        <div className="pt-2">
          <Link
            href="/admin/traces"
            className="inline-flex rounded-md bg-zinc-900 px-3 py-2 text-sm font-medium text-white hover:bg-zinc-800"
          >
            Open Traces
          </Link>
        </div>
      </div>
    </div>
  );
}

