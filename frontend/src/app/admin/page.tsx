import Link from "next/link";

export default function AdminHome() {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4">
      <div className="space-y-2">
        <h1 className="text-lg font-semibold tracking-tight text-zinc-900">
          Admin Dashboard
        </h1>
        <p className="text-sm text-zinc-600">
          Use Evals to compare metrics and Traces to inspect Agent Thought/Action/Observation steps.
        </p>
        <div className="flex flex-wrap gap-2 pt-2">
          <Link
            href="/admin/evals"
            className="inline-flex rounded-md bg-zinc-900 px-3 py-2 text-sm font-medium text-white hover:bg-zinc-800"
          >
            Open Evals
          </Link>
          <Link
            href="/admin/traces"
            className="inline-flex rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm font-medium text-zinc-800 hover:bg-zinc-100"
          >
            Open Traces
          </Link>
        </div>
      </div>
    </div>
  );
}
