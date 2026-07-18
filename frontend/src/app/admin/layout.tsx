import Link from "next/link";
import type { ReactNode } from "react";

export default function AdminLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-zinc-50 px-3 py-6 font-sans sm:px-4 sm:py-10">
      <div className="mx-auto w-full max-w-5xl space-y-4 sm:space-y-6">
        <header className="flex flex-col gap-3 rounded-lg border border-zinc-200 bg-white p-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-1">
            <div className="text-lg font-semibold tracking-tight text-zinc-900">
              DevAssist Admin
            </div>
            <div className="text-sm text-zinc-600">Dashboard</div>
          </div>
          <nav className="flex flex-wrap items-center gap-3 text-sm">
            <Link
              href="/"
              className="rounded-md border border-zinc-200 bg-white px-3 py-1.5 text-zinc-800 hover:bg-zinc-100"
            >
              Chat
            </Link>
            <Link
              href="/admin/evals"
              className="rounded-md border border-zinc-200 bg-white px-3 py-1.5 text-zinc-800 hover:bg-zinc-100"
            >
              Evals
            </Link>
            <Link
              href="/admin/traces"
              className="rounded-md border border-zinc-200 bg-white px-3 py-1.5 text-zinc-800 hover:bg-zinc-100"
            >
              Traces
            </Link>
          </nav>
        </header>
        <main>{children}</main>
      </div>
    </div>
  );
}
