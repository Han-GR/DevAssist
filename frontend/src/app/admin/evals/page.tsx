import { EvalsPage, EvalResultItem } from "@/components/admin/EvalsPage";

export default async function AdminEvals() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  let initialResults: EvalResultItem[] | null = null;
  try {
    const resp = await fetch(`${apiUrl}/admin/eval-results?limit=200`, {
      cache: "no-store",
    });
    if (resp.ok) {
      const data = (await resp.json()) as unknown;
      initialResults = Array.isArray(data) ? (data as EvalResultItem[]) : [];
    }
  } catch {
    initialResults = null;
  }

  return <EvalsPage apiUrl={apiUrl} initialResults={initialResults} />;
}
