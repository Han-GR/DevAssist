import { TracesPage } from "@/components/admin/TracesPage";

export default function AdminTraces() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  return <TracesPage apiUrl={apiUrl} />;
}

