import { ChatApp } from "@/components/chat/ChatApp";

export default function Home() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  return (
    <div className="flex min-h-screen justify-center bg-zinc-50 px-4 py-10 font-sans">
      <main className="w-full max-w-3xl space-y-6">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-900">
          DevAssist Chat UI
        </h1>
        <ChatApp apiUrl={apiUrl} />
      </main>
    </div>
  );
}
