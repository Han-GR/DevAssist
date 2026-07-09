import { MessageBubble } from "@/components/chat/MessageBubble";
import { StreamingDemo } from "@/components/chat/StreamingDemo";

export default function Home() {
  return (
    <div className="flex min-h-screen justify-center bg-zinc-50 px-4 py-12 font-sans">
      <main className="w-full max-w-3xl space-y-6">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-900">
          DevAssist Chat UI（Day 11：MessageBubble）
        </h1>

        <div className="space-y-4">
          <MessageBubble
            role="assistant"
            content={[
              "我可以用 Markdown 回答问题，比如：",
              "",
              "- 列表",
              "- `inline code`",
              "",
              "```python",
              "print('hello')",
              "```",
              "",
              "> 以及引用块",
            ].join("\n")}
          />
          <MessageBubble
            role="user"
            content={"收到。那你也可以给我一个链接吗？"}
          />
          <MessageBubble
            role="assistant"
            content={
              "可以：这是一个示例链接 [FastAPI](https://fastapi.tiangolo.com/)。"
            }
          />
        </div>

        <div className="space-y-3">
          <h2 className="text-lg font-semibold tracking-tight text-zinc-900">
            流式演示
          </h2>
          <StreamingDemo apiUrl="http://localhost:8000" />
        </div>
      </main>
    </div>
  );
}
