import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import rehypeSanitize from "rehype-sanitize";
import remarkGfm from "remark-gfm";

export interface MessageBubbleProps {
  role: "system" | "user" | "assistant";
  content: string;
}

function getBubbleStyle(role: MessageBubbleProps["role"]): {
  wrapper: string;
  bubble: string;
} {
  if (role === "user") {
    return {
      wrapper: "flex w-full justify-end",
      bubble:
        "max-w-[85%] rounded-2xl bg-zinc-900 px-4 py-3 text-zinc-50 shadow-sm",
    };
  }

  if (role === "system") {
    return {
      wrapper: "flex w-full justify-center",
      bubble:
        "max-w-[85%] rounded-2xl bg-zinc-100 px-4 py-3 text-sm text-zinc-600",
    };
  }

  return {
    wrapper: "flex w-full justify-start",
    bubble:
      "max-w-[85%] rounded-2xl bg-white px-4 py-3 text-zinc-900 shadow-sm ring-1 ring-zinc-200",
  };
}

export function MessageBubble(props: MessageBubbleProps) {
  const styles = getBubbleStyle(props.role);
  const components: Components = {
    p: ({ children }) => (
      <p className="leading-7 [&:not(:first-child)]:mt-3">{children}</p>
    ),
    a: ({ children, href }) => (
      <a
        href={href}
        className="underline underline-offset-2"
        target="_blank"
        rel="noreferrer"
      >
        {children}
      </a>
    ),
    ul: ({ children }) => (
      <ul className="mt-3 list-disc space-y-1 pl-5">{children}</ul>
    ),
    ol: ({ children }) => (
      <ol className="mt-3 list-decimal space-y-1 pl-5">{children}</ol>
    ),
    li: ({ children }) => <li className="leading-7">{children}</li>,
    blockquote: ({ children }) => (
      <blockquote className="mt-3 border-l-2 border-zinc-300 pl-4 text-zinc-700">
        {children}
      </blockquote>
    ),
    h1: ({ children }) => (
      <h1 className="text-xl font-semibold leading-8">{children}</h1>
    ),
    h2: ({ children }) => (
      <h2 className="text-lg font-semibold leading-8">{children}</h2>
    ),
    h3: ({ children }) => (
      <h3 className="text-base font-semibold leading-7">{children}</h3>
    ),
    code: ({ children, className }) => {
      if (!className) {
        return (
          <code className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-[0.9em] text-zinc-900">
            {children}
          </code>
        );
      }

      return (
        <code className={["font-mono", className].filter(Boolean).join(" ")}>
          {children}
        </code>
      );
    },
    pre: ({ children }) => (
      <pre className="mt-3 overflow-x-auto rounded-lg bg-zinc-950 p-3 text-sm text-zinc-50">
        {children}
      </pre>
    ),
  };

  return (
    <div className={styles.wrapper}>
      <div className={styles.bubble}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeSanitize]}
          components={components}
        >
          {props.content}
        </ReactMarkdown>
      </div>
    </div>
  );
}
