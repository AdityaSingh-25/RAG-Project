// Minimal Server-Sent-Events parser for fetch(POST).
//
// EventSource doesn't support POST bodies, so we read the response stream
// ourselves: split on blank-line frame boundaries, parse `event:` / `data:`
// lines, and dispatch to a per-event-name handler.

export type SseHandlers = Record<string, (data: unknown) => void>;

export async function consumeSse(
  res: Response,
  handlers: SseHandlers,
  signal?: AbortSignal,
): Promise<void> {
  if (!res.body) {
    throw new Error("SSE response has no body");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  const onAbort = () => {
    reader.cancel().catch(() => {});
  };
  signal?.addEventListener("abort", onAbort, { once: true });

  try {
    for (;;) {
      if (signal?.aborted) return;
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      // Frames are separated by blank lines. Handle both \n\n and \r\n\r\n.
      let idx: number;
      while ((idx = nextFrameEnd(buf)) !== -1) {
        const raw = buf.slice(0, idx).replace(/\r/g, "");
        buf = buf.slice(idx).replace(/^(\r?\n){2}/, "");
        dispatch(raw, handlers);
      }
    }
    const trailing = (buf + decoder.decode()).trim();
    if (trailing) dispatch(trailing.replace(/\r/g, ""), handlers);
  } finally {
    signal?.removeEventListener("abort", onAbort);
  }
}

function nextFrameEnd(s: string): number {
  const a = s.indexOf("\n\n");
  const b = s.indexOf("\r\n\r\n");
  if (a === -1) return b;
  if (b === -1) return a;
  return Math.min(a, b);
}

function dispatch(raw: string, handlers: SseHandlers): void {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of raw.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }
  if (dataLines.length === 0) return;
  const handler = handlers[event];
  if (!handler) return;
  const raw_data = dataLines.join("\n");
  let data: unknown;
  try {
    data = JSON.parse(raw_data);
  } catch {
    data = raw_data;
  }
  handler(data);
}
