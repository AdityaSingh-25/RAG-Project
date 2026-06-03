import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Long enough to cover a full dataset against a CPU-only local Ollama. The
// stream emits events as they happen, so the user sees progress even if the
// run takes most of an hour.
export const maxDuration = 1800;

const BACKEND = process.env.RAG_API_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.text();
  try {
    const upstream = await fetch(`${BACKEND}/eval/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      cache: "no-store",
    });

    if (!upstream.ok || !upstream.body) {
      const text = await upstream.text().catch(() => upstream.statusText);
      return new NextResponse(text, { status: upstream.status });
    }

    return new NextResponse(upstream.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "unknown";
    return NextResponse.json(
      { detail: `Cannot reach backend at ${BACKEND}: ${message}` },
      { status: 502 },
    );
  }
}
