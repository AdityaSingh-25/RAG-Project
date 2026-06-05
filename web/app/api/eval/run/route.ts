import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Eval runs are slow (graph invocation per case). 30 minutes covers a full
// dataset against a CPU-only local Ollama. The browser's own keepalive
// timeout will usually fire first; that's fine — backend keeps going and
// the user can re-fetch.
export const maxDuration = 1800;

const BACKEND = process.env.RAG_API_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.text();
  try {
    const res = await fetch(`${BACKEND}/eval/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    const text = await res.text();
    return new NextResponse(text, {
      status: res.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "unknown";
    return NextResponse.json(
      { detail: `Cannot reach backend at ${BACKEND}: ${message}` },
      { status: 502 },
    );
  }
}
