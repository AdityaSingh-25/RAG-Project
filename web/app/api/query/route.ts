import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
// Don't cache POSTs anyway, but be explicit so dev mode doesn't surprise.
export const dynamic = "force-dynamic";

const BACKEND = process.env.RAG_API_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.text();
  try {
    const res = await fetch(`${BACKEND}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    const text = await res.text();
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    // 429 backpressure responses include Retry-After. The browser cares,
    // but only if we forward it through the proxy.
    const retryAfter = res.headers.get("retry-after");
    if (retryAfter) headers["Retry-After"] = retryAfter;
    return new NextResponse(text, {
      status: res.status,
      headers,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "unknown";
    return NextResponse.json(
      { detail: `Cannot reach backend at ${BACKEND}: ${message}` },
      { status: 502 },
    );
  }
}
