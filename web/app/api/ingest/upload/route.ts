import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND = process.env.RAG_API_URL ?? "http://localhost:8000";

// Multipart proxy. We read the body as a single ArrayBuffer so the binary
// payload and its content-type boundary header are forwarded verbatim —
// `await req.text()` (as the JSON proxy does) would corrupt non-utf8 bytes
// like PDFs, and re-parsing into FormData would double the memory cost.
export async function POST(req: NextRequest) {
  const body = await req.arrayBuffer();
  const contentType = req.headers.get("content-type") ?? "application/octet-stream";
  try {
    const res = await fetch(`${BACKEND}/ingest/upload`, {
      method: "POST",
      headers: { "Content-Type": contentType },
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
