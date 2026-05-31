import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND = process.env.RAG_API_URL ?? "http://localhost:8000";

export async function GET(req: NextRequest) {
  const path = req.nextUrl.searchParams.get("path");
  if (!path) {
    return NextResponse.json(
      { detail: "Missing required query parameter 'path'" },
      { status: 400 },
    );
  }
  const url = `${BACKEND}/corpus/source?path=${encodeURIComponent(path)}`;
  try {
    const res = await fetch(url, { cache: "no-store" });
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
