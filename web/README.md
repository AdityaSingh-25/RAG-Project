# RAG Engine — Web

Next.js 15 + Tailwind 4 frontend for the FastAPI engine.

## Layout

```
web/
├── app/
│   ├── api/        # Proxy route handlers (forward to FastAPI)
│   ├── globals.css # Tailwind 4 + theme tokens
│   ├── layout.tsx  # Root layout + theme bootstrap (no flash)
│   └── page.tsx    # Query workspace
├── components/
│   └── ThemeToggle.tsx
└── lib/
    ├── api.ts      # fetch helpers
    ├── types.ts    # /query response shape (mirrors api/main.py)
    └── utils.ts    # cn(), formatMs()
```

## Run locally

```bash
cd web
cp .env.local.example .env.local   # set RAG_API_URL if different
npm install
npm run dev
```

Then open <http://localhost:3000>. The backend must be reachable at
`RAG_API_URL` (default `http://localhost:8000`) — start it with
`uvicorn rag_engine.api.main:app --reload` from the repo root.

## How the proxy works

Browser → `/api/query` (Next route handler) → `${RAG_API_URL}/query`. The
backend URL never reaches the browser, and CORS becomes a non-issue. The
backend also has `localhost:3000` and `127.0.0.1:3000` in its
`CORSMiddleware` allow-list, so direct `fetch` from the page would work
too — the proxy is the safer default.

## Theming

Colours are CSS variables defined in `globals.css` via Tailwind 4's
`@theme` directive. `html.dark` swaps the palette. The dark/light choice
is restored from `localStorage["rag-theme"]` in an inline script in
`layout.tsx` so the page never flashes the wrong scheme.

## What ships in this phase

- Query box + answer with citations and warnings.
- Per-claim grounding panel (sentence list with support bars).
- Pipeline trace card (per-node timings from the backend's
  `pipeline_trace` field).
- Runtime metrics card (totals + average latency from `/metrics`).
- Dark mode.

Streaming, an ingest UI, and a dedicated `/metrics` dashboard with full
percentile charts land in subsequent phases.
