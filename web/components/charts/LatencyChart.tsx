"use client";

export interface ChartSeries {
  label: string;
  values: number[];
  color: string;
}

interface LatencyChartProps {
  series: ChartSeries[];
  /** Pixel height for the SVG. Width fills the container. */
  height?: number;
  /** Format a y-axis value for the legend / corner labels. */
  format?: (v: number) => string;
  /** Label shown at the start of the x-axis (e.g. "5 min ago"). */
  xStartLabel?: string;
  /** Label shown at the end of the x-axis (e.g. "now"). */
  xEndLabel?: string;
  /** Render when there are not enough points to draw a line. */
  emptyHint?: string;
}

/**
 * Multi-series line chart sized to its container width. We draw into a
 * fixed-coord viewBox and use preserveAspectRatio="none" + non-scaling
 * strokes so lines stay crisp at any width. Text labels live in HTML on top
 * of the SVG so they don't stretch.
 */
export function LatencyChart({
  series,
  height = 180,
  format = (v) => `${Math.round(v)}`,
  xStartLabel,
  xEndLabel,
  emptyHint = "Collecting samples…",
}: LatencyChartProps) {
  const lengths = series.map((s) => s.values.length);
  const len = lengths.length ? Math.max(...lengths) : 0;
  const hasData = len >= 2 && series.some((s) => s.values.some((v) => v > 0));

  const allValues = series
    .flatMap((s) => s.values)
    .filter((v) => Number.isFinite(v) && v >= 0);
  const dataMax = allValues.length ? Math.max(...allValues) : 0;
  // Round up so the top gridline lands on a clean-ish number and leaves a
  // little headroom above the highest point.
  const yMax = niceCeil(dataMax * 1.1 || 1);
  const yMid = yMax / 2;

  // Fixed viewBox; SVG fills container width and is stretched by preserveAspectRatio.
  const VB_W = 600;
  const VB_H = 180;
  const PAD_TOP = 8;
  const PAD_BOTTOM = 8;
  const plotH = VB_H - PAD_TOP - PAD_BOTTOM;
  const xFor = (i: number) => (len <= 1 ? 0 : (i / (len - 1)) * VB_W);
  const yFor = (v: number) => PAD_TOP + (1 - v / yMax) * plotH;

  return (
    <div className="relative">
      <div
        className="flex items-center gap-3 text-[11px]"
        aria-label="chart legend"
      >
        {series.map((s) => (
          <span key={s.label} className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-1.5 w-3 rounded-full"
              style={{ background: s.color }}
            />
            <span className="text-muted">{s.label}</span>
          </span>
        ))}
      </div>

      <div className="relative mt-2" style={{ height }}>
        <svg
          width="100%"
          height={height}
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          preserveAspectRatio="none"
          className="block"
          aria-hidden={!hasData}
        >
          <line
            x1={0}
            x2={VB_W}
            y1={yFor(yMax)}
            y2={yFor(yMax)}
            stroke="var(--color-border)"
            strokeDasharray="2 4"
            vectorEffect="non-scaling-stroke"
          />
          <line
            x1={0}
            x2={VB_W}
            y1={yFor(yMid)}
            y2={yFor(yMid)}
            stroke="var(--color-border)"
            strokeDasharray="2 4"
            vectorEffect="non-scaling-stroke"
          />
          <line
            x1={0}
            x2={VB_W}
            y1={yFor(0)}
            y2={yFor(0)}
            stroke="var(--color-border)"
            vectorEffect="non-scaling-stroke"
          />

          {hasData
            ? series.map((s) => (
                <SeriesPath
                  key={s.label}
                  values={s.values}
                  xFor={xFor}
                  yFor={yFor}
                  color={s.color}
                />
              ))
            : null}
        </svg>

        <div className="text-subtle pointer-events-none absolute top-0 right-1 font-mono text-[10px]">
          {format(yMax)}
        </div>
        <div className="text-subtle pointer-events-none absolute top-1/2 right-1 -translate-y-1/2 font-mono text-[10px]">
          {format(yMid)}
        </div>
        <div className="text-subtle pointer-events-none absolute right-1 bottom-0 font-mono text-[10px]">
          {format(0)}
        </div>

        {!hasData ? (
          <div className="text-subtle pointer-events-none absolute inset-0 flex items-center justify-center text-xs">
            {emptyHint}
          </div>
        ) : null}
      </div>

      {(xStartLabel || xEndLabel) && (
        <div className="text-subtle mt-1.5 flex justify-between font-mono text-[10px]">
          <span>{xStartLabel ?? ""}</span>
          <span>{xEndLabel ?? ""}</span>
        </div>
      )}
    </div>
  );
}

function SeriesPath({
  values,
  xFor,
  yFor,
  color,
}: {
  values: number[];
  xFor: (i: number) => number;
  yFor: (v: number) => number;
  color: string;
}) {
  if (values.length < 2) return null;
  const d = values
    .map((v, i) => `${i === 0 ? "M" : "L"}${xFor(i).toFixed(2)},${yFor(v).toFixed(2)}`)
    .join(" ");
  return (
    <path
      d={d}
      fill="none"
      stroke={color}
      strokeWidth={1.6}
      strokeLinejoin="round"
      strokeLinecap="round"
      vectorEffect="non-scaling-stroke"
    />
  );
}

/** Round up to a friendly scale: 10, 25, 50, 100, 250, 500, 1000… */
function niceCeil(v: number): number {
  if (v <= 0) return 1;
  const pow = Math.pow(10, Math.floor(Math.log10(v)));
  const norm = v / pow;
  let nice: number;
  if (norm <= 1) nice = 1;
  else if (norm <= 2) nice = 2;
  else if (norm <= 2.5) nice = 2.5;
  else if (norm <= 5) nice = 5;
  else nice = 10;
  return nice * pow;
}
