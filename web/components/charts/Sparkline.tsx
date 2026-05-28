interface SparklineProps {
  values: number[];
  width?: number;
  height?: number;
  /** CSS color string. Defaults to currentColor so it inherits from parent. */
  stroke?: string;
  /** Optional area fill alpha 0..1. */
  fillOpacity?: number;
  /** Highlight the most recent sample with a small dot. */
  showLatest?: boolean;
  /** Below this many samples we render an empty box (avoids a flat zero-line). */
  minSamples?: number;
  className?: string;
  /** Force a y-axis lower bound (e.g. 0 for a rate sparkline). */
  yMin?: number;
  /** Force a y-axis upper bound (e.g. 1 for hit-rate sparklines). */
  yMax?: number;
}

/**
 * Minimal SVG sparkline. Zero deps, auto-scales y to data range unless yMin
 * /yMax pin the extents. Intended for inline use next to table rows.
 */
export function Sparkline({
  values,
  width = 80,
  height = 22,
  stroke = "currentColor",
  fillOpacity,
  showLatest = true,
  minSamples = 2,
  className,
  yMin,
  yMax,
}: SparklineProps) {
  if (values.length < minSamples) {
    return (
      <svg
        width={width}
        height={height}
        className={className}
        aria-hidden
      />
    );
  }
  const finite = values.filter((v) => Number.isFinite(v));
  const dataMin = finite.length ? Math.min(...finite) : 0;
  const dataMax = finite.length ? Math.max(...finite) : 0;
  const min = yMin ?? dataMin;
  const max = yMax ?? dataMax;
  const range = max - min || 1;
  const step = values.length > 1 ? width / (values.length - 1) : 0;
  const pad = 1.5;
  const yFor = (v: number) =>
    height - pad - ((v - min) / range) * (height - 2 * pad);
  const points = values.map((v, i) => [i * step, yFor(v)] as const);
  const path = points
    .map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`)
    .join(" ");
  const area =
    fillOpacity != null
      ? `${path} L${width.toFixed(2)},${height} L0,${height} Z`
      : null;
  const last = points[points.length - 1];

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={className}
      role="img"
      aria-label={`sparkline, last value ${values[values.length - 1].toFixed(2)}`}
    >
      {area ? (
        <path d={area} fill={stroke} fillOpacity={fillOpacity} stroke="none" />
      ) : null}
      <path
        d={path}
        fill="none"
        stroke={stroke}
        strokeWidth={1.25}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {showLatest ? (
        <circle cx={last[0]} cy={last[1]} r={1.7} fill={stroke} />
      ) : null}
    </svg>
  );
}
