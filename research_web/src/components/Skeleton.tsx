/** Grey shimmering placeholders while data loads: the page keeps its shape instead of jumping. */
export function Skeleton({ width = "100%", height = 12 }: { width?: number | string; height?: number }) {
  return <span className="skeleton" style={{ width, height }} aria-hidden="true" />;
}

export function SkeletonRows({ rows = 6, columns = 4 }: { rows?: number; columns?: number }) {
  return (
    <div className="skeleton-rows" aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, r) => (
        <div key={r} className="skeleton-row">
          {Array.from({ length: columns }, (_, c) => (
            <Skeleton key={c} width={c === 0 ? `${55 + ((r * 7) % 30)}%` : `${10 + ((r + c) % 3) * 4}%`} />
          ))}
        </div>
      ))}
    </div>
  );
}

export function SkeletonList({ rows = 5 }: { rows?: number }) {
  return (
    <div className="skeleton-list" aria-busy="true">
      {Array.from({ length: rows }, (_, i) => <Skeleton key={i} width={`${60 + ((i * 13) % 35)}%`} height={10} />)}
    </div>
  );
}
