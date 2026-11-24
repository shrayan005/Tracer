/** Tracer's mark: a line traced back to its source (the dot). Plain SVG, inherits the text colour. */
export function Mark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" className="mark-logo">
      <path d="M3.5 18.5c4.5 0 5-13 9.5-13s4 6.5 7.5 6.5" fill="none" stroke="currentColor" strokeWidth="2.1"
            strokeLinecap="round" />
      <circle cx="20.5" cy="12" r="2.6" fill="var(--brand)" />
      <circle cx="3.5" cy="18.5" r="1.7" fill="currentColor" />
    </svg>
  );
}

export function Logo({ size = 22 }: { size?: number }) {
  return (
    <span className="logo">
      <Mark size={size} />
      <span className="logo-word">Tracer</span>
    </span>
  );
}
