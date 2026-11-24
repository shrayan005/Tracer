import { useUsage } from "../api/queries";
import { compactTokens, providerState } from "../lib/format";

/** Sidebar meter: today's LLM tokens against the daily budget, and which provider did the work or is failing.
 *  Free-tier trouble (a provider that only answers 429) becomes visible here before it breaks a job. */
export function UsageMeter() {
  const usage = useUsage();
  if (!usage.data) return null;                       // loading or an error: the sidebar simply shows nothing
  const { tokens_today, daily_budget, resets_at, providers } = usage.data;
  const share = daily_budget ? Math.min(1, tokens_today / daily_budget) : 0;
  const level = share >= 1 ? "full" : share >= 0.8 ? "high" : "";
  const clock = (iso: string) => new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const resets = clock(resets_at);

  return (
    <div className="usage-meter" title={`Counted since midnight UTC; resets at ${resets} your time`}>
      <div className="usage-head">
        <span>Today</span>
        <span className="usage-total">
          {compactTokens(tokens_today)}{daily_budget ? ` / ${compactTokens(daily_budget)}` : ""} tokens
        </span>
      </div>
      {daily_budget > 0 && (
        <div className="usage-bar" role="progressbar" aria-label="Daily token budget used"
             aria-valuemin={0} aria-valuemax={daily_budget} aria-valuenow={tokens_today}>
          <span className={`usage-fill ${level}`} style={{ width: `${share * 100}%` }} />
        </div>
      )}
      {providers.map((p) => {
        const state = providerState(p);
        const error = p.last_error
          ? `Last failure${p.last_error_at ? ` at ${clock(p.last_error_at)}` : ""}: ${p.last_error}` : undefined;
        return (
          <div key={p.provider} className="usage-provider" title={error}>
            <span className={`usage-dot ${state}`} aria-hidden="true" />
            <span className="usage-name">{p.provider}</span>
            {p.failed_calls > 0 && (
              <span className={`usage-failed ${state}`}>{p.failed_calls} failed</span>
            )}
            <span className="usage-tokens">{compactTokens(p.tokens)}</span>
          </div>
        );
      })}
    </div>
  );
}
