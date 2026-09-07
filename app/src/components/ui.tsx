import type { ReactNode } from 'react'

export function Panel({
  title, actions, children,
}: { title: string; actions?: ReactNode; children: ReactNode }) {
  return (
    <section className="panel">
      <header>
        <h2>{title}</h2>
        <span style={{ flex: 1 }} />
        {actions}
      </header>
      {children}
    </section>
  )
}

/** An empty state that says what to do, not just that there is nothing. */
export function Empty({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="empty">
      <strong>{title}</strong>
      <p>{children}</p>
    </div>
  )
}

export function Pill({ value }: { value?: string | null }) {
  if (!value) return <span className="pill neutral">—</span>
  const v = value.toLowerCase()
  const ok = ['resolved', 'succeeded', 'granted', 'committed', 'active', 'read_only']
  const warn = ['pending', 'pending_customer', 'open', 'partially_refunded', 'requested']
  const bad = ['escalated', 'denied', 'failed', 'malformed', 'expired', 'revoked', 'closed']
  const tone = ok.includes(v) ? 'ok' : bad.includes(v) ? 'bad' : warn.includes(v) ? 'warn' : 'neutral'
  return <span className={`pill ${tone}`}>{value.replace(/_/g, ' ')}</span>
}

export function Id({ value }: { value?: string | null }) {
  return <span className="id">{value ?? '—'}</span>
}

/** Logical ticks, labelled as such: this is simulated time, not a clock. */
export function Tick({ value }: { value?: number | null }) {
  return <span className="ts">{value === undefined || value === null ? '—' : `t${value}`}</span>
}

export function Money({ cents }: { cents?: number | null }) {
  if (cents === undefined || cents === null) return <span className="money">—</span>
  return <span className="money">${(cents / 100).toFixed(2)}</span>
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <span className="spinner" aria-hidden />
      <span style={{ fontSize: 12, color: 'var(--muted)' }}>{label ?? 'Working…'}</span>
    </span>
  )
}

export function Banner({ tone, children }: { tone: 'err' | 'warn' | 'info'; children: ReactNode }) {
  return <div className={`banner ${tone}`} role={tone === 'err' ? 'alert' : 'status'}>{children}</div>
}
