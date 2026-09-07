import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, api } from './api'
import { ActionPanel } from './components/Actions'
import { Approvals, BillingRecords, Inbox, PolicyPanel, TicketDetail } from './components/Records'
import { Reviewer } from './components/Reviewer'
import { Banner, Empty, Id, Panel, Pill, Spinner, Tick } from './components/ui'
import type { Demo, Session } from './types'

type Tab = 'inbox' | 'billing' | 'approvals' | 'policy' | 'history'

export default function App() {
  const reviewing = new URLSearchParams(window.location.search).has('review')
  return (
    <>
      <header className="masthead">
        <h1>Northwind Support &amp; Billing</h1>
        <span className="env">{reviewing ? 'reviewer · privileged' : 'operations'}</span>
        <span className="spacer" />
        {reviewing
          ? <a href="/">← Back to workspace</a>
          : <a href="/?review">Reviewer replay →</a>}
      </header>
      {reviewing ? <Reviewer /> : <Workspace />}
    </>
  )
}

function Workspace() {
  const [demos, setDemos] = useState<Demo[] | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [tab, setTab] = useState<Tab>('inbox')
  const [selectedTicket, setSelected] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [starting, setStarting] = useState<string | null>(null)
  const [error, setError] = useState<{ message: string; detail?: unknown } | null>(null)
  const [fatal, setFatal] = useState<string | null>(null)

  // One in-flight submission at a time. Guards against a double click or a
  // second handler firing before the first response lands; the server-side
  // submission token guards against a network retry.
  const inFlight = useRef(false)

  useEffect(() => {
    api.demos()
      .then(setDemos)
      .catch((e: ApiError) => setFatal(e.message))
  }, [])

  const start = useCallback(async (demoId: string) => {
    setStarting(demoId); setError(null)
    try {
      const created = await api.createSession(demoId)
      setSession(created); setSelected(null); setTab('inbox')
    } catch (e) {
      setError({ message: (e as ApiError).message, detail: (e as ApiError).detail })
    } finally {
      setStarting(null)
    }
  }, [])

  const submit = useCallback(async (action: Record<string, unknown>) => {
    if (!session || inFlight.current) return
    inFlight.current = true
    setBusy(true); setError(null)
    // Generated once per gesture. A retried POST carries the same token, and
    // the server returns the original record instead of acting again.
    const submissionId = crypto.randomUUID()
    try {
      const res = await api.act(session.session_id, action, submissionId)
      setSession(res.session)
    } catch (e) {
      const err = e as ApiError
      setError({ message: err.message, detail: err.detail })
    } finally {
      inFlight.current = false
      setBusy(false)
    }
  }, [session])

  const reset = useCallback(async () => {
    if (!session) return
    setBusy(true)
    try {
      setSession(await api.reset(session.session_id))
      setSelected(null); setError(null)
    } finally {
      setBusy(false)
    }
  }, [session])

  if (fatal) {
    return (
      <div style={{ padding: 24, maxWidth: 620 }}>
        <Banner tone="err">
          <strong>Cannot reach the workspace API.</strong>
          <div style={{ marginTop: 6, fontSize: 13 }}>
            {fatal} Start it with <code>uv run cerl serve</code>.
          </div>
        </Banner>
      </div>
    )
  }
  if (!demos) return <div style={{ padding: 24 }}><Spinner label="Loading demonstrations" /></div>

  const ticket = session?.observed.tickets.find((t) => t.id === selectedTicket)
  const o = session?.observed

  return (
    <div className="layout">
      <div>
        {session && (
          <Panel
            title="Session"
            actions={
              <button className="btn small" onClick={reset} disabled={busy} data-testid="reset-session">
                Reset
              </button>
            }
          >
            <div className="body">
              <dl className="meter">
                <div><dt>Logical time</dt><dd data-testid="logical-time">t{session.logical_time}</dd></div>
                <div><dt>Steps used</dt><dd>{session.step_index}/{session.budget_steps}</dd></div>
                <div><dt>Remaining</dt><dd>{session.steps_remaining}</dd></div>
              </dl>
              <p style={{ fontSize: 12.5, color: 'var(--muted)', marginTop: 10 }}>
                Logical time advances only when you dispatch an action. Reading
                this page, waiting, or refreshing does not move it.
              </p>
              {session.done && (
                <Banner tone="info">
                  Episode ended{session.declared_outcome ? ` — “${session.declared_outcome}”` : ''}.
                </Banner>
              )}
            </div>
          </Panel>
        )}

        <Panel title="Demonstrations">
          {demos.map((d) => (
            <div className={`demo-card ${session?.demo_id === d.demo_id ? 'active' : ''}`} key={d.demo_id}>
              <h3>{d.title}</h3>
              <p>{d.summary}</p>
              <div className="btn-row">
                <button
                  className="btn primary small"
                  onClick={() => start(d.demo_id)}
                  disabled={starting !== null}
                  data-testid={`start-${d.demo_id}`}
                >
                  {starting === d.demo_id ? 'Starting…' : session?.demo_id === d.demo_id ? 'Restart' : 'Start'}
                </button>
              </div>
              {session?.demo_id === d.demo_id && (
                <>
                  <details>
                    <summary style={{ fontSize: 12, color: 'var(--muted)', cursor: 'pointer' }}>
                      Suggested steps ({d.walkthrough.length})
                    </summary>
                    <ol>{d.walkthrough.map((w, i) => <li key={i}>{w}</li>)}</ol>
                  </details>
                  <div className="provenance">
                    <strong>{d.source}</strong> · {d.scenario_id}
                    <div style={{ marginTop: 4 }}>{d.provenance}</div>
                  </div>
                </>
              )}
            </div>
          ))}
        </Panel>

      </div>

      <div>
        {!session ? (
          <Panel title="Workspace">
            <Empty title="Start a demonstration">
              Pick one on the left. Each opens a real simulator session — the
              records you see are only those your tool calls have returned.
            </Empty>
          </Panel>
        ) : (
          <>
            <Panel title="Assigned work">
              <div className="body">
                <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', marginBottom: 6 }}>
                  <Id value={session.scenario_id.slice(0, 34) + '…'} />
                  <span style={{ flex: 1 }} />
                  <span className="ts">signed in as {session.agent_user}</span>
                </div>
                <p style={{ margin: 0, whiteSpace: 'pre-wrap', fontSize: 13.5 }}>{session.brief}</p>
                {session.notices.length > 0 && (
                  <div style={{ marginTop: 10 }}>
                    {session.notices.map((n, i) => <Banner tone="warn" key={i}>{n}</Banner>)}
                  </div>
                )}
              </div>
            </Panel>

            <section className="panel" style={{ marginTop: 12 }}>
              <div className="tabs" role="tablist">
                {([
                  ['inbox', 'Inbox', o!.tickets.length],
                  ['billing', 'Customers & billing', o!.customers.length + o!.charges.length + o!.refunds.length],
                  ['approvals', 'Approvals', o!.approvals.length + o!.messages.length],
                  ['policy', 'Policy', o!.policy_rules.length],
                  ['history', 'Activity', session.history.length],
                ] as [Tab, string, number][]).map(([id, label, count]) => (
                  <button key={id} role="tab" aria-selected={tab === id}
                          onClick={() => setTab(id)} data-testid={`tab-${id}`}>
                    {label}<span className="count">{count}</span>
                  </button>
                ))}
              </div>
            </section>

            <div style={{ marginTop: 12 }}>
              {tab === 'inbox' && (
                <>
                  <Inbox session={session} selected={selectedTicket} onSelect={setSelected} />
                  {ticket && <TicketDetail ticket={ticket} />}
                </>
              )}
              {tab === 'billing' && <BillingRecords session={session} />}
              {tab === 'approvals' && <Approvals session={session} />}
              {tab === 'policy' && <PolicyPanel rules={o!.policy_rules} />}
              {tab === 'history' && <History session={session} />}
            </div>
          </>
        )}
      </div>

      <div className="rail-right">
        {session && (
          <ActionPanel
            session={session}
            busy={busy}
            error={error}
            onSubmit={submit}
            onClearError={() => setError(null)}
          />
        )}
      </div>
    </div>
  )
}

function History({ session }: { session: Session }) {
  return (
    <Panel title={`Activity log (${session.history.length})`}>
      {session.history.length === 0 ? (
        <Empty title="Nothing dispatched yet">
          Every action you submit is recorded here with the logical time at
          which it executed.
        </Empty>
      ) : (
        <table>
          <thead><tr><th>#</th><th>Time</th><th>Action</th><th>Outcome</th><th>Detail</th></tr></thead>
          <tbody>
            {session.history.map((h) => (
              <tr key={h.index}>
                <td className="num">{h.index}</td>
                <td><Tick value={h.logical_time} /></td>
                <td>
                  <div className="id">{h.kind}</div>
                  {Object.keys(h.arguments).length > 0 && (
                    <div style={{ fontSize: 11.5, color: 'var(--muted)' }}>
                      {Object.entries(h.arguments).filter(([, v]) => v !== null && v !== '')
                        .slice(0, 3).map(([k, v]) => `${k}=${String(v)}`).join(' · ')}
                    </div>
                  )}
                </td>
                <td><Pill value={h.outcome} /></td>
                <td style={{ fontSize: 12.5 }}>
                  {h.message || '—'}
                  {h.denied_interlock && (
                    <div className="ts" style={{ color: 'var(--bad)' }}>
                      blocked by backend: {h.denied_interlock}
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  )
}
