import { useEffect, useState } from 'react'
import { ApiError, reviewApi } from '../api'
import type { ReviewDetail, ReviewRow } from '../types'
import { Banner, Empty, Id, Panel, Pill, Spinner, Tick } from './ui'

/**
 * Why this page distinguishes 0 from 404.
 *
 * The bundle requests a *relative* `/review/*`, so the answer depends on which
 * server sent the page. Loaded from the reviewer with the process stopped, the
 * fetch fails outright (status 0). Loaded from the *operational* server, the
 * request succeeds and returns 404, because that server deliberately refuses
 * every reviewer route -- and it must keep refusing them. Those are different
 * problems with different fixes, and reporting a bare "not found" for the
 * second is what made the packaged reviewer look broken rather than misaddressed.
 */
function unavailableMessage(error: ApiError): string {
  if (error.status === 0) {
    return 'The reviewer API is not running on this origin. It is a separate '
      + 'process on purpose, so privileged data is not served during agent '
      + 'evaluation. Start it with `uv run cerl serve-review`.'
  }
  if (error.status === 404) {
    return 'This page was loaded from the operational server, which does not '
      + 'serve reviewer data and is not going to. Open the reviewer on its own '
      + 'port instead: `uv run cerl serve-review`, then '
      + 'http://127.0.0.1:8001/?review'
  }
  return error.message
}

/** Recorded-episode replay. Served by a *separate* process (`cerl serve-review`). */
export function Reviewer() {
  const [rows, setRows] = useState<ReviewRow[] | null>(null)
  const [detail, setDetail] = useState<ReviewDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    reviewApi.episodes()
      .then((r) => { setRows(r); setError(null) })
      .catch((e: ApiError) => setError(unavailableMessage(e)))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="body"><Spinner label="Loading recorded episodes" /></div>
  if (error) return <div className="body"><Banner tone="warn">{error}</Banner></div>

  return (
    <div className="layout" style={{ gridTemplateColumns: '420px minmax(0,1fr)' }}>
      <div>
        <Panel title={`Recorded episodes (${rows?.length ?? 0})`}>
          {!rows || rows.length === 0 ? (
            <Empty title="No recorded episodes found">
              Recorded runs live under <code>evidence/</code>.
            </Empty>
          ) : (
            <table>
              <thead><tr><th>Episode</th><th>Kind</th><th>Task</th><th>Safety</th></tr></thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.episode_id}
                      className={`selectable ${detail?.episode_id === r.episode_id ? 'is-selected' : ''}`}
                      onClick={() => reviewApi.detail(r.episode_id).then(setDetail)}>
                    <td>
                      <div className="id">{r.episode_id}</div>
                      <div style={{ fontSize: 11.5, color: 'var(--muted)' }}>{r.branch}</div>
                    </td>
                    <td><span className="pill neutral">{r.kind_label}</span></td>
                    <td className="num">{(r.task_completion * 100).toFixed(0)}%</td>
                    <td>
                      <Pill value={r.safe_task_completion ? 'safe' : 'unsafe'} />
                      {r.committed_violations > 0 && (
                        <div style={{ fontSize: 11, color: 'var(--bad)', marginTop: 2 }}>
                          {r.committed_violations} committed
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>
      </div>
      <div>
        {!detail ? (
          <Panel title="Episode detail">
            <Empty title="Select an episode">
              Choose a recorded episode to see its ordered actions, logical time,
              state hashes and verdict.
            </Empty>
          </Panel>
        ) : (
          <EpisodeDetail detail={detail} />
        )}
      </div>
    </div>
  )
}

function EpisodeDetail({ detail }: { detail: ReviewDetail }) {
  const partial = detail.kind_label.includes('interrupted')
  return (
    <>
      <Panel title={detail.episode_id}>
        <div className="body">
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
            <span className="pill neutral">{detail.kind_label}</span>
            {partial && <span className="pill warn">Partial observation — not a completed task</span>}
          </div>
          <div className="provenance">
            <strong>Provenance:</strong> {detail.provenance || '—'}<br />
            <strong>Termination:</strong> {detail.termination || '—'}
          </div>

          {/* Task and safety are separate results and are never merged. */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 12 }}>
            <div className="panel">
              <header><h2>Task completion</h2></header>
              <div className="body detail">
                <dl>
                  <dt>Completion</dt><dd>{(detail.task.task_completion * 100).toFixed(0)}%</dd>
                  <dt>Correct final state</dt><dd><Pill value={detail.task.correct_final_state ? 'yes' : 'no'} /></dd>
                  <dt>Decision correct</dt><dd><Pill value={detail.task.decision_correct ? 'yes' : 'no'} /></dd>
                  <dt>Required</dt><dd>{detail.required_decision}</dd>
                  <dt>Declared</dt><dd>{detail.declared_outcome ?? '— none —'}</dd>
                </dl>
              </div>
            </div>
            <div className="panel">
              <header><h2>Safety</h2></header>
              <div className="body detail">
                <dl>
                  <dt>Safe task completion</dt>
                  <dd><Pill value={detail.safety.safe_task_completion ? 'safe' : 'unsafe'} /></dd>
                  <dt>Failure class</dt><dd>{detail.safety.failure_class}</dd>
                  <dt>Committed</dt>
                  <dd>{detail.safety.committed_violations.length === 0 ? '0' :
                    detail.safety.committed_violations.map((v, i) =>
                      <span className="pill bad" key={i} style={{ marginRight: 4 }}>{v.cost_class}</span>)}</dd>
                  <dt>Attempted (blocked)</dt>
                  <dd>{detail.safety.attempted_violations.length === 0 ? '0' :
                    detail.safety.attempted_violations.map((v, i) =>
                      <span className="pill warn" key={i} style={{ marginRight: 4 }}>{v.cost_class}</span>)}</dd>
                  <dt>Side effects</dt><dd>{detail.safety.prohibited_side_effects.length}</dd>
                </dl>
              </div>
            </div>
          </div>
        </div>
      </Panel>

      <Panel title={`Action timeline (${detail.timeline.length})`}>
        <table>
          <thead>
            <tr><th>#</th><th>Time</th><th>Origin</th><th>Action</th><th>Outcome</th>
              <th>Violations</th><th>State hash after</th></tr>
          </thead>
          <tbody>
            {detail.timeline.map((s) => (
              <tr key={s.index}>
                <td className="num">{s.index}</td>
                <td><Tick value={s.logical_time} /></td>
                <td>
                  <span className={`pill ${s.origin === 'responder' ? 'warn' : 'neutral'}`}>
                    {s.origin}
                  </span>
                  {s.responder_rule && <div className="ts">{s.responder_rule}</div>}
                </td>
                <td>
                  <div className="id">{s.kind}</div>
                  {Object.keys(s.arguments).length > 0 && (
                    <div style={{ fontSize: 11.5, color: 'var(--muted)' }}>
                      {Object.entries(s.arguments).slice(0, 3)
                        .map(([k, v]) => `${k}=${String(v)}`).join(' · ')}
                    </div>
                  )}
                </td>
                <td>
                  <Pill value={s.outcome} />
                  {s.denied_interlock && <div className="ts">{s.denied_interlock}</div>}
                </td>
                <td>
                  {s.committed_classes.map((c) => (
                    <span className="pill bad" key={c} style={{ marginRight: 3 }}>{c}</span>
                  ))}
                  {s.attempted_classes.map((c) => (
                    <span className="pill warn" key={c} style={{ marginRight: 3 }}>{c} (blocked)</span>
                  ))}
                  {s.committed_classes.length + s.attempted_classes.length === 0 && '—'}
                </td>
                <td><Id value={s.state_hash_after?.slice(0, 12)} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </>
  )
}
