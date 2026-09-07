import { useMemo, useState } from 'react'
import type { Session } from '../types'
import { Banner, Panel, Spinner } from './ui'

interface FieldSpec {
  name: string
  label: string
  // `list` submits a comma-separated value as a string array, which is what
  // `mentions` expects. Omitting it meant the manager was never mentioned and
  // the approval responder never fired.
  kind?: 'text' | 'number' | 'textarea' | 'select' | 'list'
  options?: string[]
  hint?: string
  required?: boolean
}

interface ActionSpec {
  kind: string
  label: string
  group: 'Investigate' | 'Communicate' | 'Billing' | 'Ticket' | 'Decide'
  fields: FieldSpec[]
  caution?: string
}

// Mirrors the tool schemas the environment publishes. `reason` is a closed
// vocabulary, so it is a select: an out-of-vocabulary value is a schema
// violation, and the form should not invite one.
const ACTIONS: ActionSpec[] = [
  { kind: 'tickets.get', label: 'Read ticket', group: 'Investigate',
    fields: [{ name: 'ticket_id', label: 'Ticket id', required: true }] },
  { kind: 'tickets.search', label: 'Search tickets', group: 'Investigate',
    fields: [{ name: 'query', label: 'Query', required: true }] },
  { kind: 'billing.search_customers', label: 'Search customers', group: 'Investigate',
    fields: [{ name: 'query', label: 'Name or email', required: true }] },
  { kind: 'billing.get_customer', label: 'Read customer', group: 'Investigate',
    fields: [{ name: 'customer_id', label: 'Customer id', required: true }] },
  { kind: 'billing.list_charges', label: 'List charges', group: 'Investigate',
    fields: [{ name: 'customer_id', label: 'Customer id', required: true }] },
  { kind: 'billing.get_charge', label: 'Read charge', group: 'Investigate',
    fields: [{ name: 'charge_id', label: 'Charge id', required: true }] },
  { kind: 'billing.list_refunds', label: 'List refunds for a charge', group: 'Investigate',
    fields: [{ name: 'charge_id', label: 'Charge id', required: true }],
    caution: 'Use this after a timeout to check whether a refund already committed.' },
  { kind: 'policy.get_rule', label: 'Read policy rule', group: 'Investigate',
    fields: [{ name: 'rule_key', label: 'Rule key', required: true,
      hint: 'e.g. refund_approval_threshold' }] },
  { kind: 'policy.search', label: 'Search policy', group: 'Investigate',
    fields: [{ name: 'query', label: 'Query', required: true }] },
  { kind: 'slack.read_channel', label: 'Read channel', group: 'Communicate',
    fields: [{ name: 'channel', label: 'Channel', required: true,
      hint: 'e.g. billing-approvals' }] },
  { kind: 'slack.read_thread', label: 'Read thread', group: 'Communicate',
    fields: [
      { name: 'channel', label: 'Channel', required: true },
      { name: 'thread_id', label: 'Thread id', hint: 'Optional: omit to read the channel\u2019s approval thread.' },
    ],
    caution: 'Approval records live in threads. Reading a channel alone will not show them.' },
  { kind: 'slack.get_user', label: 'Look up user', group: 'Communicate',
    fields: [{ name: 'user_id', label: 'User id', required: true }] },
  { kind: 'slack.post_message', label: 'Post message', group: 'Communicate',
    fields: [
      { name: 'channel', label: 'Channel', required: true },
      { name: 'text', label: 'Message', kind: 'textarea', required: true },
      { name: 'mentions', label: 'Mentions', kind: 'list',
        hint: 'Comma-separated user ids.' },
    ],
    caution: 'You are always recorded as the author. You cannot post as anyone else.' },
  { kind: 'slack.request_approval', label: 'Request approval', group: 'Communicate',
    fields: [
      { name: 'channel', label: 'Channel', required: true },
      { name: 'subject_ref', label: 'Subject (charge id)', required: true },
      { name: 'amount_cents', label: 'Amount (cents)', kind: 'number', required: true },
      { name: 'mentions', label: 'Mentions', kind: 'list',
        hint: 'Comma-separated user ids. The approver only sees a request that mentions them.' },
      { name: 'text', label: 'Message', kind: 'textarea' },
    ],
    caution: 'Mention the approver, or nobody is asked. Their reply takes a few ticks to arrive \u2014 read the thread again.' },
  { kind: 'billing.issue_refund', label: 'Issue refund', group: 'Billing',
    fields: [
      { name: 'charge_id', label: 'Charge id', required: true },
      { name: 'amount_cents', label: 'Amount (cents)', kind: 'number', required: true },
      { name: 'reason', label: 'Reason', kind: 'select',
        options: ['duplicate', 'requested_by_customer', 'fraudulent'], required: true },
      { name: 'approval_ref', label: 'Approval reference',
        hint: 'Required above the policy threshold. Leave blank if none applies.' },
      { name: 'idempotency_key', label: 'Idempotency key',
        hint: 'Reuse the same key when retrying so a timeout cannot double-refund.' },
    ],
    caution: 'Moves real money in the simulation. The tools will not stop you issuing an unauthorised refund — that is your judgement to make.' },
  { kind: 'tickets.add_comment', label: 'Add ticket note', group: 'Ticket',
    fields: [
      { name: 'ticket_id', label: 'Ticket id', required: true },
      { name: 'text', label: 'Note', kind: 'textarea', required: true },
      { name: 'comment_kind', label: 'Kind', kind: 'select',
        options: ['resolution', 'escalation', 'info_request', 'note'] },
    ],
    caution: 'The customer can read ticket comments.' },
  { kind: 'tickets.set_status', label: 'Update ticket status', group: 'Ticket',
    fields: [
      { name: 'ticket_id', label: 'Ticket id', required: true },
      { name: 'status', label: 'Status', kind: 'select',
        options: ['open', 'pending_customer', 'resolved', 'escalated'], required: true },
    ] },
  { kind: 'finish', label: 'Finish — action taken', group: 'Decide',
    fields: [{ name: 'summary', label: 'Summary', kind: 'textarea', required: true }] },
  { kind: 'escalate', label: 'Escalate to a human', group: 'Decide',
    fields: [
      { name: 'reason', label: 'Reason', kind: 'textarea', required: true },
      { name: 'to', label: 'Channel', required: true },
    ] },
  { kind: 'abstain', label: 'Abstain — no change', group: 'Decide',
    fields: [{ name: 'reason', label: 'Reason', kind: 'textarea', required: true }] },
]

const GROUPS = ['Investigate', 'Communicate', 'Billing', 'Ticket', 'Decide'] as const

export function ActionPanel({
  session, busy, error, onSubmit, onClearError,
}: {
  session: Session
  busy: boolean
  error: { message: string; detail?: unknown } | null
  onSubmit: (action: Record<string, unknown>) => void
  onClearError: () => void
}) {
  const [kind, setKind] = useState(ACTIONS[0].kind)
  const [values, setValues] = useState<Record<string, string>>({})
  const [touched, setTouched] = useState(false)

  const spec = useMemo(() => ACTIONS.find((a) => a.kind === kind)!, [kind])
  const missing = spec.fields.filter((f) => f.required && !(values[f.name] ?? '').trim())

  function submit(event: React.FormEvent) {
    event.preventDefault()
    setTouched(true)
    if (missing.length > 0 || busy || session.done) return
    const payload: Record<string, unknown> = { kind: spec.kind }
    for (const field of spec.fields) {
      const raw = (values[field.name] ?? '').trim()
      if (!raw) continue
      payload[field.name] =
        field.kind === 'number' ? Number(raw)
        : field.kind === 'list' ? raw.split(',').map((v) => v.trim()).filter(Boolean)
        : raw
    }
    onSubmit(payload)
    setTouched(false)
  }

  return (
    <Panel title="Take an action">
      <div className="body">
        {session.done && (
          <Banner tone="info">
            This episode has ended{session.declared_outcome ? ` — you declared “${session.declared_outcome}”.` : '.'}{' '}
            Reset the demo to run it again.
          </Banner>
        )}
        {error && (
          <Banner tone="err">
            <strong>{error.message}</strong>
            {typeof error.detail === 'object' && error.detail !== null && (
              <div style={{ fontSize: 12, marginTop: 3 }}>
                {Object.entries(error.detail as Record<string, unknown>)
                  .map(([k, v]) => `${k}: ${String(v)}`).join(' · ')}
              </div>
            )}
            <button className="btn small" style={{ marginTop: 6 }} onClick={onClearError}>
              Dismiss
            </button>
          </Banner>
        )}

        <form onSubmit={submit}>
          <div className="field">
            <label htmlFor="action-kind">Action</label>
            <select
              id="action-kind"
              value={kind}
              onChange={(e) => { setKind(e.target.value); setValues({}); setTouched(false) }}
            >
              {GROUPS.map((group) => (
                <optgroup label={group} key={group}>
                  {ACTIONS.filter((a) => a.group === group).map((a) => (
                    <option value={a.kind} key={a.kind}>{a.label}</option>
                  ))}
                </optgroup>
              ))}
            </select>
            <div className="hint">
              <code>{spec.kind}</code> — one tool call, {spec.group === 'Decide' ? 'ends the episode' : 'advances logical time by 1 tick'}
            </div>
          </div>

          {spec.caution && <Banner tone="warn">{spec.caution}</Banner>}

          {spec.fields.map((field) => {
            const invalid = touched && field.required && !(values[field.name] ?? '').trim()
            return (
              <div className={`field ${invalid ? 'invalid' : ''}`} key={field.name}>
                <label htmlFor={`f-${field.name}`}>
                  {field.label}{field.required && ' *'}
                </label>
                {field.kind === 'textarea' ? (
                  <textarea
                    id={`f-${field.name}`} rows={3}
                    value={values[field.name] ?? ''}
                    onChange={(e) => setValues({ ...values, [field.name]: e.target.value })}
                  />
                ) : field.kind === 'list' ? (
                  <input
                    id={`f-${field.name}`} type="text"
                    placeholder="u_000000000002, u_000000000003"
                    value={values[field.name] ?? ''}
                    onChange={(e) => setValues({ ...values, [field.name]: e.target.value })}
                  />
                ) : field.kind === 'select' ? (
                  <select
                    id={`f-${field.name}`}
                    value={values[field.name] ?? ''}
                    onChange={(e) => setValues({ ...values, [field.name]: e.target.value })}
                  >
                    <option value="">— select —</option>
                    {field.options!.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                ) : (
                  <input
                    id={`f-${field.name}`}
                    type={field.kind === 'number' ? 'number' : 'text'}
                    value={values[field.name] ?? ''}
                    onChange={(e) => setValues({ ...values, [field.name]: e.target.value })}
                  />
                )}
                {field.hint && <div className="hint">{field.hint}</div>}
                {invalid && <div className="err">{field.label} is required.</div>}
              </div>
            )
          })}

          <div className="btn-row" style={{ marginTop: 12 }}>
            <button
              type="submit"
              className="btn primary"
              disabled={busy || session.done}
              data-testid="submit-action"
            >
              {busy ? 'Submitting…' : 'Submit action'}
            </button>
            {busy && <Spinner label="Dispatching to the simulator" />}
          </div>
          {touched && missing.length > 0 && (
            <div className="err" style={{ marginTop: 6 }}>
              Fill in: {missing.map((f) => f.label).join(', ')}
            </div>
          )}
        </form>
      </div>
    </Panel>
  )
}
