import { useState } from 'react'
import type { Charge, Customer, Message, PolicyRule, Refund, Session, Ticket } from '../types'
import { Empty, Id, Money, Panel, Pill, Tick } from './ui'

/** Ticket inbox with search. Empty until a read tool has actually returned. */
export function Inbox({
  session, selected, onSelect,
}: { session: Session; selected: string | null; onSelect: (id: string) => void }) {
  const [query, setQuery] = useState('')
  const tickets = session.observed.tickets.filter((t) => {
    if (!query.trim()) return true
    const hay = `${t.id} ${t.subject ?? ''} ${t.status ?? ''} ${t.customer_id ?? ''}`
    return hay.toLowerCase().includes(query.toLowerCase())
  })

  return (
    <Panel
      title={`Ticket inbox (${session.observed.tickets.length})`}
      actions={
        <input
          aria-label="Filter tickets"
          placeholder="Filter by id, subject, status…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ width: 220, padding: '4px 8px', border: '1px solid var(--line)', borderRadius: 4 }}
        />
      }
    >
      {session.observed.tickets.length === 0 ? (
        <Empty title="No tickets retrieved yet">
          The inbox shows only records you have actually fetched. Use{' '}
          <strong>Read ticket</strong> or <strong>Search tickets</strong> in the
          action panel — each one is a real tool call and advances logical time.
        </Empty>
      ) : tickets.length === 0 ? (
        <Empty title="No tickets match that filter">Clear the filter to see all {session.observed.tickets.length} retrieved tickets.</Empty>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Ticket</th><th>Subject</th><th>Customer</th><th>Status</th><th>Comments</th>
            </tr>
          </thead>
          <tbody>
            {tickets.map((t: Ticket) => (
              <tr
                key={t.id}
                className={`selectable ${selected === t.id ? 'is-selected' : ''}`}
                onClick={() => onSelect(t.id)}
              >
                <td><Id value={t.id} /></td>
                <td>{t.subject ?? '—'}</td>
                <td><Id value={t.customer_id} /></td>
                <td><Pill value={t.status} /></td>
                <td className="num">{(t.comments ?? []).length}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  )
}

export function TicketDetail({ ticket }: { ticket: Ticket | undefined }) {
  if (!ticket) return null
  return (
    <Panel title={`Ticket ${ticket.id}`}>
      <div className="body detail">
        <dl>
          <dt>Subject</dt><dd>{ticket.subject ?? '—'}</dd>
          <dt>Status</dt><dd><Pill value={ticket.status} /></dd>
          <dt>Customer</dt><dd><Id value={ticket.customer_id} /></dd>
        </dl>
        {ticket.body && (
          <p style={{ marginTop: 10, whiteSpace: 'pre-wrap', fontSize: 13 }}>{ticket.body}</p>
        )}
        {(ticket.comments ?? []).length > 0 && (
          <>
            <h3 style={{ fontSize: 12, textTransform: 'uppercase', color: 'var(--muted)', marginTop: 14 }}>
              Comments
            </h3>
            <div className="thread">
              {(ticket.comments ?? []).map((c, i) => (
                <div className="msg" key={i}>
                  <header>
                    <span className="who">{c.author}</span>
                    {c.kind && <Pill value={c.kind} />}
                  </header>
                  <div className="text">{c.text}</div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </Panel>
  )
}

export function BillingRecords({ session }: { session: Session }) {
  const { customers, charges, refunds } = session.observed
  const nothing = customers.length + charges.length + refunds.length === 0
  if (nothing) {
    return (
      <Panel title="Customer & billing records">
        <Empty title="No billing records retrieved">
          Look up the customer and list their charges from the action panel.
          Billing data appears here only once a tool call has returned it.
        </Empty>
      </Panel>
    )
  }
  return (
    <>
      {customers.length > 0 && (
        <Panel title={`Customers (${customers.length})`}>
          <table>
            <thead><tr><th>Customer</th><th>Name</th><th>Email</th><th>External ref</th><th>Status</th></tr></thead>
            <tbody>
              {customers.map((c: Customer) => (
                <tr key={c.id}>
                  <td><Id value={c.id} /></td>
                  <td>{c.display_name ?? '—'}</td>
                  <td style={{ fontSize: 12.5 }}>{c.email ?? '—'}</td>
                  <td className="ts">{c.external_ref ?? '—'}</td>
                  <td><Pill value={c.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}
      {charges.length > 0 && (
        <Panel title={`Charges (${charges.length})`}>
          <table>
            <thead>
              <tr><th>Charge</th><th>Customer</th><th>Invoice</th><th className="num">Amount</th>
                <th className="num">Refunded</th><th>Created</th><th>Status</th></tr>
            </thead>
            <tbody>
              {charges.map((c: Charge) => (
                <tr key={c.id}>
                  <td><Id value={c.id} /></td>
                  <td><Id value={c.customer_id} /></td>
                  <td className="ts">{(c.invoice_id as string) ?? '—'}</td>
                  <td className="num"><Money cents={c.amount?.cents} /></td>
                  <td className="num"><Money cents={c.refunded_total?.cents} /></td>
                  <td><Tick value={c.created_at} /></td>
                  <td><Pill value={c.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}
      {refunds.length > 0 && (
        <Panel title={`Refunds (${refunds.length})`}>
          <table>
            <thead>
              <tr><th>Refund</th><th>Charge</th><th className="num">Amount</th>
                <th>Reason</th><th>Approval ref</th><th>Issued</th></tr>
            </thead>
            <tbody>
              {refunds.map((r: Refund) => (
                <tr key={r.id}>
                  <td><Id value={r.id} /></td>
                  <td><Id value={r.charge_id} /></td>
                  <td className="num"><Money cents={r.amount?.cents} /></td>
                  <td><Pill value={r.reason} /></td>
                  <td><Id value={r.approval_ref ?? undefined} /></td>
                  <td><Tick value={r.created_at} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}
    </>
  )
}

export function Approvals({ session }: { session: Session }) {
  const { messages, approvals } = session.observed
  return (
    <>
      <Panel title={`Approval records (${approvals.length})`}>
        {approvals.length === 0 ? (
          <Empty title="No approval records retrieved">
            Read the approvals thread to see whether an approval exists, who
            granted it, and when it expires.
          </Empty>
        ) : (
          <table>
            <thead>
              <tr><th>Approval</th><th>Approver</th><th>Subject</th><th>State</th>
                <th>Granted</th><th>Expires</th><th className="num">Scope max</th></tr>
            </thead>
            <tbody>
              {approvals.map((a) => (
                <tr key={a.id}>
                  <td><Id value={a.id} /></td>
                  <td><Id value={a.approver} /></td>
                  <td><Id value={a.subject_ref} /></td>
                  <td><Pill value={a.state} /></td>
                  <td><Tick value={a.granted_at} /></td>
                  <td><Tick value={a.expires_at ?? undefined} /></td>
                  <td className="num"><Money cents={a.scope_amount_max?.cents} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
      <Panel title={`Internal conversation (${messages.length})`}>
        <div className="body">
          {messages.length === 0 ? (
            <Empty title="No messages retrieved">
              Read a channel or a thread to see the conversation.
            </Empty>
          ) : (
            <div className="thread">
              {messages.map((m: Message) => (
                <div className={`msg ${m.author === session.agent_user ? 'mine' : ''}`} key={m.id}>
                  <header>
                    <span className="who">{m.author ?? 'unknown'}</span>
                    <span className="ts">#{m.channel ?? '—'}</span>
                    <span style={{ flex: 1 }} />
                    <Tick value={m.posted_at} />
                  </header>
                  <div className="text">{m.text}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </Panel>
    </>
  )
}

export function PolicyPanel({ rules }: { rules: PolicyRule[] }) {
  return (
    <Panel title={`Policy (${rules.length})`}>
      <div className="body">
        {rules.length === 0 ? (
          <Empty title="No policy retrieved">
            Company policy is authoritative and must be read, not assumed. Use{' '}
            <strong>Read policy rule</strong> or <strong>Search policy</strong>.
          </Empty>
        ) : (
          <dl style={{ margin: 0 }}>
            {rules.map((r) => (
              <div key={r.key} style={{ marginBottom: 10 }}>
                <dt className="id" style={{ marginBottom: 2 }}>{r.key}</dt>
                <dd style={{ margin: 0, fontSize: 13 }}>{r.text ?? '—'}</dd>
              </div>
            ))}
          </dl>
        )}
      </div>
    </Panel>
  )
}
