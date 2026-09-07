// Shapes the operational API actually returns. Nothing privileged appears here
// because nothing privileged is served: no branch, no rubric, no verdict.

export interface Demo {
  demo_id: string
  title: string
  summary: string
  walkthrough: string[]
  scenario_id: string
  source: string
  provenance: string
}

export interface Ticket {
  id: string
  subject?: string
  body?: string
  status?: string
  customer_id?: string
  comments?: { author: string; text: string; kind?: string }[]
  [k: string]: unknown
}

export interface Customer {
  id: string
  display_name?: string
  email?: string
  external_ref?: string
  status?: string
  created_at?: number
  [k: string]: unknown
}

export interface Charge {
  id: string
  customer_id?: string
  amount?: { cents: number; currency: string }
  status?: string
  created_at?: number
  description?: string
  invoice_id?: string | null
  refunded_total?: { cents: number; currency: string }
  [k: string]: unknown
}

export interface Refund {
  id: string
  charge_id?: string
  amount?: { cents: number; currency: string }
  reason?: string
  approval_ref?: string | null
  created_at?: number
  issued_by?: string
  [k: string]: unknown
}

export interface Approval {
  id: string
  approver?: string
  subject_ref?: string
  state?: string
  granted_at?: number
  expires_at?: number | null
  scope_amount_max?: { cents: number } | null
  [k: string]: unknown
}

export interface Message {
  id: string
  channel?: string
  author?: string
  text?: string
  posted_at?: number
  thread_id?: string | null
  [k: string]: unknown
}

export interface PolicyRule {
  key: string
  text?: string
  [k: string]: unknown
}

export interface HistoryRecord {
  index: number
  kind: string
  arguments: Record<string, unknown>
  outcome: string
  message: string
  logical_time: number
  denied_interlock: string | null
}

export interface Session {
  session_id: string
  demo_id: string
  scenario_id: string
  brief: string
  agent_user: string
  logical_time: number
  step_index: number
  steps_remaining: number
  budget_steps: number
  done: boolean
  declared_outcome: string | null
  notices: string[]
  observed: {
    tickets: Ticket[]
    customers: Customer[]
    charges: Charge[]
    refunds: Refund[]
    approvals: Approval[]
    messages: Message[]
    policy_rules: PolicyRule[]
    disputes: unknown[]
    users: { id: string; handle?: string; display_name?: string; roles?: string[] }[]
  }
  history: HistoryRecord[]
}

export interface ApiFailure {
  error: string
  detail?: unknown
}

// ---- reviewer (separate server) ----

export interface ReviewRow {
  episode_id: string
  run_id: string
  kind: string
  kind_label: string
  scenario_id: string
  branch: string
  failure_class: string
  task_completion: number
  safe_task_completion: boolean
  committed_violations: number
  attempted_violations: number
  steps: number
}

export interface ReviewDetail {
  episode_id: string
  kind_label: string
  provenance: string
  termination: string
  scenario_id: string
  branch: string
  required_decision: string
  declared_outcome: string | null
  task: {
    task_completion: number
    correct_final_state: boolean
    decision_correct: boolean
    rubric: Record<string, boolean>
  }
  safety: {
    safe_task_completion: boolean
    failure_class: string
    committed_violations: { cost_class?: string; detail?: string }[]
    attempted_violations: { cost_class?: string; detail?: string }[]
    prohibited_side_effects: string[]
  }
  hashes: Record<string, string>
  timeline: {
    index: number
    origin: string
    actor: string
    logical_time: number
    kind: string
    arguments: Record<string, unknown>
    outcome: string
    denied_interlock: string | null
    committed_classes: string[]
    attempted_classes: string[]
    state_hash_before: string
    state_hash_after: string
    responder_rule: string | null
  }[]
}
