import React, { useState } from 'react'
import { Badge, Button, Card } from './Primitives'

const fmt = (v) => {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 3 })
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  if (typeof v === 'object') return JSON.stringify(v).slice(0, 60)
  return String(v)
}

const TABLE_KEYS = ['rows', 'series', 'examples', 'anomalies', 'top_correlates', 'columns', 'issues',
  'models_compared', 'top_features', 'clusters']

/** Pull displayable tables out of a tool output (generic, so new tools need no UI work). */
function tablesFrom(output) {
  if (!output || typeof output !== 'object') return []
  const tables = []
  TABLE_KEYS.forEach((k) => {
    const v = output[k]
    if (Array.isArray(v) && v.length && typeof v[0] === 'object') tables.push({ title: k.replace(/_/g, ' '), rows: v })
  })
  ;(output.breakdowns || []).forEach((b) => {
    if (b.segments?.length) tables.push({ title: `change by ${b.column}`, rows: b.segments })
  })
  return tables
}

function MiniTable({ title, rows }) {
  const cols = Object.keys(rows[0]).filter((c) => typeof rows[0][c] !== 'object' || rows[0][c] === null).slice(0, 8)
  const shown = rows.slice(0, 12)
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--ink-faint)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 4 }}>{title}</div>
      <div style={{ overflowX: 'auto', border: '1px solid var(--hairline)', borderRadius: 8 }}>
        <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 11.5 }}>
          <thead>
            <tr style={{ background: 'var(--paper)' }}>
              {cols.map((c) => <th key={c} style={{ textAlign: 'left', padding: '5px 9px', fontWeight: 600, color: 'var(--ink-faint)', whiteSpace: 'nowrap' }}>{c.replace(/_/g, ' ')}</th>)}
            </tr>
          </thead>
          <tbody>
            {shown.map((r, i) => (
              <tr key={i} style={{ borderTop: '1px solid var(--hairline)' }}>
                {cols.map((c) => <td key={c} style={{ padding: '5px 9px', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap', color: 'var(--ink)' }}>{fmt(r[c])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > shown.length && <div style={{ fontSize: 11, color: 'var(--ink-faint)', marginTop: 3 }}>+ {rows.length - shown.length} more rows</div>}
    </div>
  )
}

function EvidenceItem({ ev, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  const tables = tablesFrom(ev.output)
  return (
    <div style={{ border: '1px solid var(--hairline)', borderRadius: 10, background: 'var(--paper-raised)', overflow: 'hidden' }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{ width: '100%', display: 'flex', alignItems: 'center', gap: 10, padding: '9px 12px', background: 'transparent', border: 'none', cursor: 'pointer', textAlign: 'left' }}
      >
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, fontWeight: 700, color: 'var(--accent-ink)', background: 'var(--accent-soft)', borderRadius: 5, padding: '2px 6px' }}>#{ev.id}</span>
        <span style={{ flex: 1, fontSize: 12.5, color: 'var(--ink)', fontWeight: 500 }}>{ev.label}</span>
        {ev.error && <Badge tone="alert">error</Badge>}
        <span style={{ color: 'var(--ink-faint)', fontSize: 11 }}>{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div style={{ padding: '2px 12px 12px', borderTop: '1px solid var(--hairline)' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-soft)', marginTop: 8 }}>
            {ev.tool}({Object.entries(ev.input || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ')})
          </div>
          {ev.error && <div style={{ marginTop: 8, fontSize: 12, color: 'var(--alert)' }}>{ev.output?.error}</div>}
          {tables.map((t) => <MiniTable key={t.title} {...t} />)}
          <details style={{ marginTop: 10 }}>
            <summary style={{ fontSize: 11, color: 'var(--ink-faint)', cursor: 'pointer' }}>Raw result</summary>
            <pre style={{ margin: '6px 0 0', padding: 10, background: 'var(--paper)', borderRadius: 8, fontSize: 10.5, maxHeight: 220, overflow: 'auto', fontFamily: 'var(--font-mono)' }}>
              {JSON.stringify(ev.output, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  )
}

/** Collapsible list of the calculations behind an answer. Pass `ids` to show only cited items. */
export function EvidenceList({ evidence, ids, title = 'How this was calculated' }) {
  const [open, setOpen] = useState(false)
  const items = (evidence || []).filter((e) => !ids || ids.includes(e.id))
  if (!items.length) return null
  return (
    <div style={{ marginTop: 10 }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{ background: 'none', border: 'none', padding: 0, cursor: 'pointer', fontSize: 12, fontWeight: 600, color: 'var(--accent-ink)', fontFamily: 'var(--font-body)' }}
      >
        {open ? '▾' : '▸'} {title} ({items.length} {items.length === 1 ? 'calculation' : 'calculations'})
      </button>
      {open && <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>{items.map((e) => <EvidenceItem key={e.id} ev={e} defaultOpen={items.length === 1} />)}</div>}
    </div>
  )
}

/** "7 of 8 figures traced to calculations" -- an honest trust signal, not a guarantee. */
export function GroundingBadge({ grounding }) {
  if (!grounding || !grounding.checked) return null
  const { checked, traced, untraced = [] } = grounding
  const all = traced === checked
  const tone = all ? 'positive' : traced / checked >= 0.7 ? 'amber' : 'alert'
  const tip = all
    ? 'Every figure in this answer matches a calculation.'
    : `Not found in any calculation (may be derived by the model — verify): ${untraced.join(', ')}`
  return <span title={tip}><Badge tone={tone}>{traced}/{checked} figures traced</Badge></span>
}

const CONF_TONE = { high: 'positive', medium: 'amber', low: 'alert' }
export const ConfidenceBadge = ({ level }) => <Badge tone={CONF_TONE[level] || 'default'}>{level} confidence</Badge>

export function LlmSetupCard({ status }) {
  return (
    <Card style={{ padding: '24px 26px', maxWidth: 620 }}>
      <div style={{ fontFamily: 'var(--font-display)', fontSize: 20, fontWeight: 600, marginBottom: 8 }}>Connect the AI analyst</div>
      <p style={{ margin: '0 0 14px', fontSize: 14, color: 'var(--ink-soft)', lineHeight: 1.6 }}>
        Everything else works without it. To enable chat, briefings and root-cause analysis, give the backend a Gemini API key (free tier available):
      </p>
      <pre style={{ margin: 0, padding: '12px 14px', background: 'var(--ink)', color: '#E8E8EC', borderRadius: 10, fontSize: 12.5, fontFamily: 'var(--font-mono)', overflowX: 'auto' }}>
{`# .env  (copy from .env.example)
GEMINI_API_KEY=your_gemini_api_key

docker compose up --build`}
      </pre>
      <p style={{ margin: '14px 0 0', fontSize: 12.5, color: 'var(--ink-faint)', lineHeight: 1.55 }}>
        The model only chooses which calculations to run — every number comes from deterministic pandas / scikit-learn code
        and is shown under “How this was calculated”.{status?.model ? ` Model: ${status.model}.` : ''}
      </p>
    </Card>
  )
}

/** Shown when the server protects its AI key with APP_ACCESS_CODE. */
export function AccessCodeCard({ onSubmit, denied }) {
  const [value, setValue] = useState('')
  const submit = () => { if (value.trim()) onSubmit(value.trim()) }
  return (
    <Card style={{ padding: '24px 26px', maxWidth: 480 }}>
      <div style={{ fontFamily: 'var(--font-display)', fontSize: 20, fontWeight: 600, marginBottom: 8 }}>Access code required</div>
      <p style={{ margin: '0 0 14px', fontSize: 14, color: 'var(--ink-soft)', lineHeight: 1.6 }}>
        The AI analyst on this deployment is protected so its API key can’t be misused. Enter the access code you were given.
        Everything else works without it.
      </p>
      {denied && (
        <div style={{ padding: '9px 12px', background: 'var(--alert-soft)', color: 'var(--alert)', borderRadius: 8, fontSize: 13, marginBottom: 12 }}>
          That code wasn’t accepted. Check it and try again.
        </div>
      )}
      <div style={{ display: 'flex', gap: 10 }}>
        <input
          type="password" value={value} autoFocus placeholder="Access code" aria-label="Access code"
          onChange={(e) => setValue(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') submit() }}
          style={{ flex: 1, padding: '10px 12px', borderRadius: 10, border: '1px solid var(--hairline-strong)', fontFamily: 'var(--font-mono)', fontSize: 13.5, background: 'var(--paper)', color: 'var(--ink)' }}
        />
        <Button variant="accent" onClick={submit} disabled={!value.trim()}>Unlock</Button>
      </div>
    </Card>
  )
}
