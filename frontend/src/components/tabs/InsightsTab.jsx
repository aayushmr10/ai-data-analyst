import React, { useState } from 'react'
import api, { errorMessage } from '../../api/client'
import { Badge, Button, Card, SectionLabel, Spinner } from '../Primitives'
import RichText from '../RichText'
import { AccessCodeCard, ConfidenceBadge, EvidenceList, GroundingBadge, LlmSetupCard } from '../AiParts'

const TYPE_LABEL = { trend: 'Trend', segment: 'Segment', relationship: 'Relationship', anomaly: 'Anomaly', data_quality: 'Data quality', driver: 'Driver' }
const VERDICT = {
  explained: { tone: 'positive', label: 'Explained' },
  partially_explained: { tone: 'amber', label: 'Partially explained' },
  inconclusive: { tone: 'default', label: 'Inconclusive' },
}

const Bullets = ({ title, items, tone }) => !items?.length ? null : (
  <div style={{ marginTop: 14 }}>
    <div style={{ fontSize: 11, fontWeight: 700, fontFamily: 'var(--font-mono)', letterSpacing: '0.06em', textTransform: 'uppercase', color: tone || 'var(--ink-faint)', marginBottom: 6 }}>{title}</div>
    <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13.5, color: 'var(--ink-soft)', lineHeight: 1.6 }}>
      {items.map((t, i) => <li key={i} style={{ marginBottom: 3 }}>{t}</li>)}
    </ul>
  </div>
)

const ErrorNote = ({ children }) => (
  <div style={{ padding: '10px 14px', background: 'var(--alert-soft)', color: 'var(--alert)', borderRadius: 10, fontSize: 13.5, marginTop: 12 }}>{children}</div>
)

const Working = ({ children }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '28px 4px', color: 'var(--ink-faint)', fontSize: 13 }}>
    <Spinner size={16} /> {children}
  </div>
)

// ---------------------------------------------------------------- briefing
function Briefing({ datasetId, briefing, setBriefing, onAsk, onAccessDenied }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const run = async (force) => {
    setLoading(true); setError(null)
    try { setBriefing(await api.briefing(datasetId, force)) }
    catch (e) { if (e?.response?.status === 403) onAccessDenied(); setError(errorMessage(e)) }
    finally { setLoading(false) }
  }

  const b = briefing?.structured
  return (
    <section style={{ marginBottom: 44 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <SectionLabel tone="accent">Analyst briefing</SectionLabel>
        {b && <Button variant="subtle" size="sm" onClick={() => run(true)} disabled={loading}>Regenerate</Button>}
      </div>

      {!b && !loading && (
        <Card style={{ padding: '22px 24px', maxWidth: 640 }}>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: 20, fontWeight: 600, marginBottom: 6 }}>Let the analyst investigate</div>
          <p style={{ margin: '0 0 16px', fontSize: 14, color: 'var(--ink-soft)', lineHeight: 1.6 }}>
            It profiles the data, checks trends, segments, relationships and anomalies, then reports the findings that matter —
            each with a confidence level and the calculations behind it. Takes about 30–60 seconds.
          </p>
          <Button variant="accent" onClick={() => run(false)}>Generate briefing</Button>
        </Card>
      )}
      {loading && <Working>The analyst is investigating your data…</Working>}
      {error && <ErrorNote>{error}</ErrorNote>}

      {b && !loading && (
        <>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 26, fontWeight: 600, lineHeight: 1.25, margin: '0 0 10px', maxWidth: 760, color: 'var(--ink)' }}>{b.headline}</h2>
          <div style={{ display: 'flex', gap: 8, marginBottom: 18, alignItems: 'center' }}>
            <GroundingBadge grounding={briefing.grounding} />
            {briefing.cached && <Badge>saved result</Badge>}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {b.insights.map((ins, i) => (
              <Card key={i} style={{ padding: '18px 20px' }}>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8, flexWrap: 'wrap' }}>
                  <Badge tone="accent">{TYPE_LABEL[ins.type] || ins.type}</Badge>
                  <ConfidenceBadge level={ins.confidence} />
                  {ins.unsupported && <Badge tone="alert">no supporting calculation</Badge>}
                </div>
                <div style={{ fontSize: 16, fontWeight: 650, color: 'var(--ink)', marginBottom: 6 }}>{ins.title}</div>
                <RichText text={ins.finding} />
                <div style={{ fontSize: 12, color: 'var(--ink-faint)', margin: '2px 0 8px' }}>Why {ins.confidence} confidence: {ins.confidence_reason}</div>
                <div style={{ fontSize: 13.5, color: 'var(--accent-ink)', fontWeight: 500 }}>→ {ins.action}</div>
                <EvidenceList evidence={briefing.evidence} ids={ins.evidence_ids} title="Evidence" />
              </Card>
            ))}
          </div>

          <Bullets title="Risks & caveats" items={b.risks} tone="var(--alert)" />
          {b.follow_up_questions?.length > 0 && (
            <div style={{ marginTop: 18 }}>
              <div style={{ fontSize: 11, fontWeight: 700, fontFamily: 'var(--font-mono)', letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--ink-faint)', marginBottom: 8 }}>Dig deeper</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                {b.follow_up_questions.map((q) => (
                  <button key={q} onClick={() => onAsk(q)} style={{ textAlign: 'left', padding: '9px 13px', borderRadius: 10, border: '1px solid var(--hairline-strong)', background: 'var(--paper-raised)', cursor: 'pointer', fontSize: 13.5, color: 'var(--ink)', fontFamily: 'var(--font-body)' }}>
                    {q} <span style={{ color: 'var(--accent)' }}>→</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </section>
  )
}

// ---------------------------------------------------------------- root cause
function RootCause({ datasetId, columns, rootCause, setRootCause, onAccessDenied }) {
  const numeric = columns.filter((c) => c.kind === 'numeric').map((c) => c.name)
  const dates = columns.filter((c) => c.kind === 'datetime').map((c) => c.name)
  const [question, setQuestion] = useState('')
  const [metric, setMetric] = useState('')
  const [dateCol, setDateCol] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const investigate = async () => {
    setLoading(true); setError(null)
    try {
      setRootCause(await api.rootCause(datasetId, { question, metric: metric || null, date_column: dateCol || null }))
    } catch (e) { if (e?.response?.status === 403) onAccessDenied(); setError(errorMessage(e)) }
    finally { setLoading(false) }
  }

  const r = rootCause?.structured
  const sel = { padding: '8px 10px', borderRadius: 8, border: '1px solid var(--hairline-strong)', background: 'var(--paper)', fontFamily: 'var(--font-mono)', fontSize: 12.5, color: 'var(--ink)', minWidth: 160 }
  const label = { fontSize: 11.5, color: 'var(--ink-faint)', fontWeight: 600, marginBottom: 5 }
  const example = numeric[0] ? `Why did ${numeric.find((n) => /revenue|sales|profit/i.test(n)) || numeric[0]} change recently?` : 'Why did the main metric change?'

  return (
    <section>
      <SectionLabel tone="accent">Root-cause investigator</SectionLabel>
      <Card style={{ padding: '18px 20px', marginBottom: 18 }}>
        <div style={label}>What do you want explained?</div>
        <textarea
          value={question} onChange={(e) => setQuestion(e.target.value)} rows={2} maxLength={2000}
          placeholder={example}
          style={{ width: '100%', resize: 'none', padding: '10px 12px', borderRadius: 10, border: '1px solid var(--hairline-strong)', fontFamily: 'var(--font-body)', fontSize: 14, background: 'var(--paper)', color: 'var(--ink)' }}
        />
        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'flex-end', marginTop: 12 }}>
          <div><div style={label}>Metric (optional)</div>
            <select value={metric} onChange={(e) => setMetric(e.target.value)} style={sel}>
              <option value="">Auto-detect</option>{numeric.map((c) => <option key={c}>{c}</option>)}
            </select></div>
          <div><div style={label}>Date column (optional)</div>
            <select value={dateCol} onChange={(e) => setDateCol(e.target.value)} style={sel}>
              <option value="">Auto-detect</option>{dates.map((c) => <option key={c}>{c}</option>)}
            </select></div>
          <Button variant="accent" onClick={investigate} disabled={loading || question.trim().length < 3}>
            {loading ? <Spinner size={14} /> : 'Investigate'}
          </Button>
        </div>
      </Card>

      {loading && <Working>Locating when it changed, then where…</Working>}
      {error && <ErrorNote>{error}</ErrorNote>}

      {r && !loading && (
        <Card style={{ padding: '22px 24px' }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
            <Badge tone={VERDICT[r.verdict]?.tone}>{VERDICT[r.verdict]?.label || r.verdict}</Badge>
            <ConfidenceBadge level={r.confidence} />
            <GroundingBadge grounding={rootCause.grounding} />
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--ink-faint)', marginBottom: 6 }}>{r.restated_question}</div>
          <RichText text={r.conclusion} size={15.5} />
          <div style={{ fontSize: 12, color: 'var(--ink-faint)', marginTop: 2 }}>Why {r.confidence} confidence: {r.confidence_reason}</div>

          {r.drivers?.length > 0 && (
            <div style={{ marginTop: 20 }}>
              <div style={{ fontSize: 11, fontWeight: 700, fontFamily: 'var(--font-mono)', letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--ink-faint)', marginBottom: 10 }}>Drivers</div>
              {r.drivers.map((d, i) => {
                const share = typeof d.share_of_change_pct === 'number' ? Math.min(Math.abs(d.share_of_change_pct), 100) : null
                return (
                  <div key={i} style={{ padding: '12px 0', borderTop: i ? '1px solid var(--hairline)' : 'none' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'baseline' }}>
                      <div style={{ fontWeight: 650, fontSize: 14.5, color: 'var(--ink)' }}>
                        <span style={{ color: d.direction === 'decrease' ? 'var(--alert)' : 'var(--positive)', marginRight: 6 }}>{d.direction === 'decrease' ? '▼' : '▲'}</span>{d.factor}
                      </div>
                      {share !== null && <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 700 }}>{d.share_of_change_pct}% of movement</span>}
                    </div>
                    {share !== null && (
                      <div style={{ height: 5, background: 'var(--hairline)', borderRadius: 3, margin: '8px 0' }}>
                        <div style={{ width: `${share}%`, height: '100%', background: d.direction === 'decrease' ? 'var(--alert)' : 'var(--positive)', borderRadius: 3 }} />
                      </div>
                    )}
                    <div style={{ fontSize: 13.5, color: 'var(--ink-soft)', lineHeight: 1.55 }}>{d.impact}</div>
                    {d.unsupported && <div style={{ marginTop: 6 }}><Badge tone="alert">no supporting calculation</Badge></div>}
                    <EvidenceList evidence={rootCause.evidence} ids={d.evidence_ids} title="Evidence" />
                  </div>
                )
              })}
            </div>
          )}

          <Bullets title="Checked and ruled out" items={r.ruled_out} />
          <Bullets title="Caveats" items={r.caveats} tone="var(--alert)" />
          <Bullets title="Recommended actions" items={r.recommended_actions} tone="var(--accent-ink)" />
          <EvidenceList evidence={rootCause.evidence} title="Full investigation" />
        </Card>
      )}
    </section>
  )
}

export default function InsightsTab({ datasetId, llm, columns, briefing, setBriefing, rootCause, setRootCause, onAsk, accessCode, onAccessCode, onAccessDenied, accessDenied }) {
  if (!llm) return <div style={{ padding: 60, display: 'flex', justifyContent: 'center' }}><Spinner size={22} /></div>
  if (!llm.configured) return <div><SectionLabel tone="accent">AI Insights</SectionLabel><LlmSetupCard status={llm} /></div>
  if (llm.requires_code && !accessCode) {
    return <div><SectionLabel tone="accent">AI Insights</SectionLabel><AccessCodeCard onSubmit={onAccessCode} denied={accessDenied} /></div>
  }
  return (
    <div>
      <Briefing datasetId={datasetId} briefing={briefing} setBriefing={setBriefing} onAsk={onAsk} onAccessDenied={onAccessDenied} />
      <RootCause datasetId={datasetId} columns={columns} rootCause={rootCause} setRootCause={setRootCause} onAccessDenied={onAccessDenied} />
    </div>
  )
}
