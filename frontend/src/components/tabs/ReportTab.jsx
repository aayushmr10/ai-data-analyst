import React, { useState } from 'react'
import { Card, SectionLabel, Button, Spinner } from '../Primitives'

export default function ReportTab({ columns, onGenerate, downloadUrl, generating }) {
  const [target, setTarget] = useState('')
  const [includeML, setIncludeML] = useState(true)

  return (
    <div>
      <SectionLabel>Generate Report</SectionLabel>
      <Card style={{ padding: '28px 30px', maxWidth: 560 }}>
        <div style={{ fontFamily: 'var(--font-display)', fontSize: 19, fontWeight: 600, marginBottom: 8 }}>
          Full PDF report
        </div>
        <div style={{ fontSize: 13.5, color: 'var(--ink-soft)', lineHeight: 1.6, marginBottom: 22 }}>
          Includes executive summary, dataset overview, data quality findings, charts, statistics,
          correlation analysis, model performance, and recommendations — built from the current
          state of your dataset (any cleaning you've applied is included).
        </div>

        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 12, color: 'var(--ink-faint)', marginBottom: 6, fontWeight: 600 }}>
            ML target column (optional)
          </div>
          <select
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            disabled={!includeML}
            style={{
              width: '100%', padding: '9px 12px', borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--hairline-strong)', fontFamily: 'var(--font-mono)',
              fontSize: 13, background: includeML ? 'var(--paper)' : 'var(--hairline)', color: 'var(--ink)',
            }}
          >
            <option value="">— No target (clustering) —</option>
            {columns.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>

        <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 22, cursor: 'pointer', fontSize: 13.5, color: 'var(--ink-soft)' }}>
          <input type="checkbox" checked={includeML} onChange={(e) => setIncludeML(e.target.checked)} />
          Include machine learning section
        </label>

        <div style={{ display: 'flex', gap: 10 }}>
          <Button variant="accent" onClick={() => onGenerate(target || null, includeML)} disabled={generating}>
            {generating ? <Spinner size={14} /> : 'Generate PDF report'}
          </Button>
          {downloadUrl && !generating && (
            <a href={downloadUrl} download style={{ textDecoration: 'none' }}>
              <Button variant="primary">Download report ↓</Button>
            </a>
          )}
        </div>
      </Card>
    </div>
  )
}
