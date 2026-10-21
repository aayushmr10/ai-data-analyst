import React from 'react'
import { Card, SectionLabel, EmptyState, Badge } from '../Primitives'

function cellColor(r) {
  if (r === null || r === undefined) return 'var(--paper)'
  const abs = Math.abs(r)
  const alpha = 0.08 + abs * 0.55
  return r >= 0 ? `rgba(91, 79, 232, ${alpha})` : `rgba(220, 91, 62, ${alpha})`
}

export default function CorrelationTab({ correlation }) {
  if (!correlation) return null
  const { columns = [], matrix = {}, top_pairs = [] } = correlation

  if (columns.length < 2) {
    return <Card><EmptyState title="Not enough numeric columns" description="Correlation analysis needs at least two numeric columns." /></Card>
  }

  return (
    <div>
      <SectionLabel>Correlation Matrix</SectionLabel>
      <Card style={{ padding: '20px', marginBottom: 28, overflow: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', fontSize: 11.5 }}>
          <thead>
            <tr>
              <th />
              {columns.map((c) => (
                <th key={c} style={{ padding: '4px 8px', fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--ink-soft)', writingMode: 'vertical-rl', textOrientation: 'mixed', maxHeight: 90, fontSize: 10.5 }}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {columns.map((rowCol) => (
              <tr key={rowCol}>
                <td style={{ padding: '4px 10px', fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--ink-soft)', whiteSpace: 'nowrap', fontSize: 10.5 }}>{rowCol}</td>
                {columns.map((colCol) => {
                  const v = matrix[rowCol]?.[colCol]
                  return (
                    <td
                      key={colCol}
                      title={`${rowCol} × ${colCol}: ${v ?? 'n/a'}`}
                      style={{
                        width: 46, height: 34, textAlign: 'center',
                        background: cellColor(v),
                        fontFamily: 'var(--font-mono)', fontSize: 10.5,
                        color: Math.abs(v ?? 0) > 0.6 ? '#fff' : 'var(--ink)',
                        border: '1px solid var(--paper)',
                      }}
                    >
                      {v !== null && v !== undefined ? v.toFixed(2) : '—'}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <SectionLabel>Strongest Relationships</SectionLabel>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {top_pairs.map((p, i) => (
          <Card key={i} style={{ padding: '14px 18px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16 }}>
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 13, marginBottom: 3 }}>
                {p.column_a} <span style={{ color: 'var(--ink-faint)', fontWeight: 400 }}>×</span> {p.column_b}
              </div>
              <div style={{ fontSize: 13, color: 'var(--ink-soft)' }}>{p.explanation}</div>
            </div>
            <Badge tone={Math.abs(p.correlation) >= 0.7 ? 'accent' : Math.abs(p.correlation) >= 0.4 ? 'amber' : 'default'}>
              r = {p.correlation}
            </Badge>
          </Card>
        ))}
      </div>
    </div>
  )
}
