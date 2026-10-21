import React from 'react'
import { Card, SectionLabel, EmptyState } from '../Primitives'

const METRICS = [
  { key: 'mean', label: 'Mean' },
  { key: 'median', label: 'Median' },
  { key: 'std', label: 'Std Dev' },
  { key: 'min', label: 'Min' },
  { key: 'max', label: 'Max' },
  { key: 'q1', label: 'Q1' },
  { key: 'q3', label: 'Q3' },
  { key: 'skewness', label: 'Skew' },
]

export default function StatsTab({ stats }) {
  if (!stats) return null
  const descriptive = stats.descriptive || []

  if (descriptive.length === 0) {
    return <Card><EmptyState title="No numeric columns" description="Statistical analysis requires at least one numeric column." /></Card>
  }

  return (
    <div>
      <SectionLabel>Statistical Analysis</SectionLabel>
      <Card style={{ overflow: 'hidden', marginBottom: 24 }}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12.5 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--hairline)' }}>
                <th style={{ textAlign: 'left', padding: '10px 16px', color: 'var(--ink-faint)', fontWeight: 600, fontSize: 11, textTransform: 'uppercase' }}>Column</th>
                {METRICS.map((m) => (
                  <th key={m.key} style={{ textAlign: 'right', padding: '10px 14px', color: 'var(--ink-faint)', fontWeight: 600, fontSize: 11, textTransform: 'uppercase' }}>
                    {m.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {descriptive.map((row, i) => (
                <tr key={row.column} style={{ borderBottom: i < descriptive.length - 1 ? '1px solid var(--hairline)' : 'none' }}>
                  <td style={{ padding: '10px 16px', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>{row.column}</td>
                  {METRICS.map((m) => (
                    <td key={m.key} style={{ padding: '10px 14px', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--ink-soft)' }}>
                      {row[m.key] !== null && row[m.key] !== undefined ? row[m.key].toFixed(2) : '—'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <SectionLabel>Distribution Notes</SectionLabel>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
        {descriptive.map((row) => (
          <Card key={row.column} style={{ padding: '14px 16px' }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 13, marginBottom: 4 }}>{row.column}</div>
            <div style={{ fontSize: 13, color: 'var(--ink-soft)' }}>{row.summary}</div>
          </Card>
        ))}
      </div>
    </div>
  )
}
