import React from 'react'
import { Card, SectionLabel, Stat, Badge } from '../Primitives'

const KIND_TONE = {
  numeric: 'accent',
  categorical: 'default',
  datetime: 'positive',
  boolean: 'amber',
}

export default function OverviewTab({ profile, preview }) {
  if (!profile) return null
  const { overview, columns } = profile

  return (
    <div>
      <SectionLabel>Dataset Overview</SectionLabel>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: 1,
          background: 'var(--hairline)',
          border: '1px solid var(--hairline)',
          borderRadius: 'var(--radius-lg)',
          overflow: 'hidden',
          marginBottom: 32,
        }}
      >
        {[
          { label: 'Rows', value: overview.n_rows.toLocaleString() },
          { label: 'Columns', value: overview.n_columns },
          { label: 'Numeric', value: overview.numeric_columns, tone: 'accent' },
          { label: 'Categorical', value: overview.categorical_columns },
          { label: 'Missing', value: `${overview.missing_pct}%`, tone: overview.missing_pct > 10 ? 'alert' : 'default' },
          { label: 'Duplicates', value: overview.duplicate_rows, tone: overview.duplicate_rows > 0 ? 'amber' : 'default' },
        ].map((s) => (
          <div key={s.label} style={{ background: 'var(--paper-raised)', padding: '18px 18px' }}>
            <Stat label={s.label} value={s.value} tone={s.tone || 'default'} />
          </div>
        ))}
      </div>

      <SectionLabel>Columns</SectionLabel>
      <Card style={{ marginBottom: 32, overflow: 'hidden' }}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--hairline)' }}>
                {['Column', 'Type', 'Missing', 'Unique', 'Detail'].map((h) => (
                  <th key={h} style={{ textAlign: 'left', padding: '10px 16px', color: 'var(--ink-faint)', fontWeight: 600, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {columns.map((col, i) => (
                <tr key={col.name} style={{ borderBottom: i < columns.length - 1 ? '1px solid var(--hairline)' : 'none' }}>
                  <td style={{ padding: '10px 16px', fontWeight: 600, fontFamily: 'var(--font-mono)', fontSize: 12.5 }}>{col.name}</td>
                  <td style={{ padding: '10px 16px' }}>
                    <Badge tone={KIND_TONE[col.kind] || 'default'}>{col.kind}</Badge>
                  </td>
                  <td style={{ padding: '10px 16px', color: col.missing_pct > 10 ? 'var(--alert)' : 'var(--ink-soft)', fontFamily: 'var(--font-mono)' }}>
                    {col.missing_pct}%
                  </td>
                  <td style={{ padding: '10px 16px', color: 'var(--ink-soft)', fontFamily: 'var(--font-mono)' }}>{col.unique_count}</td>
                  <td style={{ padding: '10px 16px', color: 'var(--ink-faint)', fontSize: 12.5 }}>
                    {col.kind === 'numeric' && col.mean !== undefined
                      ? `mean ${col.mean.toFixed(2)} · range ${col.min?.toFixed(1)}–${col.max?.toFixed(1)}`
                      : col.top_values?.[0]
                      ? `top: ${col.top_values[0].value} (${col.top_values[0].count})`
                      : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {preview && (
        <>
          <SectionLabel>Data Preview</SectionLabel>
          <Card style={{ overflow: 'hidden' }}>
            <div style={{ overflowX: 'auto', maxHeight: 420 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12.5 }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--hairline)', position: 'sticky', top: 0, background: 'var(--paper-raised)' }}>
                    {preview.columns.map((c) => (
                      <th key={c} style={{ textAlign: 'left', padding: '9px 14px', color: 'var(--ink-faint)', fontWeight: 600, fontFamily: 'var(--font-mono)', fontSize: 11, whiteSpace: 'nowrap' }}>
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.rows.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid var(--hairline)' }}>
                      {preview.columns.map((c) => (
                        <td key={c} style={{ padding: '8px 14px', color: 'var(--ink)', whiteSpace: 'nowrap', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                          {row[c] === null || row[c] === undefined ? <span style={{ color: 'var(--ink-faint)' }}>—</span> : String(row[c])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ padding: '10px 16px', fontSize: 11.5, color: 'var(--ink-faint)', borderTop: '1px solid var(--hairline)' }}>
              Showing {preview.rows.length} of {preview.total_rows.toLocaleString()} rows
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
