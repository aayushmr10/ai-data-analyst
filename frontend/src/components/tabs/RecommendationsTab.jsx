import React from 'react'
import { Card, SectionLabel, EmptyState, Spinner } from '../Primitives'

export default function RecommendationsTab({ recommendations, loading }) {
  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: 60, gap: 12 }}>
        <Spinner size={22} />
        <div style={{ color: 'var(--ink-faint)', fontSize: 13 }}>Generating recommendations…</div>
      </div>
    )
  }

  if (!recommendations || recommendations.length === 0) {
    return <Card><EmptyState title="No recommendations yet" description="Run the ML analysis first for more targeted recommendations." /></Card>
  }

  return (
    <div>
      <SectionLabel>Recommendations</SectionLabel>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {recommendations.map((r, i) => (
          <Card key={i} style={{ padding: '16px 18px', display: 'flex', gap: 14, alignItems: 'flex-start' }}>
            <div
              style={{
                flexShrink: 0,
                width: 26, height: 26, borderRadius: '50%',
                background: 'var(--accent-soft)', color: 'var(--accent-ink)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 12,
              }}
            >
              {i + 1}
            </div>
            <div style={{ fontSize: 14, color: 'var(--ink)', lineHeight: 1.55, paddingTop: 3 }}>{r}</div>
          </Card>
        ))}
      </div>
    </div>
  )
}
