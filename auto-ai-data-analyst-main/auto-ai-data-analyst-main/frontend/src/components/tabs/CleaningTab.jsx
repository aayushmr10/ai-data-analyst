import React, { useState } from 'react'
import { Card, SectionLabel, Badge, Button, EmptyState, Spinner } from '../Primitives'

const SEVERITY_TONE = { high: 'alert', medium: 'amber', low: 'default' }

export default function CleaningTab({ profile, onApply, onApplyAll, onReset, applying, log }) {
  const [appliedSet, setAppliedSet] = useState(new Set())

  if (!profile) return null
  const suggestions = profile.cleaning_suggestions || []

  const handleApplyOne = async (s, idx) => {
    await onApply([{ action: s.action, column: s.column }])
    setAppliedSet((prev) => new Set(prev).add(idx))
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 14 }}>
        <SectionLabel>Data Quality &amp; Cleaning Suggestions</SectionLabel>
        <div style={{ display: 'flex', gap: 8 }}>
          <Button variant="ghost" size="sm" onClick={onReset} disabled={applying}>Reset to original</Button>
          <Button variant="accent" size="sm" onClick={onApplyAll} disabled={applying || suggestions.length === 0}>
            {applying ? <Spinner size={13} /> : `Apply all (${suggestions.length})`}
          </Button>
        </div>
      </div>

      {suggestions.length === 0 ? (
        <Card>
          <EmptyState title="No issues found" description="This dataset has no missing values, duplicate rows, or major outlier clusters. It's ready for analysis." />
        </Card>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 24 }}>
          {suggestions.map((s, idx) => {
            const isApplied = appliedSet.has(idx)
            return (
              <Card key={idx} style={{ padding: '16px 18px', opacity: isApplied ? 0.55 : 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16 }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                      {s.column && (
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 13 }}>{s.column}</span>
                      )}
                      <Badge tone={SEVERITY_TONE[s.severity]}>{s.severity}</Badge>
                    </div>
                    <div style={{ fontSize: 13.5, color: 'var(--ink)', marginBottom: 3 }}>{s.issue}</div>
                    <div style={{ fontSize: 13, color: 'var(--ink-soft)' }}>→ {s.suggestion}</div>
                  </div>
                  <Button
                    size="sm"
                    variant={isApplied ? 'subtle' : 'ghost'}
                    disabled={applying || isApplied || s.action === 'flag_outliers'}
                    onClick={() => handleApplyOne(s, idx)}
                  >
                    {isApplied ? 'Applied ✓' : s.action === 'flag_outliers' ? 'Informational' : 'Apply'}
                  </Button>
                </div>
              </Card>
            )
          })}
        </div>
      )}

      {log && log.length > 0 && (
        <>
          <SectionLabel>Cleaning Log</SectionLabel>
          <Card style={{ padding: '14px 18px' }}>
            {log.map((entry, i) => (
              <div key={i} style={{ fontSize: 12.5, fontFamily: 'var(--font-mono)', color: 'var(--ink-soft)', padding: '4px 0' }}>
                {entry}
              </div>
            ))}
          </Card>
        </>
      )}
    </div>
  )
}
