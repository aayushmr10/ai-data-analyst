import React from 'react'
import { SignalTicks } from './Primitives'

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'insights', label: 'AI Insights', ai: true },
  { id: 'chat', label: 'Ask the Analyst', ai: true },
  { id: 'cleaning', label: 'Data Quality' },
  { id: 'stats', label: 'Statistics' },
  { id: 'charts', label: 'Visualizations' },
  { id: 'correlation', label: 'Correlations' },
  { id: 'ml', label: 'ML Models' },
  { id: 'recommendations', label: 'Recommendations' },
  { id: 'report', label: 'Report' },
]

export default function Sidebar({ active, onSelect, filename, rowCount, colCount, onNewDataset }) {
  return (
    <div
      style={{
        width: 232,
        flexShrink: 0,
        borderRight: '1px solid var(--hairline)',
        background: 'var(--paper-raised)',
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        position: 'sticky',
        top: 0,
      }}
    >
      <div style={{ padding: '22px 20px 18px', borderBottom: '1px solid var(--hairline)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <svg width="20" height="12" viewBox="0 0 70 34" fill="none">
            <polyline
              points="0,26 10,26 15,8 22,32 29,14 34,26 70,26"
              stroke="var(--accent)" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" fill="none"
            />
          </svg>
          <div style={{ fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: 15.5 }}>Data Analyst</div>
        </div>
        <div
          title={filename}
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 11.5,
            color: 'var(--ink-soft)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            marginTop: 8,
          }}
        >
          {filename}
        </div>
        <div style={{ fontSize: 11, color: 'var(--ink-faint)', marginTop: 3, fontFamily: 'var(--font-mono)' }}>
          {rowCount?.toLocaleString()} rows · {colCount} cols
        </div>
      </div>

      <nav style={{ flex: 1, padding: '14px 10px', overflowY: 'auto' }}>
        {TABS.map((tab, idx) => {
          const isActive = active === tab.id
          const prev = TABS[idx - 1]
          const sectionLabel = tab.ai && !prev?.ai ? 'AI analyst' : !tab.ai && prev?.ai ? 'Analysis' : null
          return (
            <React.Fragment key={tab.id}>
              {sectionLabel && (
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase',
                  color: tab.ai ? 'var(--accent-ink)' : 'var(--ink-faint)', padding: '14px 12px 6px', fontWeight: 600 }}>
                  {sectionLabel}
                </div>
              )}
            <button
              onClick={() => onSelect(tab.id)}
              style={{
                display: 'block',
                width: '100%',
                textAlign: 'left',
                padding: '9px 12px',
                marginBottom: 2,
                borderRadius: 'var(--radius-sm)',
                border: 'none',
                background: isActive ? 'var(--accent-soft)' : 'transparent',
                color: isActive ? 'var(--accent-ink)' : 'var(--ink-soft)',
                fontWeight: isActive ? 600 : 500,
                fontSize: 13.5,
                cursor: 'pointer',
                transition: 'background 0.12s ease',
              }}
              onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = 'var(--paper)' }}
              onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = 'transparent' }}
            >
              {tab.label}
            </button>
            </React.Fragment>
          )
        })}
      </nav>

      <div style={{ padding: 14, borderTop: '1px solid var(--hairline)' }}>
        <button
          onClick={onNewDataset}
          style={{
            width: '100%',
            padding: '9px 12px',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--hairline-strong)',
            background: 'var(--paper)',
            color: 'var(--ink-soft)',
            fontSize: 12.5,
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          ← Upload new dataset
        </button>
      </div>
    </div>
  )
}
