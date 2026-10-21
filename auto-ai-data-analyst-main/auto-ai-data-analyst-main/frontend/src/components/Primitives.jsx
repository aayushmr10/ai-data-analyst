import React from 'react'

export function Card({ children, style, className = '' }) {
  return (
    <div
      className={className}
      style={{
        background: 'var(--paper-raised)',
        border: '1px solid var(--hairline)',
        borderRadius: 'var(--radius-lg)',
        boxShadow: 'var(--shadow-sm)',
        ...style,
      }}
    >
      {children}
    </div>
  )
}

export function SectionLabel({ children, tone = 'default' }) {
  const colorMap = {
    default: 'var(--ink-soft)',
    accent: 'var(--accent-ink)',
  }
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        color: colorMap[tone],
        fontWeight: 600,
        marginBottom: 14,
      }}
    >
      <SignalTicks />
      {children}
    </div>
  )
}

/** Signature element: a small oscilloscope-style tick strip used before section labels. */
export function SignalTicks({ animated = false }) {
  return (
    <svg width="22" height="12" viewBox="0 0 22 12" fill="none" style={{ flexShrink: 0 }}>
      <polyline
        points="0,9 4,9 6,3 9,11 12,5 14,9 22,9"
        stroke="var(--accent)"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
        opacity={animated ? 1 : 0.55}
      >
        {animated && (
          <animate attributeName="opacity" values="0.4;1;0.4" dur="1.6s" repeatCount="indefinite" />
        )}
      </polyline>
    </svg>
  )
}

export function Stat({ label, value, unit, tone = 'default', mono = true }) {
  const toneColor = {
    default: 'var(--ink)',
    accent: 'var(--accent-ink)',
    positive: 'var(--positive)',
    alert: 'var(--alert)',
  }[tone]
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--ink-faint)', marginBottom: 4, fontWeight: 500 }}>{label}</div>
      <div
        style={{
          fontFamily: mono ? 'var(--font-mono)' : 'var(--font-display)',
          fontSize: 26,
          fontWeight: 600,
          color: toneColor,
          lineHeight: 1.1,
        }}
      >
        {value}
        {unit && <span style={{ fontSize: 14, color: 'var(--ink-faint)', marginLeft: 4, fontWeight: 500 }}>{unit}</span>}
      </div>
    </div>
  )
}

export function Badge({ children, tone = 'default' }) {
  const styles = {
    default: { bg: 'var(--hairline)', fg: 'var(--ink-soft)' },
    accent: { bg: 'var(--accent-soft)', fg: 'var(--accent-ink)' },
    positive: { bg: 'var(--positive-soft)', fg: 'var(--positive)' },
    alert: { bg: 'var(--alert-soft)', fg: 'var(--alert)' },
    amber: { bg: 'var(--amber-soft)', fg: 'var(--amber)' },
  }[tone]
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '3px 9px',
        borderRadius: 100,
        fontSize: 11,
        fontWeight: 600,
        fontFamily: 'var(--font-mono)',
        background: styles.bg,
        color: styles.fg,
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </span>
  )
}

export function Button({ children, onClick, variant = 'primary', disabled, style, type = 'button', size = 'md' }) {
  const base = {
    fontFamily: 'var(--font-body)',
    fontWeight: 600,
    borderRadius: 'var(--radius-md)',
    cursor: disabled ? 'not-allowed' : 'pointer',
    border: '1px solid transparent',
    transition: 'transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease',
    opacity: disabled ? 0.5 : 1,
    fontSize: size === 'sm' ? 13 : 14,
    padding: size === 'sm' ? '7px 12px' : '10px 18px',
  }
  const variants = {
    primary: { background: 'var(--ink)', color: 'var(--paper)' },
    accent: { background: 'var(--accent)', color: '#fff' },
    ghost: { background: 'transparent', color: 'var(--ink)', border: '1px solid var(--hairline-strong)' },
    subtle: { background: 'var(--paper)', color: 'var(--ink-soft)', border: '1px solid var(--hairline)' },
  }
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      style={{ ...base, ...variants[variant], ...style }}
      onMouseEnter={(e) => { if (!disabled) e.currentTarget.style.transform = 'translateY(-1px)' }}
      onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)' }}
    >
      {children}
    </button>
  )
}

export function EmptyState({ title, description }) {
  return (
    <div style={{ padding: '48px 24px', textAlign: 'center', color: 'var(--ink-faint)' }}>
      <div style={{ fontFamily: 'var(--font-display)', fontSize: 18, color: 'var(--ink-soft)', marginBottom: 6 }}>{title}</div>
      <div style={{ fontSize: 13.5, maxWidth: 340, margin: '0 auto' }}>{description}</div>
    </div>
  )
}

export function Spinner({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" style={{ animation: 'spin 0.8s linear infinite' }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <circle cx="12" cy="12" r="9" stroke="var(--hairline-strong)" strokeWidth="3" fill="none" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="var(--accent)" strokeWidth="3" fill="none" strokeLinecap="round" />
    </svg>
  )
}
