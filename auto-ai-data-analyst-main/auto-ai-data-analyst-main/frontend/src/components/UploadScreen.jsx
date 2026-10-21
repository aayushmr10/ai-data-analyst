import React, { useCallback, useRef, useState } from 'react'
import { SignalTicks, Spinner } from './Primitives'

export default function UploadScreen({ onUpload, loading, error }) {
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef(null)

  const handleFiles = useCallback((files) => {
    if (files && files[0]) onUpload(files[0])
  }, [onUpload])

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
        background:
          'radial-gradient(circle at 20% 15%, rgba(91,79,232,0.06), transparent 45%), radial-gradient(circle at 85% 80%, rgba(14,164,122,0.05), transparent 40%), var(--paper)',
      }}
    >
      <div style={{ maxWidth: 620, width: '100%', textAlign: 'center' }}>
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 22 }}>
          <svg width="70" height="34" viewBox="0 0 70 34" fill="none">
            <polyline
              points="0,26 10,26 15,8 22,32 29,14 34,26 70,26"
              stroke="var(--accent)"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
          </svg>
        </div>
        <h1
          style={{
            fontFamily: 'var(--font-display)',
            fontSize: 'clamp(32px, 5vw, 46px)',
            fontWeight: 600,
            margin: '0 0 12px',
            letterSpacing: '-0.01em',
            color: 'var(--ink)',
          }}
        >
          Bring a dataset.<br />Leave with a report.
        </h1>
        <p style={{ color: 'var(--ink-soft)', fontSize: 16, lineHeight: 1.6, maxWidth: 460, margin: '0 auto 36px' }}>
          Upload a CSV, Excel, or JSON file. This reads it, profiles it, cleans it, models it,
          and writes the summary — no SQL, no notebook.
        </p>

        <label
          onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragActive(false)
            handleFiles(e.dataTransfer.files)
          }}
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 10,
            padding: '52px 24px',
            border: `1.5px dashed ${dragActive ? 'var(--accent)' : 'var(--hairline-strong)'}`,
            borderRadius: 'var(--radius-lg)',
            background: dragActive ? 'var(--accent-soft)' : 'var(--paper-raised)',
            cursor: 'pointer',
            transition: 'border-color 0.15s ease, background 0.15s ease',
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.xlsx,.xls,.json,.tsv"
            style={{ display: 'none' }}
            onChange={(e) => handleFiles(e.target.files)}
          />
          {loading ? (
            <>
              <Spinner size={22} />
              <div style={{ color: 'var(--ink-soft)', fontSize: 14, marginTop: 4 }}>Reading and profiling your data…</div>
            </>
          ) : (
            <>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--accent-ink)', fontWeight: 600 }}>
                DROP FILE OR CLICK TO BROWSE
              </div>
              <div style={{ color: 'var(--ink-faint)', fontSize: 13 }}>.csv · .xlsx · .xls · .json · .tsv</div>
            </>
          )}
        </label>

        {error && (
          <div
            style={{
              marginTop: 16,
              padding: '10px 14px',
              background: 'var(--alert-soft)',
              color: 'var(--alert)',
              borderRadius: 'var(--radius-md)',
              fontSize: 13.5,
              textAlign: 'left',
            }}
          >
            {error}
          </div>
        )}

        <div
          style={{
            marginTop: 44,
            display: 'flex',
            justifyContent: 'center',
            gap: 28,
            flexWrap: 'wrap',
            fontFamily: 'var(--font-mono)',
            fontSize: 11.5,
            color: 'var(--ink-faint)',
            letterSpacing: '0.04em',
            textTransform: 'uppercase',
          }}
        >
          <span>Profiling</span>
          <span>·</span>
          <span>Statistics</span>
          <span>·</span>
          <span>Charts</span>
          <span>·</span>
          <span>ML Models</span>
          <span>·</span>
          <span>PDF Report</span>
        </div>
      </div>
    </div>
  )
}
