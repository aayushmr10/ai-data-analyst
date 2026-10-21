import React, { useCallback, useEffect, useRef, useState } from 'react'
import api, { errorMessage } from '../../api/client'
import { Button, SectionLabel, Spinner } from '../Primitives'
import RichText from '../RichText'
import { AccessCodeCard, EvidenceList, GroundingBadge, LlmSetupCard } from '../AiParts'

export default function ChatTab({ datasetId, llm, messages, setMessages, pending, onPendingHandled, accessCode, onAccessCode, onAccessDenied, accessDenied }) {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [suggestions, setSuggestions] = useState([])
  const bottomRef = useRef(null)

  useEffect(() => {
    api.chatSuggestions(datasetId).then((r) => setSuggestions(r.questions || [])).catch(() => setSuggestions([]))
  }, [datasetId])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [messages, loading])

  const send = useCallback(async (text) => {
    const message = (text || '').trim()
    if (!message || loading) return
    const history = messages
      .filter((m) => !m.error && m.content)
      .map((m) => ({ role: m.role, content: m.content }))
    setMessages((prev) => [...prev, { role: 'user', content: message }])
    setInput('')
    setLoading(true)
    try {
      const res = await api.chat(datasetId, message, history)
      setMessages((prev) => [...prev, {
        role: 'assistant', content: res.answer || 'I could not produce an answer.',
        evidence: res.evidence, grounding: res.grounding, steps: res.steps,
      }])
    } catch (e) {
      if (e?.response?.status === 403) onAccessDenied()
      setMessages((prev) => [...prev, { role: 'assistant', error: true, content: errorMessage(e) }])
    } finally {
      setLoading(false)
    }
  }, [datasetId, loading, messages, setMessages, onAccessDenied])

  // A question handed over from another tab (e.g. a briefing follow-up)
  useEffect(() => {
    if (pending && llm?.configured && !loading) {
      onPendingHandled()
      send(pending)
    }
  }, [pending, llm, loading, onPendingHandled, send])

  if (!llm) return <div style={{ padding: 60, display: 'flex', justifyContent: 'center' }}><Spinner size={22} /></div>

  if (!llm.configured) {
    return (
      <div>
        <SectionLabel tone="accent">Ask the Analyst</SectionLabel>
        <LlmSetupCard status={llm} />
        {suggestions.length > 0 && (
          <div style={{ marginTop: 28, maxWidth: 620 }}>
            <div style={{ fontSize: 12, color: 'var(--ink-faint)', marginBottom: 8, fontWeight: 600 }}>Questions it could answer for this dataset</div>
            {suggestions.map((q) => <div key={q} style={{ fontSize: 13.5, color: 'var(--ink-soft)', padding: '7px 0', borderBottom: '1px solid var(--hairline)' }}>{q}</div>)}
          </div>
        )}
      </div>
    )
  }

  if (llm.requires_code && !accessCode) {
    return (
      <div>
        <SectionLabel tone="accent">Ask the Analyst</SectionLabel>
        <AccessCodeCard onSubmit={onAccessCode} denied={accessDenied} />
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 64px)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <SectionLabel tone="accent">Ask the Analyst</SectionLabel>
        {messages.length > 0 && <Button variant="subtle" size="sm" onClick={() => setMessages([])} disabled={loading}>Clear</Button>}
      </div>

      <div style={{ flex: 1, overflowY: 'auto', paddingRight: 6, paddingTop: 12, display: 'flex', flexDirection: 'column', gap: 14 }}>
        {messages.length === 0 && (
          <div style={{ maxWidth: 640 }}>
            <p style={{ margin: '0 0 6px', fontFamily: 'var(--font-display)', fontSize: 24, fontWeight: 600, color: 'var(--ink)' }}>
              Ask anything about this dataset.
            </p>
            <p style={{ margin: '0 0 20px', fontSize: 14, color: 'var(--ink-soft)', lineHeight: 1.6 }}>
              The analyst runs real calculations on your data to answer — and shows its working, so you can check every number.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {suggestions.map((q) => (
                <button key={q} onClick={() => send(q)} style={{
                  textAlign: 'left', padding: '11px 14px', borderRadius: 10, border: '1px solid var(--hairline-strong)',
                  background: 'var(--paper-raised)', cursor: 'pointer', fontSize: 13.5, color: 'var(--ink)', fontFamily: 'var(--font-body)',
                }}>{q}</button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => m.role === 'user' ? (
          <div key={i} style={{ alignSelf: 'flex-end', maxWidth: '78%', background: 'var(--ink)', color: 'var(--paper)', padding: '10px 15px', borderRadius: '14px 14px 4px 14px', fontSize: 14, lineHeight: 1.55 }}>
            {m.content}
          </div>
        ) : (
          <div key={i} style={{
            alignSelf: 'flex-start', maxWidth: '88%', padding: '14px 18px', borderRadius: '14px 14px 14px 4px',
            background: m.error ? 'var(--alert-soft)' : 'var(--paper-raised)', border: `1px solid ${m.error ? 'transparent' : 'var(--hairline)'}`,
          }}>
            {m.error ? <div style={{ color: 'var(--alert)', fontSize: 13.5 }}>{m.content}</div> : <RichText text={m.content} />}
            {!m.error && (
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginTop: 6 }}>
                <GroundingBadge grounding={m.grounding} />
              </div>
            )}
            {!m.error && <EvidenceList evidence={m.evidence} />}
          </div>
        ))}

        {loading && (
          <div style={{ alignSelf: 'flex-start', display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px', color: 'var(--ink-faint)', fontSize: 13 }}>
            <Spinner size={15} /> Running calculations…
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div style={{ display: 'flex', gap: 10, paddingTop: 14, alignItems: 'flex-end' }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) } }}
          placeholder="Ask a question about your data…  (Enter to send, Shift+Enter for a new line)"
          rows={2}
          maxLength={2000}
          style={{ flex: 1, resize: 'none', padding: '11px 14px', borderRadius: 12, border: '1px solid var(--hairline-strong)', fontFamily: 'var(--font-body)', fontSize: 14, background: 'var(--paper-raised)', color: 'var(--ink)' }}
        />
        <Button variant="accent" onClick={() => send(input)} disabled={loading || !input.trim()}>Send</Button>
      </div>
    </div>
  )
}
