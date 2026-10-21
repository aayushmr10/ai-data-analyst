import React from 'react'

/**
 * Tiny markdown subset: **bold**, `code`, "- " / "* " bullets, "1. " lists, paragraphs.
 * Builds React elements only (never innerHTML), so model output can't inject markup.
 */
function inline(text, keyPrefix) {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean).map((part, i) => {
    const key = `${keyPrefix}-${i}`
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return <strong key={key} style={{ fontWeight: 650, color: 'var(--ink)' }}>{part.slice(2, -2)}</strong>
    }
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      return (
        <code key={key} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9em', background: 'var(--paper)',
          border: '1px solid var(--hairline)', borderRadius: 4, padding: '1px 5px' }}>
          {part.slice(1, -1)}
        </code>
      )
    }
    return <React.Fragment key={key}>{part}</React.Fragment>
  })
}

export default function RichText({ text, size = 14 }) {
  if (!text) return null
  const blocks = []
  let list = null

  const flush = () => { if (list) { blocks.push(list); list = null } }

  text.split('\n').forEach((raw, idx) => {
    const line = raw.trimEnd()
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/)
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/)
    if (bullet || numbered) {
      const kind = bullet ? 'ul' : 'ol'
      if (!list || list.kind !== kind) { flush(); list = { kind, items: [] } }
      list.items.push((bullet || numbered)[1])
    } else if (line.trim() === '') {
      flush()
    } else {
      flush()
      blocks.push({ kind: 'p', text: line.trim(), idx })
    }
  })
  flush()

  return (
    <div style={{ fontSize: size, lineHeight: 1.6, color: 'var(--ink-soft)' }}>
      {blocks.map((b, i) => {
        if (b.kind === 'p') return <p key={i} style={{ margin: '0 0 8px' }}>{inline(b.text, `p${i}`)}</p>
        const Tag = b.kind
        return (
          <Tag key={i} style={{ margin: '0 0 8px', paddingLeft: 20 }}>
            {b.items.map((it, j) => <li key={j} style={{ marginBottom: 3 }}>{inline(it, `l${i}-${j}`)}</li>)}
          </Tag>
        )
      })}
    </div>
  )
}
