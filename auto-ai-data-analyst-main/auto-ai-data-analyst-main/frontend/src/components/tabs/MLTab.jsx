import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import { Card, SectionLabel, Badge, Button, Spinner, EmptyState, Stat } from '../Primitives'

const tooltipStyle = {
  background: 'var(--ink)', border: 'none', borderRadius: 8, fontSize: 12,
  fontFamily: 'var(--font-mono)', color: '#fff', padding: '8px 12px',
}

export default function MLTab({ columns, onRun, result, running }) {
  const [target, setTarget] = useState('')

  const handleRun = () => onRun(target || null)

  return (
    <div>
      <SectionLabel>Machine Learning</SectionLabel>
      <Card style={{ padding: '18px 20px', marginBottom: 24 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 220 }}>
            <div style={{ fontSize: 12, color: 'var(--ink-faint)', marginBottom: 6, fontWeight: 600 }}>
              Target column (leave blank to cluster)
            </div>
            <select
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              style={{
                width: '100%', padding: '9px 12px', borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--hairline-strong)', fontFamily: 'var(--font-mono)',
                fontSize: 13, background: 'var(--paper)', color: 'var(--ink)',
              }}
            >
              <option value="">— No target (clustering) —</option>
              {columns.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <Button variant="accent" onClick={handleRun} disabled={running}>
            {running ? <Spinner size={14} /> : 'Run analysis'}
          </Button>
        </div>
        <div style={{ fontSize: 12, color: 'var(--ink-faint)', marginTop: 10 }}>
          Numeric target with many unique values → regression · categorical or low-cardinality numeric → classification · no target → clustering
        </div>
      </Card>

      {running && (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: 60, gap: 12 }}>
          <Spinner size={22} />
          <div style={{ color: 'var(--ink-faint)', fontSize: 13 }}>Training and evaluating models…</div>
        </div>
      )}

      {!running && !result && (
        <Card><EmptyState title="No model run yet" description="Pick a target column and run the analysis, or leave it blank to auto-cluster the dataset." /></Card>
      )}

      {!running && result && result.error && (
        <Card><EmptyState title="Couldn't run" description={result.error} /></Card>
      )}

      {!running && result && !result.error && (
        <MLResult result={result} />
      )}
    </div>
  )
}

function MLResult({ result }) {
  if (result.problem_type === 'clustering') return <ClusteringResult result={result} />
  if (result.problem_type === 'regression') return <RegressionResult result={result} />
  return <ClassificationResult result={result} />
}

function ModelCompareTable({ models, primaryKey, primaryLabel }) {
  const cols = Object.keys(models[0]).filter((k) => k !== 'model')
  return (
    <Card style={{ overflow: 'hidden', marginBottom: 20 }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--hairline)' }}>
            <th style={{ textAlign: 'left', padding: '10px 16px', color: 'var(--ink-faint)', fontSize: 11, textTransform: 'uppercase' }}>Model</th>
            {cols.map((c) => (
              <th key={c} style={{ textAlign: 'right', padding: '10px 16px', color: 'var(--ink-faint)', fontSize: 11, textTransform: 'uppercase' }}>{c.replace(/_/g, ' ')}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((m, i) => (
            <tr key={m.model} style={{ borderBottom: i < models.length - 1 ? '1px solid var(--hairline)' : 'none', background: m[primaryKey] === Math.max(...models.map(x => x[primaryKey])) ? 'var(--accent-soft)' : 'transparent' }}>
              <td style={{ padding: '10px 16px', fontWeight: 600 }}>{m.model}</td>
              {cols.map((c) => (
                <td key={c} style={{ padding: '10px 16px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{m[c]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}

function FeatureImportance({ features }) {
  if (!features || features.length === 0) return null
  const data = [...features].reverse()
  return (
    <>
      <SectionLabel>Feature Importance</SectionLabel>
      <Card style={{ padding: '18px 20px', marginBottom: 24 }}>
        <ResponsiveContainer width="100%" height={Math.max(160, data.length * 32)}>
          <BarChart data={data} layout="vertical" margin={{ left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--hairline)" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 11, fontFamily: 'var(--font-mono)', fill: 'var(--ink-faint)' }} />
            <YAxis type="category" dataKey="feature" tick={{ fontSize: 11.5, fontFamily: 'var(--font-mono)', fill: 'var(--ink-soft)' }} width={120} />
            <Tooltip contentStyle={tooltipStyle} />
            <Bar dataKey="importance" fill="#5B4FE8" radius={[0, 3, 3, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </>
  )
}

function RegressionResult({ result }) {
  return (
    <div>
      <Card style={{ padding: '18px 20px', marginBottom: 20 }}>
        <div style={{ display: 'flex', gap: 32, marginBottom: 14, flexWrap: 'wrap' }}>
          <Stat label="Best model" value={result.best_model} mono={false} />
          <Stat label="R² score" value={result.best_r2} tone="accent" />
          <Stat label="Target" value={result.target} mono={false} />
        </div>
        <div style={{ fontSize: 14, color: 'var(--ink-soft)' }}>{result.summary}</div>
      </Card>
      <SectionLabel>Model Comparison</SectionLabel>
      <ModelCompareTable models={result.models_compared} primaryKey="r2_score" />
      <FeatureImportance features={result.feature_importance} />
    </div>
  )
}

function ClassificationResult({ result }) {
  return (
    <div>
      <Card style={{ padding: '18px 20px', marginBottom: 20 }}>
        <div style={{ display: 'flex', gap: 32, marginBottom: 14, flexWrap: 'wrap' }}>
          <Stat label="Best model" value={result.best_model} mono={false} />
          <Stat label="Accuracy" value={`${(result.best_accuracy * 100).toFixed(1)}%`} tone="accent" />
          <Stat label="Classes" value={result.class_names?.length} />
        </div>
        <div style={{ fontSize: 14, color: 'var(--ink-soft)' }}>{result.summary}</div>
      </Card>
      <SectionLabel>Model Comparison</SectionLabel>
      <ModelCompareTable models={result.models_compared} primaryKey="accuracy" />

      {result.confusion_matrix && (
        <>
          <SectionLabel>Confusion Matrix</SectionLabel>
          <Card style={{ padding: 20, marginBottom: 24, overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', margin: '0 auto' }}>
              <tbody>
                {result.confusion_matrix.map((row, i) => (
                  <tr key={i}>
                    {row.map((val, j) => (
                      <td key={j} style={{
                        width: 56, height: 56, textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 15,
                        background: i === j ? 'var(--positive-soft)' : 'var(--alert-soft)',
                        color: i === j ? 'var(--positive)' : 'var(--alert)',
                        border: '1px solid var(--paper)',
                      }}>
                        {val}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ textAlign: 'center', fontSize: 11.5, color: 'var(--ink-faint)', marginTop: 10 }}>
              Rows = actual class, columns = predicted class (index order: {result.class_names?.join(', ')})
            </div>
          </Card>
        </>
      )}
      <FeatureImportance features={result.feature_importance} />
    </div>
  )
}

function ClusteringResult({ result }) {
  return (
    <div>
      <Card style={{ padding: '18px 20px', marginBottom: 20 }}>
        <div style={{ display: 'flex', gap: 32, marginBottom: 14, flexWrap: 'wrap' }}>
          <Stat label="Clusters found" value={result.n_clusters} tone="accent" />
          <Stat label="Silhouette score" value={result.silhouette_score} />
        </div>
        <div style={{ fontSize: 14, color: 'var(--ink-soft)' }}>{result.summary}</div>
      </Card>

      <SectionLabel>Cluster Breakdown</SectionLabel>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 12 }}>
        {result.cluster_summary?.map((c) => (
          <Card key={c.cluster} style={{ padding: '16px 18px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 13 }}>Cluster {c.cluster}</span>
              <Badge tone="accent">{c.size} rows · {c.pct_of_data}%</Badge>
            </div>
            {Object.entries(c.mean_values || {}).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, padding: '3px 0', color: 'var(--ink-soft)' }}>
                <span>{k}</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>{v}</span>
              </div>
            ))}
          </Card>
        ))}
      </div>
    </div>
  )
}
