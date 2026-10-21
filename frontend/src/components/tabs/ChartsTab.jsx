import React, { useEffect, useState, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, ScatterChart, Scatter, LineChart, Line,
} from 'recharts'
import { Card, SectionLabel, EmptyState, Spinner } from '../Primitives'
import api from '../../api/client'

const PIE_COLORS = ['#5B4FE8', '#0EA47A', '#DC5B3E', '#B8860B', '#8A7FF0', '#4FA88E', '#C97B60', '#9A9CA3']

function ChartCard({ title, children, subtitle }) {
  return (
    <Card style={{ padding: '18px 20px', marginBottom: 20 }}>
      <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 13, marginBottom: 2 }}>{title}</div>
      {subtitle && <div style={{ fontSize: 12, color: 'var(--ink-faint)', marginBottom: 14 }}>{subtitle}</div>}
      {!subtitle && <div style={{ marginBottom: 14 }} />}
      {children}
    </Card>
  )
}

const tooltipStyle = {
  background: 'var(--ink)', border: 'none', borderRadius: 8, fontSize: 12,
  fontFamily: 'var(--font-mono)', color: '#fff', padding: '8px 12px',
}

export default function ChartsTab({ datasetId }) {
  const [suggestions, setSuggestions] = useState(null)
  const [chartData, setChartData] = useState({})
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    const sug = await api.getChartSuggestions(datasetId)
    setSuggestions(sug)

    const results = {}
    const jobs = []

    for (const col of sug.suggested.histograms || []) {
      jobs.push(api.chartHistogram(datasetId, col).then((d) => { results[`hist_${col}`] = d }))
    }
    for (const col of sug.suggested.bar_charts || []) {
      jobs.push(api.chartBar(datasetId, col).then((d) => { results[`bar_${col}`] = d }))
    }
    for (const col of sug.suggested.pie_charts || []) {
      jobs.push(api.chartPie(datasetId, col).then((d) => { results[`pie_${col}`] = d }))
    }
    const scatterPairs = (sug.suggested.scatter_pairs || []).slice(0, 2)
    for (const [x, y] of scatterPairs) {
      jobs.push(api.chartScatter(datasetId, x, y).then((d) => { results[`scatter_${x}_${y}`] = d }))
    }
    for (const ts of sug.suggested.timeseries || []) {
      jobs.push(api.chartTimeseries(datasetId, ts.date, ts.value).then((d) => { results[`ts_${ts.date}_${ts.value}`] = d }))
    }

    await Promise.all(jobs)
    setChartData(results)
    setLoading(false)
  }, [datasetId])

  useEffect(() => { load().catch(() => setLoading(false)) }, [load])

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: 80, gap: 12 }}>
        <Spinner size={22} />
        <div style={{ color: 'var(--ink-faint)', fontSize: 13 }}>Generating visualizations…</div>
      </div>
    )
  }

  if (!suggestions) return null

  const { histograms = [], bar_charts = [], pie_charts = [], timeseries = [] } = suggestions.suggested
  const scatterPairs = (suggestions.suggested.scatter_pairs || []).slice(0, 2)
  const hasAny = histograms.length || bar_charts.length || pie_charts.length || timeseries.length || scatterPairs.length

  if (!hasAny) {
    return <Card><EmptyState title="Not enough data to chart" description="Add numeric or categorical columns to generate visualizations." /></Card>
  }

  return (
    <div>
      <SectionLabel>Visualizations</SectionLabel>

      {timeseries.map((ts) => {
        const d = chartData[`ts_${ts.date}_${ts.value}`]
        if (!d || !d.data?.length) return null
        return (
          <ChartCard key={`ts_${ts.date}`} title={`${ts.value} over time`} subtitle={`Grouped by ${d.frequency === 'M' ? 'month' : 'day'}`}>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={d.data}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--hairline)" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 11, fontFamily: 'var(--font-mono)', fill: 'var(--ink-faint)' }} />
                <YAxis tick={{ fontSize: 11, fontFamily: 'var(--font-mono)', fill: 'var(--ink-faint)' }} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="value" stroke="#5B4FE8" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        )
      })}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: 0 }}>
        {histograms.map((col) => {
          const d = chartData[`hist_${col}`]
          if (!d || !d.bins?.length) return null
          return (
            <ChartCard key={`hist_${col}`} title={`Distribution — ${col}`}>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={d.bins}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--hairline)" vertical={false} />
                  <XAxis dataKey="bin" tick={false} />
                  <YAxis tick={{ fontSize: 11, fontFamily: 'var(--font-mono)', fill: 'var(--ink-faint)' }} width={32} />
                  <Tooltip contentStyle={tooltipStyle} labelFormatter={() => ''} formatter={(v, n, p) => [v, p.payload.bin]} />
                  <Bar dataKey="count" fill="#5B4FE8" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          )
        })}

        {bar_charts.map((col) => {
          const d = chartData[`bar_${col}`]
          if (!d || !d.data?.length) return null
          return (
            <ChartCard key={`bar_${col}`} title={`Top values — ${col}`}>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={d.data} layout="vertical" margin={{ left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--hairline)" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 11, fontFamily: 'var(--font-mono)', fill: 'var(--ink-faint)' }} />
                  <YAxis type="category" dataKey="category" tick={{ fontSize: 11, fontFamily: 'var(--font-body)', fill: 'var(--ink-soft)' }} width={90} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="count" fill="#0EA47A" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          )
        })}

        {pie_charts.map((col) => {
          const d = chartData[`pie_${col}`]
          if (!d || !d.data?.length) return null
          return (
            <ChartCard key={`pie_${col}`} title={`Share — ${col}`}>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={d.data} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={75} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`} labelLine={false} style={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}>
                    {d.data.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                  </Pie>
                  <Tooltip contentStyle={tooltipStyle} />
                </PieChart>
              </ResponsiveContainer>
            </ChartCard>
          )
        })}

        {scatterPairs.map(([x, y]) => {
          const d = chartData[`scatter_${x}_${y}`]
          if (!d || !d.data?.length) return null
          return (
            <ChartCard key={`scatter_${x}_${y}`} title={`${x} vs ${y}`}>
              <ResponsiveContainer width="100%" height={220}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--hairline)" />
                  <XAxis dataKey="x" name={x} tick={{ fontSize: 11, fontFamily: 'var(--font-mono)', fill: 'var(--ink-faint)' }} />
                  <YAxis dataKey="y" name={y} tick={{ fontSize: 11, fontFamily: 'var(--font-mono)', fill: 'var(--ink-faint)' }} />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter data={d.data} fill="#DC5B3E" fillOpacity={0.55} />
                </ScatterChart>
              </ResponsiveContainer>
            </ChartCard>
          )
        })}
      </div>
    </div>
  )
}
