import React, { useState, useCallback, useEffect } from 'react'
import api, { getAccessCode, setSessionExpiredHandler, storeAccessCode } from './api/client'
import UploadScreen from './components/UploadScreen'
import Sidebar from './components/Sidebar'
import OverviewTab from './components/tabs/OverviewTab'
import CleaningTab from './components/tabs/CleaningTab'
import StatsTab from './components/tabs/StatsTab'
import ChartsTab from './components/tabs/ChartsTab'
import CorrelationTab from './components/tabs/CorrelationTab'
import MLTab from './components/tabs/MLTab'
import RecommendationsTab from './components/tabs/RecommendationsTab'
import ReportTab from './components/tabs/ReportTab'
import ChatTab from './components/tabs/ChatTab'
import InsightsTab from './components/tabs/InsightsTab'
import { Spinner } from './components/Primitives'

export default function App() {
  const [dataset, setDataset] = useState(null) // { id, filename, profile, preview }
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')

  // per-tab lazy-loaded state
  const [stats, setStats] = useState(null)
  const [correlation, setCorrelation] = useState(null)
  const [mlResult, setMlResult] = useState(null)
  const [mlRunning, setMlRunning] = useState(false)
  const [recommendations, setRecommendations] = useState(null)
  const [recLoading, setRecLoading] = useState(false)
  const [cleaningLog, setCleaningLog] = useState([])
  const [applying, setApplying] = useState(false)
  const [reportUrl, setReportUrl] = useState(null)
  const [reportGenerating, setReportGenerating] = useState(false)

  // AI analyst state (lifted here so it survives switching tabs)
  const [llm, setLlm] = useState(null) // { configured, model }
  const [chatMessages, setChatMessages] = useState([])
  const [briefing, setBriefing] = useState(null)
  const [rootCause, setRootCause] = useState(null)
  const [pendingQuestion, setPendingQuestion] = useState(null)

  useEffect(() => {
    api.llmStatus().then(setLlm).catch(() => setLlm({ configured: false, model: null }))
  }, [])

  // AI answers describe a specific version of the data -- drop them when it changes.
  const invalidateAI = useCallback(() => {
    setChatMessages([]); setBriefing(null); setRootCause(null); setPendingQuestion(null)
  }, [])

  // Access code for servers that protect their AI key (APP_ACCESS_CODE)
  const [accessCode, setAccessCode] = useState(getAccessCode)
  const [accessDenied, setAccessDenied] = useState(false)
  const handleAccessCode = useCallback((code) => { storeAccessCode(code); setAccessCode(code); setAccessDenied(false) }, [])
  const handleAccessDenied = useCallback(() => { storeAccessCode(''); setAccessCode(''); setAccessDenied(true) }, [])

  // Free hosts restart or evict idle sessions. Recover by returning to the upload screen with an explanation.
  useEffect(() => {
    setSessionExpiredHandler(() => {
      setDataset(null)
      setStats(null); setCorrelation(null); setMlResult(null); setRecommendations(null)
      setCleaningLog([]); setReportUrl(null); setActiveTab('overview')
      invalidateAI()
      setUploadError('Your session expired (the server was idle or restarted). Please upload your file again.')
    })
    return () => setSessionExpiredHandler(null)
  }, [invalidateAI])

  const handleAsk = useCallback((question) => {
    setPendingQuestion(question)
    setActiveTab('chat')
  }, [])
  const clearPending = useCallback(() => setPendingQuestion(null), [])

  const handleUpload = useCallback(async (file) => {
    setUploading(true)
    setUploadError(null)
    try {
      const data = await api.upload(file)
      setDataset({
        id: data.dataset_id,
        filename: data.filename,
        profile: data.profile,
        preview: { columns: Object.keys(data.preview[0] || {}), rows: data.preview, total_rows: data.profile.overview.n_rows },
      })
      setActiveTab('overview')
    } catch (e) {
      setUploadError(e?.response?.data?.detail || 'Could not process this file. Check the format and try again.')
    } finally {
      setUploading(false)
    }
  }, [])

  const refreshProfile = useCallback(async (id) => {
    const profile = await api.getProfile(id)
    const preview = await api.getPreview(id, 25)
    setDataset((prev) => ({ ...prev, profile, preview }))
  }, [])

  const handleApplyCleaning = useCallback(async (actions) => {
    if (!dataset) return
    setApplying(true)
    try {
      const res = await api.applyCleaning(dataset.id, actions)
      setCleaningLog((prev) => [...prev, ...res.log])
      await refreshProfile(dataset.id)
      // invalidate downstream cached results since data changed
      setStats(null); setCorrelation(null); setMlResult(null); setRecommendations(null); setReportUrl(null)
      invalidateAI()
    } finally {
      setApplying(false)
    }
  }, [dataset, refreshProfile, invalidateAI])

  const handleApplyAll = useCallback(async () => {
    if (!dataset?.profile?.cleaning_suggestions?.length) return
    const actions = dataset.profile.cleaning_suggestions
      .filter((s) => s.action !== 'flag_outliers')
      .map((s) => ({ action: s.action, column: s.column }))
    await handleApplyCleaning(actions)
  }, [dataset, handleApplyCleaning])

  const handleReset = useCallback(async () => {
    if (!dataset) return
    setApplying(true)
    try {
      await api.resetCleaning(dataset.id)
      setCleaningLog([])
      await refreshProfile(dataset.id)
      setStats(null); setCorrelation(null); setMlResult(null); setRecommendations(null); setReportUrl(null)
      invalidateAI()
    } finally {
      setApplying(false)
    }
  }, [dataset, refreshProfile, invalidateAI])

  // lazy-load stats / correlation when their tabs are opened
  useEffect(() => {
    if (!dataset) return
    if (activeTab === 'stats' && !stats) {
      api.getStats(dataset.id).then(setStats).catch(() => {}) // expiry is handled by the client interceptor
    }
    if (activeTab === 'correlation' && !correlation) {
      api.getCorrelation(dataset.id).then(setCorrelation).catch(() => {})
    }
  }, [activeTab, dataset, stats, correlation])

  const handleRunML = useCallback(async (target) => {
    if (!dataset) return
    setMlRunning(true)
    try {
      const result = await api.runML(dataset.id, target, null)
      setMlResult(result)
    } catch (e) {
      setMlResult({ error: e?.response?.data?.detail || 'Model training failed on this dataset.' })
    } finally {
      setMlRunning(false)
    }
  }, [dataset])

  useEffect(() => {
    if (!dataset) return
    if (activeTab === 'recommendations' && !recommendations) {
      setRecLoading(true)
      api.getRecommendations(dataset.id, mlResult?.target || null)
        .then((r) => setRecommendations(r.recommendations))
        .finally(() => setRecLoading(false))
    }
  }, [activeTab, dataset, recommendations, mlResult])

  const handleGenerateReport = useCallback(async (target, includeML) => {
    if (!dataset) return
    setReportGenerating(true)
    try {
      await api.createReport(dataset.id, target, includeML)
      setReportUrl(api.reportDownloadUrl(dataset.id))
    } finally {
      setReportGenerating(false)
    }
  }, [dataset])

  const handleNewDataset = useCallback(async () => {
    if (dataset) {
      try { await api.deleteDataset(dataset.id) } catch {}
    }
    setDataset(null)
    setStats(null); setCorrelation(null); setMlResult(null); setRecommendations(null)
    setCleaningLog([]); setReportUrl(null); setActiveTab('overview')
    invalidateAI()
  }, [dataset, invalidateAI])

  if (!dataset) {
    return <UploadScreen onUpload={handleUpload} loading={uploading} error={uploadError} />
  }

  const columnNames = dataset.profile?.overview?.column_names || []

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      <Sidebar
        active={activeTab}
        onSelect={setActiveTab}
        filename={dataset.filename}
        rowCount={dataset.profile?.overview?.n_rows}
        colCount={dataset.profile?.overview?.n_columns}
        onNewDataset={handleNewDataset}
      />
      <main style={{ flex: 1, padding: '32px 40px', maxWidth: 1120 }}>
        {activeTab === 'overview' && <OverviewTab profile={dataset.profile} preview={dataset.preview} />}
        {activeTab === 'insights' && (
          <InsightsTab
            datasetId={dataset.id} llm={llm} columns={dataset.profile?.columns || []}
            briefing={briefing} setBriefing={setBriefing}
            rootCause={rootCause} setRootCause={setRootCause} onAsk={handleAsk}
            accessCode={accessCode} onAccessCode={handleAccessCode} onAccessDenied={handleAccessDenied} accessDenied={accessDenied}
          />
        )}
        {activeTab === 'chat' && (
          <ChatTab
            datasetId={dataset.id} llm={llm}
            messages={chatMessages} setMessages={setChatMessages}
            pending={pendingQuestion} onPendingHandled={clearPending}
            accessCode={accessCode} onAccessCode={handleAccessCode} onAccessDenied={handleAccessDenied} accessDenied={accessDenied}
          />
        )}
        {activeTab === 'cleaning' && (
          <CleaningTab
            profile={dataset.profile}
            onApply={handleApplyCleaning}
            onApplyAll={handleApplyAll}
            onReset={handleReset}
            applying={applying}
            log={cleaningLog}
          />
        )}
        {activeTab === 'stats' && (
          stats ? <StatsTab stats={stats} /> : <LoadingBlock label="Computing statistics…" />
        )}
        {activeTab === 'charts' && <ChartsTab datasetId={dataset.id} />}
        {activeTab === 'correlation' && (
          correlation ? <CorrelationTab correlation={correlation} /> : <LoadingBlock label="Computing correlations…" />
        )}
        {activeTab === 'ml' && (
          <MLTab columns={columnNames} onRun={handleRunML} result={mlResult} running={mlRunning} />
        )}
        {activeTab === 'recommendations' && (
          <RecommendationsTab recommendations={recommendations} loading={recLoading} />
        )}
        {activeTab === 'report' && (
          <ReportTab columns={columnNames} onGenerate={handleGenerateReport} downloadUrl={reportUrl} generating={reportGenerating} />
        )}
      </main>
    </div>
  )
}

function LoadingBlock({ label }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: 80, gap: 12 }}>
      <Spinner size={22} />
      <div style={{ color: 'var(--ink-faint)', fontSize: 13 }}>{label}</div>
    </div>
  )
}
