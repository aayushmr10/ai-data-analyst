import axios from 'axios'

// Same-origin by default (single container). For a split deployment (e.g. Vercel + a separate
// API host) set VITE_API_URL=https://your-api.example.com at build time.
const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

const client = axios.create({
  baseURL: `${API_BASE}/api`,
  timeout: 60000,
})

// ---- access code for the AI endpoints (only needed if the server sets APP_ACCESS_CODE) ----
const CODE_KEY = 'ada_access_code'
export const getAccessCode = () => { try { return sessionStorage.getItem(CODE_KEY) || '' } catch { return '' } }
export const storeAccessCode = (code) => {
  try { code ? sessionStorage.setItem(CODE_KEY, code) : sessionStorage.removeItem(CODE_KEY) } catch { /* private mode */ }
}

// ---- a free host may restart or evict idle sessions: tell the app so it can recover gracefully ----
let onSessionExpired = null
export const setSessionExpiredHandler = (fn) => { onSessionExpired = fn }
const watchExpiry = (err) => {
  const detail = err?.response?.data?.detail
  if (err?.response?.status === 404 && typeof detail === 'string' && detail.startsWith('Dataset not found') && onSessionExpired) {
    onSessionExpired()
  }
  return Promise.reject(err)
}
client.interceptors.response.use((r) => r, watchExpiry)

// LLM calls run multi-step tool loops, so they get a much longer timeout.
const llmClient = axios.create({ baseURL: `${API_BASE}/api`, timeout: 180000 })
llmClient.interceptors.request.use((cfg) => {
  const code = getAccessCode()
  if (code) cfg.headers['X-Access-Code'] = code
  return cfg
})
llmClient.interceptors.response.use((r) => r, watchExpiry)

export const api = {
  upload: async (file) => {
    const form = new FormData()
    form.append('file', file)
    const { data } = await client.post('/datasets/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return data
  },
  getProfile: async (id) => (await client.get(`/datasets/${id}/profile`)).data,
  getPreview: async (id, rows = 25) => (await client.get(`/datasets/${id}/preview`, { params: { rows } })).data,
  applyCleaning: async (id, actions) => (await client.post(`/datasets/${id}/clean`, { actions })).data,
  resetCleaning: async (id) => (await client.post(`/datasets/${id}/reset`)).data,
  getStats: async (id) => (await client.get(`/datasets/${id}/stats`)).data,
  getCorrelation: async (id) => (await client.get(`/datasets/${id}/correlation`)).data,
  getChartSuggestions: async (id) => (await client.get(`/datasets/${id}/chart-suggestions`)).data,
  chartHistogram: async (id, column) => (await client.post(`/datasets/${id}/charts/histogram`, { column })).data,
  chartBoxplot: async (id, column) => (await client.post(`/datasets/${id}/charts/boxplot`, { column })).data,
  chartBar: async (id, column) => (await client.post(`/datasets/${id}/charts/bar`, { column })).data,
  chartPie: async (id, column) => (await client.post(`/datasets/${id}/charts/pie`, { column })).data,
  chartScatter: async (id, x, y) => (await client.post(`/datasets/${id}/charts/scatter`, { x, y })).data,
  chartTimeseries: async (id, date_column, value_column) =>
    (await client.post(`/datasets/${id}/charts/timeseries`, { date_column, value_column })).data,
  runML: async (id, target, n_clusters) => (await client.post(`/datasets/${id}/ml/run`, { target, n_clusters })).data,
  getRecommendations: async (id, target) =>
    (await client.get(`/datasets/${id}/recommendations`, { params: { target } })).data,
  createReport: async (id, target, include_ml = true) =>
    (await client.post(`/datasets/${id}/report`, { target, include_ml })).data,
  reportDownloadUrl: (id) => `${API_BASE}/api/datasets/${id}/report/download`,
  deleteDataset: async (id) => (await client.delete(`/datasets/${id}`)).data,

  // ---- AI analyst ----
  llmStatus: async () => (await client.get('/llm/status')).data,
  chatSuggestions: async (id) => (await client.get(`/datasets/${id}/chat/suggestions`)).data,
  chat: async (id, message, history) =>
    (await llmClient.post(`/datasets/${id}/chat`, { message, history })).data,
  briefing: async (id, force = false) =>
    (await llmClient.post(`/datasets/${id}/briefing`, { force })).data,
  rootCause: async (id, payload) =>
    (await llmClient.post(`/datasets/${id}/root-cause`, payload)).data,
}

export const errorMessage = (e, fallback = 'Something went wrong. Please try again.') =>
  e?.response?.data?.detail || (e?.code === 'ECONNABORTED' ? 'The analysis timed out. Try a narrower question.' : fallback)

export default api
