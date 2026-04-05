import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000/api'
})

export const fetchConfig = async () => {
  const res = await api.get('/config')
  return res.data
}

export const fetchClusters = async () => {
  const res = await api.get('/clusters')
  return res.data
}

export const fetchProducts = async (params) => {
  const res = await api.get('/products', { params })
  return res.data
}

export const analyzeSession = async (data) => {
  const res = await api.post('/session_analysis', data)
  return res.data
}

export const compareInterventions = async (data) => {
  const res = await api.post('/intervention_compare', data)
  return res.data
}
