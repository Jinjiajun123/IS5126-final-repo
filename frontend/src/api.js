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
