import { trackedFetch } from './networkActivity'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''
const AUTOMATIC_CATEGORY_MODE = 'soft_category'

function getErrorMessage(error) {
  if (error instanceof TypeError) {
    return 'Không thể kết nối tới máy chủ. Hãy kiểm tra backend đang chạy.'
  }
  return error.message || 'Đã có lỗi xảy ra. Vui lòng thử lại.'
}

async function readResponse(response) {
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = data?.detail
    throw new Error(detail?.message || detail?.reason || `Yêu cầu thất bại (${response.status})`)
  }
  return data
}

export async function checkServerReady() {
  try {
    const response = await trackedFetch(`${API_BASE}/api/health/ready`)
    return response.ok
  } catch {
    return false
  }
}

export async function searchByText({ query, language, topK }) {
  try {
    const response = await trackedFetch(`${API_BASE}/api/recommendations/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query.trim(),
        language,
        top_k: topK,
      }),
    })
    return await readResponse(response)
  } catch (error) {
    throw new Error(getErrorMessage(error), { cause: error })
  }
}

export async function searchByImage({ file, topK }) {
  const payload = new FormData()
  payload.append('file', file)
  payload.append('top_k', String(topK))
  payload.append('category_mode', AUTOMATIC_CATEGORY_MODE)

  try {
    const response = await trackedFetch(`${API_BASE}/api/recommendations/image`, {
      method: 'POST',
      body: payload,
    })
    return await readResponse(response)
  } catch (error) {
    throw new Error(getErrorMessage(error), { cause: error })
  }
}

export async function searchByCombined({ file, query, language, topK }) {
  const payload = new FormData()
  payload.append('file', file)
  payload.append('query', query.trim())
  payload.append('language', language)
  payload.append('top_k', String(topK))

  try {
    const response = await trackedFetch(`${API_BASE}/api/recommendations/combined`, {
      method: 'POST',
      body: payload,
    })
    return await readResponse(response)
  } catch (error) {
    throw new Error(getErrorMessage(error), { cause: error })
  }
}

export function getCatalogImageUrl(reference) {
  return reference ? `${API_BASE}/catalog-images/${encodeURIComponent(reference)}` : ''
}
