import { trackedFetch } from './networkActivity'
import { getInterfaceLanguage, translateMessage } from '../i18n'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''
const requestCache = new Map()

function getJson(path) {
  if (requestCache.has(path)) return requestCache.get(path)

  const request = trackedFetch(`${API_BASE}${path}`).then(async (response) => {
    if (!response.ok) {
      const language = getInterfaceLanguage()
      throw new Error(response.status === 503
        ? translateMessage(language, 'api.catalogNotPrepared')
        : translateMessage(language, 'api.catalogLoad', { status: response.status }))
    }
    return response.json()
  }).catch((error) => {
    requestCache.delete(path)
    throw error
  })

  requestCache.set(path, request)
  return request
}

export function getCategories() {
  return getJson('/api/catalog/categories')
}

export function getProducts({ category, page = 1, pageSize = 24 } = {}) {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
  if (category) params.set('category', category)
  return getJson(`/api/catalog/products?${params}`)
}

export function getProduct(productId) {
  return getJson(`/api/catalog/products/${encodeURIComponent(productId)}`)
}

export function categoryUrl(name) {
  return `/category/${encodeURIComponent(name)}`
}

export function productUrl(id) {
  return `/product/${encodeURIComponent(id)}`
}

const FEATURED_NAMES = ['Tshirts', 'Shirts', 'Dresses', 'Caps', 'Casual Shoes']

export function featuredCategories(categories, limit = 5) {
  const byName = new Map(categories.map((item) => [item.name, item]))
  const preferred = FEATURED_NAMES.map((name) => byName.get(name)).filter(Boolean)
  const remaining = categories.filter((item) => !FEATURED_NAMES.includes(item.name))
  return [...preferred, ...remaining].slice(0, limit)
}
