const STORAGE_KEY = 'thread-find.favorites.v1'
const CHANGE_EVENT = 'thread-find:favorites-changed'

function normalizeProduct(product) {
  return {
    product_id: String(product.product_id),
    product_name: product.product_name || null,
    category: product.category || null,
    image_reference: product.image_reference || null,
    description: product.description || null,
    saved_at: new Date().toISOString(),
  }
}

function saveFavorites(items) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(items))
  } catch {
    return false
  }
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: items }))
  return true
}

export function loadFavorites() {
  try {
    const stored = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '[]')
    return Array.isArray(stored)
      ? stored.filter((item) => item?.product_id)
      : []
  } catch {
    return []
  }
}

export function isFavorite(productId) {
  const id = String(productId)
  return loadFavorites().some((item) => String(item.product_id) === id)
}

export function toggleFavorite(product) {
  if (!product?.product_id) return false
  const id = String(product.product_id)
  const current = loadFavorites()
  const existing = current.some((item) => String(item.product_id) === id)
  const next = existing
    ? current.filter((item) => String(item.product_id) !== id)
    : [normalizeProduct(product), ...current]
  saveFavorites(next)
  return !existing
}

export function subscribeFavorites(callback) {
  const onCustomChange = (event) => callback(event.detail || loadFavorites())
  const onStorageChange = (event) => {
    if (event.key === STORAGE_KEY) callback(loadFavorites())
  }
  window.addEventListener(CHANGE_EVENT, onCustomChange)
  window.addEventListener('storage', onStorageChange)
  return () => {
    window.removeEventListener(CHANGE_EVENT, onCustomChange)
    window.removeEventListener('storage', onStorageChange)
  }
}
