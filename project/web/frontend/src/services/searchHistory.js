const HISTORY_KEY = 'thread-find.search-history.v1'
const MAX_HISTORY_ITEMS = 15
const DATABASE_NAME = 'thread-find-history'
const DATABASE_VERSION = 1
const FILE_STORE = 'query-images'

function readHistory() {
  try {
    const value = window.localStorage.getItem(HISTORY_KEY)
    const entries = value ? JSON.parse(value) : []
    return Array.isArray(entries) ? entries : []
  } catch {
    return []
  }
}

function notifyHistoryChanged() {
  window.dispatchEvent(new CustomEvent('thread-find:history-changed'))
}

function writeHistory(entries) {
  try {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(entries))
    notifyHistoryChanged()
    return true
  } catch {
    return false
  }
}

function openHistoryDatabase() {
  return new Promise((resolve, reject) => {
    if (!window.indexedDB) {
      reject(new Error('IndexedDB is unavailable.'))
      return
    }

    const request = window.indexedDB.open(DATABASE_NAME, DATABASE_VERSION)
    request.onupgradeneeded = () => {
      const database = request.result
      if (!database.objectStoreNames.contains(FILE_STORE)) {
        database.createObjectStore(FILE_STORE)
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

async function saveHistoryFile(id, file) {
  if (!file) return false

  try {
    const database = await openHistoryDatabase()
    await new Promise((resolve, reject) => {
      const transaction = database.transaction(FILE_STORE, 'readwrite')
      transaction.objectStore(FILE_STORE).put({
        blob: file.slice(0, file.size, file.type),
        name: file.name,
        type: file.type,
        lastModified: file.lastModified,
      }, id)
      transaction.oncomplete = resolve
      transaction.onerror = () => reject(transaction.error)
      transaction.onabort = () => reject(transaction.error)
    })
    database.close()
    return true
  } catch {
    return false
  }
}

async function removeHistoryFile(id) {
  try {
    const database = await openHistoryDatabase()
    await new Promise((resolve, reject) => {
      const transaction = database.transaction(FILE_STORE, 'readwrite')
      transaction.objectStore(FILE_STORE).delete(id)
      transaction.oncomplete = resolve
      transaction.onerror = () => reject(transaction.error)
      transaction.onabort = () => reject(transaction.error)
    })
    database.close()
  } catch {
    // History remains usable when browser file storage is unavailable.
  }
}

function createHistoryId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID()
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`
}

function requestFingerprint(mode, query, file) {
  const normalizedQuery = String(query || '').trim().toLocaleLowerCase()
  const imageFingerprint = file
    ? `${file.name}:${file.size}:${file.lastModified}`
    : ''
  return `${mode}|${normalizedQuery}|${imageFingerprint}`
}

function previewItems(data) {
  return (data?.items || []).slice(0, 3).map((item) => ({
    product_id: item.product_id,
    product_name: item.product_name,
    category: item.category,
    image_reference: item.image_reference,
  }))
}

export function loadSearchHistory() {
  return readHistory()
}

export function loadSearchHistoryEntry(id) {
  return readHistory().find((entry) => entry.id === id) || null
}

export async function loadSearchHistoryFile(id) {
  try {
    const database = await openHistoryDatabase()
    const record = await new Promise((resolve, reject) => {
      const request = database.transaction(FILE_STORE, 'readonly')
        .objectStore(FILE_STORE)
        .get(id)
      request.onsuccess = () => resolve(request.result || null)
      request.onerror = () => reject(request.error)
    })
    database.close()
    if (!record?.blob) return null

    return new File([record.blob], record.name || 'query-image', {
      type: record.type || record.blob.type,
      lastModified: record.lastModified || Date.now(),
    })
  } catch {
    return null
  }
}

export async function addSearchHistory({ mode, query, language, topK, file, data }) {
  if (!['text', 'image', 'combined'].includes(mode)) return null

  const id = createHistoryId()
  const fingerprint = requestFingerprint(mode, query, file)
  const imageStored = file ? await saveHistoryFile(id, file) : false
  const previous = readHistory()
  const duplicates = previous.filter((entry) => entry.fingerprint === fingerprint)
  const retained = previous.filter((entry) => entry.fingerprint !== fingerprint)

  const entry = {
    id,
    mode,
    query: String(query || '').trim(),
    language: language === 'en' ? 'en' : 'vi',
    topK: Number(topK) || 5,
    createdAt: new Date().toISOString(),
    fingerprint,
    image: file ? {
      name: file.name,
      size: file.size,
      type: file.type,
      stored: imageStored,
    } : null,
    preview: previewItems(data),
    predictedCategory: data?.predicted_category || null,
  }

  const next = [entry, ...retained].slice(0, MAX_HISTORY_ITEMS)
  if (!writeHistory(next)) {
    if (imageStored) await removeHistoryFile(id)
    return null
  }

  const removed = [
    ...duplicates,
    ...retained.slice(MAX_HISTORY_ITEMS - 1),
  ]
  await Promise.all(removed.map((item) => removeHistoryFile(item.id)))
  return entry
}

export async function deleteSearchHistoryEntry(id) {
  const next = readHistory().filter((entry) => entry.id !== id)
  writeHistory(next)
  await removeHistoryFile(id)
}

export async function clearSearchHistory() {
  const entries = readHistory()
  try {
    window.localStorage.removeItem(HISTORY_KEY)
    notifyHistoryChanged()
  } catch {
    // Ignore unavailable local storage.
  }
  await Promise.all(entries.map((entry) => removeHistoryFile(entry.id)))
}
