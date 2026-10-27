const FORM_KEY = 'thread-find.search-form.v1'
const RESULT_KEY = 'thread-find.search-result.v1'
const FILE_MARKER_KEY = 'thread-find.search-file-present.v1'
const DATABASE_NAME = 'thread-find-ui'
const STORE_NAME = 'search-files'
const FILE_KEY = 'current-query-image'

function readSessionValue(key) {
  try {
    const value = window.sessionStorage.getItem(key)
    return value ? JSON.parse(value) : null
  } catch {
    return null
  }
}

function writeSessionValue(key, value) {
  try {
    window.sessionStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Persistence is optional; the search must still work without storage.
  }
}

export function loadSearchFormState() {
  return readSessionValue(FORM_KEY) || {}
}

export function saveSearchFormState(state) {
  writeSessionValue(FORM_KEY, state)
}

export function clearSearchQueryState() {
  const current = loadSearchFormState()
  writeSessionValue(FORM_KEY, { ...current, query: '' })
}

export function loadSearchResult() {
  const saved = readSessionValue(RESULT_KEY)
  if (!saved?.data || !['text', 'image', 'combined'].includes(saved.mode)) return null
  return saved
}

export function saveSearchResult(data, mode) {
  writeSessionValue(RESULT_KEY, { data, mode })
}

export function clearSearchResult() {
  try {
    window.sessionStorage.removeItem(RESULT_KEY)
  } catch {
    // Ignore unavailable browser storage.
  }
}

function openFileDatabase() {
  return new Promise((resolve, reject) => {
    if (!window.indexedDB) {
      reject(new Error('IndexedDB is unavailable.'))
      return
    }

    const request = window.indexedDB.open(DATABASE_NAME, 1)
    request.onupgradeneeded = () => {
      const database = request.result
      if (!database.objectStoreNames.contains(STORE_NAME)) database.createObjectStore(STORE_NAME)
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

export async function saveSearchFile(file) {
  try {
    const database = await openFileDatabase()
    await new Promise((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, 'readwrite')
      transaction.objectStore(STORE_NAME).put({
        blob: file.slice(0, file.size, file.type),
        name: file.name,
        type: file.type,
        lastModified: file.lastModified,
      }, FILE_KEY)
      transaction.oncomplete = resolve
      transaction.onerror = () => reject(transaction.error)
      transaction.onabort = () => reject(transaction.error)
    })
    database.close()
    window.sessionStorage.setItem(FILE_MARKER_KEY, '1')
    return true
  } catch {
    return false
  }
}

export async function loadSearchFile() {
  try {
    if (window.sessionStorage.getItem(FILE_MARKER_KEY) !== '1') return null
    const database = await openFileDatabase()
    const record = await new Promise((resolve, reject) => {
      const request = database.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(FILE_KEY)
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

export async function clearSearchFile() {
  try {
    window.sessionStorage.removeItem(FILE_MARKER_KEY)
    const database = await openFileDatabase()
    await new Promise((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, 'readwrite')
      transaction.objectStore(STORE_NAME).delete(FILE_KEY)
      transaction.oncomplete = resolve
      transaction.onerror = () => reject(transaction.error)
      transaction.onabort = () => reject(transaction.error)
    })
    database.close()
  } catch {
    // Ignore cleanup errors; they must not block the interface.
  }
}
