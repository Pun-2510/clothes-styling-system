const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

async function readJson(response) {
  const data = await response.json().catch(() => null)
  if (!response.ok) throw new Error(data?.detail?.message || `HTTP ${response.status}`)
  return data
}

export async function getDeveloperSnapshot() {
  const cacheKey = Date.now()
  const [loadBalancerResult, backendResult] = await Promise.allSettled([
    fetch(`${API_BASE}/lb-health?check=${cacheKey}`, { cache: 'no-store' }),
    fetch(`${API_BASE}/api/health/system?check=${cacheKey}`, { cache: 'no-store' }),
  ])

  let loadBalancer = { ready: false, message: null }
  if (loadBalancerResult.status === 'fulfilled') {
    loadBalancer = {
      ready: loadBalancerResult.value.ok,
      message: await loadBalancerResult.value.text().catch(() => null),
    }
  }

  let backend = null
  let upstream = null
  let backendError = null
  if (backendResult.status === 'fulfilled') {
    upstream = backendResult.value.headers.get('x-upstream-addr')
    try {
      backend = await readJson(backendResult.value)
    } catch (error) {
      backendError = error.message
    }
  } else {
    backendError = backendResult.reason?.message || 'Network error'
  }

  return {
    loadBalancer,
    backend,
    upstream,
    backendError,
    checkedAt: new Date().toISOString(),
  }
}
