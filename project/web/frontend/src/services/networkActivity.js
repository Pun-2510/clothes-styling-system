let pendingRequests = 0
const listeners = new Set()

function notify() {
  listeners.forEach((listener) => listener(pendingRequests > 0))
}

export function subscribeToNetworkActivity(listener) {
  listeners.add(listener)
  listener(pendingRequests > 0)
  return () => listeners.delete(listener)
}

export async function trackedFetch(...args) {
  pendingRequests += 1
  notify()
  try {
    return await fetch(...args)
  } finally {
    pendingRequests = Math.max(0, pendingRequests - 1)
    notify()
  }
}
