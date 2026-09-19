import { useCallback, useEffect, useState } from 'react'
import { checkServerReady } from '../services/recommendationApi'

export default function useServerStatus() {
  const [status, setStatus] = useState('checking')

  const checkStatus = useCallback(async () => {
    setStatus('checking')
    const isReady = await checkServerReady()
    setStatus(isReady ? 'ready' : 'offline')
  }, [])

  useEffect(() => {
    let active = true

    checkServerReady().then((isReady) => {
      if (active) setStatus(isReady ? 'ready' : 'offline')
    })

    return () => {
      active = false
    }
  }, [])

  return { status, checkStatus }
}
