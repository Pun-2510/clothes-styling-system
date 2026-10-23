import { useEffect, useState } from 'react'
import { subscribeToNetworkActivity } from '../../services/networkActivity'

export default function LoadingBar() {
  const [active, setActive] = useState(false)

  useEffect(() => subscribeToNetworkActivity(setActive), [])

  return (
    <div
      className={`global-loading-bar ${active ? 'is-active' : ''}`}
      role="progressbar"
      aria-label="Đang tải dữ liệu"
      aria-hidden={!active}
    >
      <span />
    </div>
  )
}
