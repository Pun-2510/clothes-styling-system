import Icon from '../common/Icon'

const STATUS_LABELS = {
  ready: 'AI sẵn sàng',
  checking: 'Đang kết nối',
  offline: 'Backend ngoại tuyến',
}

export default function Header({ serverStatus, onRetry }) {
  return (
    <header className="site-header">
      <a className="brand" href="#top" aria-label="Thread Find trang chủ">
        <span className="brand-mark">TF</span>
        <span>THREAD/FIND</span>
      </a>
      <div className={`status status--${serverStatus}`}>
        <span className="status-dot" />
        {STATUS_LABELS[serverStatus]}
        {serverStatus === 'offline' && (
          <button type="button" onClick={onRetry} aria-label="Thử kết nối lại">
            <Icon name="retry" size={15} />
          </button>
        )}
      </div>
    </header>
  )
}
