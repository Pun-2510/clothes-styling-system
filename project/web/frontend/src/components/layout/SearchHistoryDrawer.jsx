import { useEffect } from 'react'
import { createPortal } from 'react-dom'
import Icon from '../common/Icon'
import ProductImage from '../results/ProductImage'

const MODE_LABELS = {
  image: 'Tìm bằng ảnh',
  text: 'Tìm bằng mô tả',
  combined: 'Ảnh + mô tả',
}

function formatDate(value) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return new Intl.DateTimeFormat('vi-VN', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

function historyTitle(entry) {
  if (entry.query) return entry.query
  return entry.image?.name || 'Truy vấn bằng hình ảnh'
}

export default function SearchHistoryDrawer({ open, entries, onClose, onDelete, onClear }) {
  useEffect(() => {
    if (!open) return undefined

    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', closeOnEscape)

    return () => {
      document.body.style.overflow = previousOverflow
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [open, onClose])

  if (!open) return null

  return createPortal(
    <div className="history-layer" role="presentation">
      <button className="history-backdrop" type="button" onClick={onClose} aria-label="Đóng lịch sử tìm kiếm" />
      <aside className="history-drawer" aria-label="Lịch sử tìm kiếm" aria-modal="true" role="dialog">
        <div className="history-drawer-header">
          <div>
            <span className="eyebrow">TÌM KIẾM GẦN ĐÂY</span>
            <h2>Lịch sử của bạn</h2>
          </div>
          <button className="icon-button" type="button" onClick={onClose} aria-label="Đóng">
            <Icon name="close" size={19} />
          </button>
        </div>

        {entries.length === 0 ? (
          <div className="history-empty">
            <span><Icon name="history" size={24} /></span>
            <h3>Chưa có lượt tìm kiếm</h3>
            <p>Các truy vấn gần đây sẽ được lưu cục bộ trên trình duyệt này.</p>
          </div>
        ) : (
          <div className="history-list">
            {entries.map((entry) => {
              const preview = entry.preview?.[0]
              const canRestore = !entry.image || entry.image.stored
              return (
                <article className="history-item" key={entry.id}>
                  <div className="history-item-image">
                    {preview ? (
                      <ProductImage reference={preview.image_reference} name={preview.product_name} />
                    ) : (
                      <Icon name={entry.mode === 'text' ? 'search' : 'image'} size={22} />
                    )}
                  </div>
                  <div className="history-item-copy">
                    <div className="history-item-meta">
                      <span>{MODE_LABELS[entry.mode]}</span>
                      <time dateTime={entry.createdAt}>{formatDate(entry.createdAt)}</time>
                    </div>
                    <h3>{historyTitle(entry)}</h3>
                    <p>
                      {preview
                        ? `${preview.product_name || 'Sản phẩm'} · ${entry.topK} kết quả`
                        : `${entry.topK} kết quả`}
                    </p>
                    {!canRestore && <small>Ảnh gốc không còn trong bộ nhớ trình duyệt.</small>}
                  </div>
                  <div className="history-item-actions">
                    {canRestore ? (
                      <a href={`/?history=${encodeURIComponent(entry.id)}#search`} onClick={onClose}>
                        Dùng lại <Icon name="arrow" size={15} />
                      </a>
                    ) : (
                      <span className="history-restore-disabled">Không thể dùng lại</span>
                    )}
                    <button type="button" onClick={() => onDelete(entry.id)} aria-label={`Xóa ${historyTitle(entry)}`}>
                      <Icon name="trash" size={15} />
                    </button>
                  </div>
                </article>
              )
            })}
          </div>
        )}

        <div className="history-drawer-footer">
          <p><Icon name="lock" size={14} /> Chỉ được lưu trên thiết bị này.</p>
          {entries.length > 0 && (
            <button type="button" onClick={onClear}>Xóa toàn bộ lịch sử</button>
          )}
        </div>
      </aside>
    </div>,
    document.body,
  )
}
