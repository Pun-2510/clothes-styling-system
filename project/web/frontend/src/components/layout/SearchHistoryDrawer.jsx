import { useEffect } from 'react'
import { createPortal } from 'react-dom'
import Icon from '../common/Icon'
import ProductImage from '../results/ProductImage'
import { useLanguage } from '../../i18n'

function formatDate(value, language) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return new Intl.DateTimeFormat(language === 'en' ? 'en-US' : 'vi-VN', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

function historyTitle(entry, t) {
  if (entry.query) return entry.query
  return entry.image?.name || t('history.imageQuery')
}

export default function SearchHistoryDrawer({ open, entries, onClose, onDelete, onClear }) {
  const { language, t } = useLanguage()
  const modeLabels = {
    image: t('history.imageMode'),
    text: t('history.textMode'),
    combined: t('history.combinedMode'),
  }
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
      <button className="history-backdrop" type="button" onClick={onClose} aria-label={t('history.close')} />
      <aside className="history-drawer" aria-label={t('header.history')} aria-modal="true" role="dialog">
        <div className="history-drawer-header">
          <div>
            <span className="eyebrow">{t('history.recent')}</span>
            <h2>{t('history.title')}</h2>
          </div>
          <button className="icon-button" type="button" onClick={onClose} aria-label={t('history.close')}>
            <Icon name="close" size={19} />
          </button>
        </div>

        {entries.length === 0 ? (
          <div className="history-empty">
            <span><Icon name="history" size={24} /></span>
            <h3>{t('history.emptyTitle')}</h3>
            <p>{t('history.emptyText')}</p>
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
                      <span>{modeLabels[entry.mode]}</span>
                      <time dateTime={entry.createdAt}>{formatDate(entry.createdAt, language)}</time>
                    </div>
                    <h3>{historyTitle(entry, t)}</h3>
                    <p>
                      {preview
                        ? t('history.resultSummary', { product: preview.product_name || t('product.unnamed'), count: entry.topK })
                        : t('history.resultsOnly', { count: entry.topK })}
                    </p>
                    {!canRestore && <small>{t('history.missingImage')}</small>}
                  </div>
                  <div className="history-item-actions">
                    {canRestore ? (
                      <a href={`/?history=${encodeURIComponent(entry.id)}#search`} onClick={onClose}>
                        {t('history.reuse')} <Icon name="arrow" size={15} />
                      </a>
                    ) : (
                      <span className="history-restore-disabled">{t('history.cannotReuse')}</span>
                    )}
                    <button type="button" onClick={() => onDelete(entry.id)} aria-label={t('history.delete', { title: historyTitle(entry, t) })}>
                      <Icon name="trash" size={15} />
                    </button>
                  </div>
                </article>
              )
            })}
          </div>
        )}

        <div className="history-drawer-footer">
          <p><Icon name="lock" size={14} /> {t('history.localOnly')}</p>
          {entries.length > 0 && (
            <button type="button" onClick={onClear}>{t('history.clear')}</button>
          )}
        </div>
      </aside>
    </div>,
    document.body,
  )
}
