import Icon from '../common/Icon'
import ProductCard from './ProductCard'
import { useLanguage } from '../../i18n'

export default function ResultsSection({ data, mode, sectionRef, onNewSearch }) {
  const { t } = useLanguage()
  if (!data) return null

  const items = data.items || []
  const elapsed = mode === 'image' ? data.processing_times?.total : data.elapsed_seconds

  return (
    <section ref={sectionRef} className="results" aria-live="polite">
      <div className="results-header">
        <div>
          <span className="eyebrow">{t('results.eyebrow')}</span>
          <h2>{t('results.count', { count: items.length })}</h2>
        </div>
        <div className="result-actions">
          <div className="result-meta">
            {mode === 'text' && data.query_used !== data.query && data.query_used && (
              <span>{t('results.query', { query: data.query_used })}</span>
            )}
            {mode === 'combined' && (
              <span>
                {t('results.weights', { image: Math.round((data.image_weight || 0) * 100), text: Math.round((data.text_weight || 0) * 100) })}
              </span>
            )}
            {mode === 'combined' && data.query_used !== data.query && data.query_used && (
              <span>{t('results.translated', { query: data.query_used })}</span>
            )}
            {mode === 'image' && data.category_mode === 'soft_category' && data.predicted_category && (
              <span>{t('results.categoryPriority', { category: data.predicted_category, confidence: Math.round((data.category_confidence || 0) * 100) })}</span>
            )}
            {mode === 'image' && data.category_mode === 'no_category' && (
              <span>{t('results.noCategory')}</span>
            )}
            <span>{t('results.seconds', { seconds: Number(elapsed || 0).toFixed(2) })}</span>
          </div>
          {onNewSearch && (
            <button className="new-search-button" type="button" onClick={onNewSearch}>
              <Icon name="search" size={15} /> {t('results.newSearch')}
            </button>
          )}
        </div>
      </div>

      {items.length > 0 ? (
        <div className="product-grid">
          {items.map((item, index) => (
            <ProductCard key={`${item.product_id}-${index}`} item={item} index={index} mode={mode} />
          ))}
        </div>
      ) : (
        <div className="empty-result">{t('results.empty')}</div>
      )}
    </section>
  )
}
