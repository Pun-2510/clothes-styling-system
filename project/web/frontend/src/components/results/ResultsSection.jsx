import Icon from '../common/Icon'
import ProductCard from './ProductCard'
import { useLanguage } from '../../i18n'

function QueryGuidance({ guidance, onUseSuggestion, t }) {
  if (!guidance || guidance.status === 'unrestricted') return null

  if (guidance.status === 'category_applied') {
    return (
      <div className="query-guidance query-guidance--applied">
        <Icon name="spark" size={17} />
        <p>{t('results.categoryApplied', { category: guidance.applied_category })}</p>
      </div>
    )
  }

  const isSpelling = guidance.status === 'spelling_suggestion'
  return (
    <div className={`query-guidance query-guidance--${isSpelling ? 'spelling' : 'broad'}`}>
      <Icon name={isSpelling ? 'search' : 'spark'} size={18} />
      <div>
        <strong>{t(isSpelling ? 'results.spellingSuggestion' : 'results.broadQuery')}</strong>
        <div className="query-guidance-options">
          {(guidance.suggestions || []).map((suggestion) => (
            <button
              key={`${suggestion.category}-${suggestion.corrected_query}`}
              type="button"
              onClick={() => onUseSuggestion?.(suggestion.corrected_query)}
            >
              {suggestion.category}
              {suggestion.confidence != null && <small>{Math.round(suggestion.confidence * 100)}%</small>}
            </button>
          ))}
        </div>
        <span>{t('results.suggestionHint')}</span>
      </div>
    </div>
  )
}

export default function ResultsSection({ data, mode, sectionRef, onNewSearch, onUseSuggestion }) {
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

      {mode === 'text' && (
        <QueryGuidance guidance={data.guidance} onUseSuggestion={onUseSuggestion} t={t} />
      )}

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
