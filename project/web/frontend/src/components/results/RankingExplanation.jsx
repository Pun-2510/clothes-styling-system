import { useState } from 'react'
import { useLanguage } from '../../i18n'
import ProductImage from './ProductImage'
import CategoryRelationGraph from './CategoryRelationGraph'

function ScoreLine({ label, value, weight, accent = false }) {
  if (value == null || Number.isNaN(Number(value))) return null
  return (
    <div className={`explanation-score${accent ? ' explanation-score--final' : ''}`}>
      <span>{label}{weight != null && <small> × {Number(weight).toFixed(2)}</small>}</span>
      <strong>{Number(value).toFixed(3)}</strong>
    </div>
  )
}

export default function RankingExplanation({ data, mode }) {
  const { t } = useLanguage()
  const [selectedIndex, setSelectedIndex] = useState(0)
  const items = data?.items || []
  const explanation = data?.explanation

  if (!items.length || !explanation) {
    return <div className="explanation-empty">{t('explanation.unavailable')}</div>
  }

  const selected = items[Math.min(selectedIndex, items.length - 1)]
  const weights = explanation.weights || {}
  const routeKey = `explanation.route.${explanation.category_route || explanation.ranking_policy}`
  const translatedRoute = t(routeKey)
  const routeLabel = translatedRoute === routeKey
    ? explanation.category_route || explanation.ranking_policy
    : translatedRoute
  const reasons = (selected.reason_codes || [])
    .map((code) => t(`explanation.reason.${code}`))
    .join(', ')

  return (
    <div className="ranking-explanation">
      <header className="explanation-summary">
        <div>
          <span className="eyebrow">RANKING TRACE</span>
          <h3>{t('explanation.title')}</h3>
        </div>
        <div className="explanation-chips">
          <span>{t(`explanation.mode.${mode}`)}</span>
          <span>{routeLabel}</span>
          {data.category_confidence != null && (
            <span>{t('explanation.confidence', { value: Math.round(Number(data.category_confidence) * 100) })}</span>
          )}
        </div>
      </header>

      <div className="rank-selector" role="tablist" aria-label={t('explanation.selectProduct')}>
        {items.map((item, index) => (
          <button
            key={`${item.product_id}-${index}`}
            type="button"
            role="tab"
            aria-selected={selectedIndex === index}
            className={selectedIndex === index ? 'active' : ''}
            onClick={() => setSelectedIndex(index)}
          >
            <span className="rank-selector-image">
              <ProductImage reference={item.image_reference} name={item.product_name} />
            </span>
            <span><strong>#{index + 1}</strong><small>{item.category || t('product.fashion')}</small></span>
          </button>
        ))}
      </div>

      <div className="explanation-layout">
        <article className="score-breakdown-card">
          <div className="explanation-card-heading">
            <span>{t('explanation.scoreBreakdown')}</span>
            <code>{explanation.formula}</code>
          </div>
          <h4>{selected.product_name || t('product.unnamed')}</h4>

          {mode === 'text' && (
            <>
              <ScoreLine label="Text → image" value={selected.text_to_image_score} weight={weights.text_to_image} />
              <ScoreLine label="Text → text" value={selected.text_to_text_score} weight={weights.text_to_text} />
            </>
          )}
          {mode === 'image' && (
            <>
              <ScoreLine label="Image → image" value={selected.visual_similarity} />
              <ScoreLine label={t('explanation.categoryBonus')} value={selected.category_bonus} />
            </>
          )}
          {mode === 'combined' && (
            <>
              <ScoreLine label="Image → image" value={selected.visual_similarity} weight={weights.image} />
              <ScoreLine label={t('explanation.textBranch')} value={selected.text_score} weight={weights.text} />
              <ScoreLine label={t('explanation.attributeMatch')} value={selected.attribute_match} />
              <ScoreLine label={t('explanation.categoryEvidence')} value={selected.category_match} />
            </>
          )}
          <ScoreLine label={t('explanation.finalScore')} value={selected.final_score} accent />

          <p className="explanation-callout">
            {t('explanation.reasonSummary', {
              rank: selectedIndex + 1,
              reasons: reasons || t('explanation.similarityReason'),
            })}
          </p>
          <p className="score-disclaimer">{t('explanation.scoreDisclaimer')}</p>
        </article>

        <article className="graph-card">
          <div className="explanation-card-heading">
            <span>{t('explanation.graphTitle')}</span>
            <small>{t('explanation.graphSubtitle')}</small>
          </div>
          <CategoryRelationGraph
            graph={explanation.graph}
            selectedCategory={selected.category}
            t={t}
          />
        </article>
      </div>

      {(explanation.category_candidates || []).length > 0 && (
        <div className="category-candidates">
          <div><span className="eyebrow">CATEGORY EVIDENCE</span><strong>{t('explanation.topCategories')}</strong></div>
          <div className="candidate-list">
            {explanation.category_candidates.slice(0, 5).map((candidate, index) => (
              <span key={`${candidate.category}-${index}`}>
                <b>#{index + 1}</b> {candidate.category}
                {candidate.score != null && <small>{Number(candidate.score).toFixed(3)}</small>}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
