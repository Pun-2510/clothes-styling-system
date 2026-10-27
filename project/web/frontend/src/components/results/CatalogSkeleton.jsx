function SkeletonCard() {
  return (
    <article className="skeleton-card" aria-hidden="true">
      <div className="skeleton-card-visual skeleton-shimmer" />
      <div className="skeleton-card-copy">
        <span className="skeleton-line skeleton-line--short skeleton-shimmer" />
        <span className="skeleton-line skeleton-shimmer" />
        <span className="skeleton-line skeleton-line--medium skeleton-shimmer" />
      </div>
    </article>
  )
}

export function ProductGridSkeleton({ count = 5, className = '' }) {
  return (
    <div className={`product-grid skeleton-grid ${className}`.trim()} aria-hidden="true">
      {Array.from({ length: count }, (_, index) => <SkeletonCard key={index} />)}
    </div>
  )
}

export function ProductDetailSkeleton() {
  const { t } = useLanguage()

  return (
    <section className="product-detail product-detail-skeleton" aria-label={t('loading.productDetail')} aria-busy="true">
      <div className="detail-visual skeleton-shimmer" />
      <div className="detail-copy">
        <span className="skeleton-line skeleton-line--short skeleton-shimmer" />
        <span className="skeleton-heading skeleton-shimmer" />
        <span className="skeleton-line skeleton-line--medium skeleton-shimmer" />
        <div className="detail-divider" />
        <span className="skeleton-line skeleton-line--short skeleton-shimmer" />
        <span className="skeleton-paragraph skeleton-shimmer" />
      </div>
    </section>
  )
}

export default function CatalogSkeleton() {
  const { t } = useLanguage()

  return (
    <section className="catalog-section catalog-loading-section" aria-label={t('loading.catalog')} aria-busy="true">
      <div className="skeleton-section-heading">
        <span className="skeleton-line skeleton-line--short skeleton-shimmer" />
        <span className="skeleton-heading skeleton-shimmer" />
      </div>
      <ProductGridSkeleton count={5} className="product-grid--five" />
    </section>
  )
}
import { useLanguage } from '../../i18n'

