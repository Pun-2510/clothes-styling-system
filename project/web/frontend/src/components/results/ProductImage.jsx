import { useState } from 'react'
import Icon from '../common/Icon'
import { getCatalogImageUrl } from '../../services/recommendationApi'
import { useLanguage } from '../../i18n'

export default function ProductImage({ reference, name }) {
  const { t } = useLanguage()
  const [failed, setFailed] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const source = getCatalogImageUrl(reference)

  if (!source || failed) {
    return (
      <div className="product-image product-image--fallback" aria-label={t('image.unavailableLabel')}>
        <Icon name="image" size={28} />
        <span>{reference || t('image.unavailable')}</span>
      </div>
    )
  }

  return (
    <div className={`product-image ${loaded ? 'product-image--loaded' : 'product-image--loading'}`} aria-busy={!loaded}>
      {!loaded && <span className="image-skeleton skeleton-shimmer" aria-hidden="true" />}
      <img
        src={source}
        alt={name || t('image.productAlt')}
        loading="lazy"
        onLoad={() => setLoaded(true)}
        onError={() => setFailed(true)}
      />
    </div>
  )
}
