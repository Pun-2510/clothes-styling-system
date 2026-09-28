import { useState } from 'react'
import Icon from '../common/Icon'
import { getCatalogImageUrl } from '../../services/recommendationApi'

export default function ProductImage({ reference, name }) {
  const [failed, setFailed] = useState(false)
  const source = getCatalogImageUrl(reference)

  if (!source || failed) {
    return (
      <div className="product-image product-image--fallback" aria-label="Không có ảnh sản phẩm">
        <Icon name="image" size={28} />
        <span>{reference || 'Không có ảnh'}</span>
      </div>
    )
  }

  return (
    <div className="product-image">
      <img src={source} alt={name || 'Sản phẩm thời trang'} loading="lazy" onError={() => setFailed(true)} />
    </div>
  )
}
