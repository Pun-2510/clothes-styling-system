import { useEffect, useState } from 'react'
import { useLanguage } from '../../i18n'
import { isFavorite, subscribeFavorites, toggleFavorite } from '../../services/favorites'
import Icon from './Icon'

export default function FavoriteButton({ product, variant = 'icon', className = '' }) {
  const { t } = useLanguage()
  const [active, setActive] = useState(() => isFavorite(product?.product_id))

  useEffect(() => subscribeFavorites(() => {
    setActive(isFavorite(product?.product_id))
  }), [product?.product_id])

  function handleToggle(event) {
    event.preventDefault()
    event.stopPropagation()
    setActive(toggleFavorite(product))
  }

  const label = active ? t('favorites.remove') : t('favorites.add')
  return (
    <button
      className={`favorite-button favorite-button--${variant}${active ? ' active' : ''}${className ? ` ${className}` : ''}`}
      type="button"
      aria-label={label}
      aria-pressed={active}
      title={label}
      onClick={handleToggle}
    >
      <Icon name="heart" size={variant === 'detail' ? 19 : 18} />
      {variant === 'detail' && <span>{label}</span>}
    </button>
  )
}
