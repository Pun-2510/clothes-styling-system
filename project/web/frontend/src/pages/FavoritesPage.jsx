import { useEffect, useState } from 'react'
import Icon from '../components/common/Icon'
import ProductCard from '../components/results/ProductCard'
import { loadFavorites, subscribeFavorites } from '../services/favorites'
import { useLanguage } from '../i18n'

export default function FavoritesPage() {
  const { t } = useLanguage()
  const [favorites, setFavorites] = useState(loadFavorites)

  useEffect(() => subscribeFavorites(setFavorites), [])

  return (
    <section className="listing-page favorites-page">
      <nav className="breadcrumbs">
        <a href="/">{t('categories.home')}</a><span>/</span><span>{t('favorites.title')}</span>
      </nav>
      <div className="page-heading favorites-heading">
        <span className="eyebrow">{t('favorites.eyebrow')}</span>
        <h1>{t('favorites.title')}</h1>
        <p>{t('favorites.description', { count: favorites.length })}</p>
      </div>

      {favorites.length > 0 ? (
        <div className="product-grid category-product-grid favorites-grid">
          {favorites.map((item) => <ProductCard key={item.product_id} item={item} />)}
        </div>
      ) : (
        <div className="favorites-empty">
          <span><Icon name="heart" size={28} /></span>
          <h2>{t('favorites.emptyTitle')}</h2>
          <p>{t('favorites.emptyDescription')}</p>
          <a href="/">{t('favorites.explore')} <Icon name="arrow" size={16} /></a>
        </div>
      )}
    </section>
  )
}
