import { useEffect, useState } from 'react'
import ProductCard from '../components/results/ProductCard'
import ProductImage from '../components/results/ProductImage'
import { ProductDetailSkeleton } from '../components/results/CatalogSkeleton'
import { categoryUrl, getProduct } from '../services/catalogApi'
import { useLanguage } from '../i18n'

export default function ProductPage({ productId }) {
  const { t } = useLanguage()
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    getProduct(productId).then((response) => { if (active) setData(response) })
      .catch((failure) => { if (active) setError(failure.message) })
    return () => { active = false }
  }, [productId])

  if (error) return <div className="state-message" role="alert">{error} <a href="/categories">{t('detail.viewCategories')}</a></div>
  if (!data) return <ProductDetailSkeleton />

  const { product, related } = data
  return (
    <>
      <nav className="breadcrumbs">
        <a href="/">{t('categories.home')}</a><span>/</span><a href={categoryUrl(product.category)}>{product.category}</a><span>/</span><span>{product.product_name}</span>
      </nav>
      <section className="product-detail">
        <div className="detail-visual"><ProductImage reference={product.image_reference} name={product.product_name} /></div>
        <div className="detail-copy">
          <span className="eyebrow">{product.category || t('detail.product')}</span>
          <h1>{product.product_name}</h1>
          <span className="detail-id">{t('detail.id', { id: product.product_id })}</span>
          <div className="detail-divider" />
          <h2>{t('detail.info')}</h2>
          <p>{product.description || t('detail.noDescription')}</p>
          <a className="detail-category-link" href={categoryUrl(product.category)}>{t('detail.moreCategory', { category: product.category })} →</a>
        </div>
      </section>
      {related.length > 0 && (
        <section className="catalog-section related-section">
          <div className="section-heading"><div><span className="eyebrow">{t('detail.related')}</span><h2>{t('detail.sameCategory')}</h2></div><a href={categoryUrl(product.category)}>{t('catalog.seeAll')} →</a></div>
          <div className="product-grid product-grid--five">{related.map((item) => <ProductCard key={item.product_id} item={item} />)}</div>
        </section>
      )}
    </>
  )
}
