import { useEffect, useState } from 'react'
import ProductCard from '../components/results/ProductCard'
import { ProductGridSkeleton } from '../components/results/CatalogSkeleton'
import { getProducts } from '../services/catalogApi'
import { useLanguage } from '../i18n'

export default function CategoryPage({ name }) {
  const { language, t } = useLanguage()
  const searchParams = new URLSearchParams(window.location.search)
  const requestedPage = Number(searchParams.get('page'))
  const page = Number.isInteger(requestedPage) && requestedPage > 0 ? requestedPage : 1
  const audience = searchParams.get('audience') || ''
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    getProducts({ category: name, audience, page, pageSize: 24 })
      .then((response) => { if (active) setData(response) })
      .catch((failure) => { if (active) setError(failure.message) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [name, audience, page])

  const pageUrl = (number) => {
    const params = new URLSearchParams({ page: String(number) })
    if (audience) params.set('audience', audience)
    return `/category/${encodeURIComponent(name)}?${params}`
  }
  const audienceUrl = (value) => {
    const params = new URLSearchParams()
    if (value) params.set('audience', value)
    const query = params.toString()
    return `/category/${encodeURIComponent(name)}${query ? `?${query}` : ''}`
  }

  return (
    <section className="listing-page">
      <nav className="breadcrumbs"><a href="/">{t('categories.home')}</a><span>/</span><a href="/categories">{t('header.categories')}</a><span>/</span><span>{name}</span></nav>
      <div className="page-heading">
        <span className="eyebrow">{t('category.eyebrow')}</span>
        <h1>{name}</h1>
        <p>{data ? t('search.products', { count: data.total.toLocaleString(language === 'en' ? 'en-US' : 'vi-VN') }) : t('category.loading')}</p>
      </div>
      <nav className="audience-filter" aria-label={t('audience.filterLabel')}>
        <a className={!audience ? 'active' : ''} href={audienceUrl('')}>{t('audience.all')}</a>
        {['women', 'men', 'kids', 'unisex'].map((value) => (
          <a className={audience === value ? 'active' : ''} href={audienceUrl(value)} key={value}>{t(`audience.${value}`)}</a>
        ))}
      </nav>
      {error && <div className="state-message" role="alert">{error}</div>}
      {loading && <ProductGridSkeleton count={8} className="category-product-grid" />}
      {data && <div className="product-grid category-product-grid">{data.items.map((item) => <ProductCard key={item.product_id} item={item} />)}</div>}
      {data?.total_pages > 1 && (
        <nav className="pagination" aria-label={t('pagination.labelProducts')}>
          {page <= 1 ? <span className="page-disabled">{t('pagination.previous')}</span> : <a href={pageUrl(page - 1)}>{t('pagination.previous')}</a>}
          <span>{t('pagination.page', { page, total: data.total_pages })}</span>
          {page >= data.total_pages ? <span className="page-disabled">{t('pagination.next')}</span> : <a href={pageUrl(page + 1)}>{t('pagination.next')}</a>}
        </nav>
      )}
    </section>
  )
}
