import { useEffect, useMemo, useState } from 'react'
import ProductImage from '../components/results/ProductImage'
import { ProductGridSkeleton } from '../components/results/CatalogSkeleton'
import { categoryUrl, getCategories } from '../services/catalogApi'
import { useLanguage } from '../i18n'

const PAGE_SIZE = 24

export default function CategoriesPage() {
  const { language, t } = useLanguage()
  const [categories, setCategories] = useState([])
  const [filter, setFilter] = useState('')
  const [page, setPage] = useState(1)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    getCategories().then((data) => { if (active) setCategories(data.categories) })
      .catch((failure) => { if (active) setError(failure.message) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [])

  const filtered = useMemo(
    () => categories.filter((item) => item.name.toLocaleLowerCase().includes(filter.trim().toLocaleLowerCase())),
    [categories, filter],
  )
  const totalPages = Math.ceil(filtered.length / PAGE_SIZE)
  const visible = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  return (
    <section className="directory-page">
      <nav className="breadcrumbs"><a href="/">{t('categories.home')}</a><span>/</span><span>{t('header.categories')}</span></nav>
      <div className="page-heading">
        <span className="eyebrow">{t('categories.eyebrow')}</span>
        <h1>{t('categories.title')}</h1>
        <p>{t('categories.description')}</p>
      </div>
      <div className="directory-tools">
        <input
          type="search"
          value={filter}
          onChange={(event) => { setFilter(event.target.value); setPage(1) }}
          placeholder={t('categories.filter')}
          aria-label={t('categories.filterLabel')}
        />
        <span>{t('categories.count', { count: filtered.length })}</span>
      </div>
      {error && <div className="state-message" role="alert">{error}</div>}
      {loading && <ProductGridSkeleton count={12} className="category-grid" />}
      {!error && !loading && categories.length === 0 && <div className="state-message">{t('categories.empty')}</div>}
      {categories.length > 0 && filtered.length === 0 && <div className="state-message">{t('categories.noMatch')}</div>}
      {!loading && <div className="category-grid">
        {visible.map((item) => (
          <a className="category-tile" key={item.name} href={categoryUrl(item.name)}>
            <div className="category-tile-image"><ProductImage reference={item.image_reference} name={item.name} /></div>
            <div><h2>{item.name}</h2><span>{t('search.products', { count: item.count.toLocaleString(language === 'en' ? 'en-US' : 'vi-VN') })}</span></div>
          </a>
        ))}
      </div>}
      {totalPages > 1 && (
        <nav className="pagination" aria-label={t('pagination.labelCategories')}>
          <button disabled={page === 1} onClick={() => { setPage(page - 1); window.scrollTo(0, 0) }}>{t('pagination.previous')}</button>
          <span>{t('pagination.page', { page, total: totalPages })}</span>
          <button disabled={page === totalPages} onClick={() => { setPage(page + 1); window.scrollTo(0, 0) }}>{t('pagination.next')}</button>
        </nav>
      )}
    </section>
  )
}
