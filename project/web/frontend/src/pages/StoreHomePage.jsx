import { useEffect, useRef, useState } from 'react'
import ErrorBanner from '../components/common/ErrorBanner'
import Icon from '../components/common/Icon'
import ProductCard from '../components/results/ProductCard'
import CatalogSkeleton from '../components/results/CatalogSkeleton'
import ProductImage from '../components/results/ProductImage'
import ResultsSection from '../components/results/ResultsSection'
import CombinedSearchForm from '../components/search/CombinedSearchForm'
import { categoryUrl, featuredCategories, getCategories, getProducts } from '../services/catalogApi'
import { addSearchHistory, loadSearchHistoryEntry, loadSearchHistoryFile } from '../services/searchHistory'
import { clearSearchFile, clearSearchQueryState, clearSearchResult, loadSearchResult, saveSearchResult } from '../services/searchPersistence'
import { useLanguage } from '../i18n'

export default function StoreHomePage() {
  const { language, t } = useLanguage()
  const [restoredSearch] = useState(loadSearchResult)
  const [categories, setCategories] = useState([])
  const [sections, setSections] = useState([])
  const [catalogError, setCatalogError] = useState('')
  const [catalogLoading, setCatalogLoading] = useState(true)
  const [searchMode, setSearchMode] = useState(restoredSearch?.mode || null)
  const [searchResult, setSearchResult] = useState(restoredSearch?.data || null)
  const [searchError, setSearchError] = useState('')
  const [searching, setSearching] = useState(false)
  const [formResetVersion, setFormResetVersion] = useState(0)
  const [historyRestore, setHistoryRestore] = useState(null)
  const resultsRef = useRef(null)

  useEffect(() => {
    let active = true
    getCategories().then(async ({ categories: all }) => {
      const featured = featuredCategories(all, 5)
      const lists = await Promise.all(featured.map(async (category) => ({
        category,
        products: (await getProducts({ category: category.name, pageSize: 5 })).items,
      })))
      if (active) {
        setCategories(all)
        setSections(lists)
      }
    }).catch((failure) => { if (active) setCatalogError(failure.message) })
      .finally(() => { if (active) setCatalogLoading(false) })
    return () => { active = false }
  }, [])

  useEffect(() => {
    const historyId = new URLSearchParams(window.location.search).get('history')
    if (!historyId) return undefined

    let active = true
    const restoreHistory = async () => {
      const entry = loadSearchHistoryEntry(historyId)
      if (!entry) {
        if (active) setSearchError('Không tìm thấy lượt tìm kiếm đã lưu.')
        return
      }

      const file = entry.image ? await loadSearchHistoryFile(entry.id) : null
      if (!active) return
      if (entry.image && !file) {
        setSearchError('Ảnh của lượt tìm kiếm này không còn trong bộ nhớ trình duyệt.')
        return
      }

      setHistoryRestore({ ...entry, file })
      const url = new URL(window.location.href)
      url.searchParams.delete('history')
      window.history.replaceState({}, '', `${url.pathname}${url.search}${url.hash}`)
      window.requestAnimationFrame(() => {
        const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
        document.getElementById('search')?.scrollIntoView({
          behavior: reduceMotion ? 'auto' : 'smooth',
          block: 'center',
        })
      })
    }

    restoreHistory()
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (!searchResult || !resultsRef.current) return undefined

    const frame = window.requestAnimationFrame(() => {
      const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      resultsRef.current?.scrollIntoView({
        behavior: reduceMotion ? 'auto' : 'smooth',
        block: 'start',
      })
    })

    return () => window.cancelAnimationFrame(frame)
  }, [searchResult])

  const featured = featuredCategories(categories, 5)
  const initialQuery = new URLSearchParams(window.location.search).get('q') || ''

  function clearSearchState() {
    clearSearchResult()
    setSearchResult(null)
    setSearchMode(null)
    setSearchError('')
  }

  function startNewSearch() {
    clearSearchQueryState()
    clearSearchFile()
    clearSearchState()
    setSearching(false)
    setHistoryRestore(null)

    const url = new URL(window.location.href)
    url.searchParams.delete('q')
    window.history.replaceState({}, '', `${url.pathname}${url.search}${url.hash}`)
    setFormResetVersion((version) => version + 1)

    window.requestAnimationFrame(() => {
      const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      document.getElementById('search')?.scrollIntoView({
        behavior: reduceMotion ? 'auto' : 'smooth',
        block: 'center',
      })
    })
  }

  const searchHandlers = {
    loading: searching,
    onSearchStart: () => { setSearching(true); clearSearchState() },
    onSearchSuccess: (data, mode, requestContext) => {
      saveSearchResult(data, mode)
      addSearchHistory({ ...requestContext, data, mode }).catch(() => {})
      setSearchResult(data)
      setSearchMode(mode)
      setSearching(false)
    },
    onSearchError: (message) => {
      setSearchError(message)
      setSearching(false)
    },
  }

  return (
    <>
      <section className="store-hero">
        <div className="store-hero-intro">
          <span className="eyebrow">{t('hero.eyebrow')}</span>
          <h1>{t('hero.titlePrefix')} <em>CLIP</em></h1>
          <p>{t('hero.description')}</p>
        </div>

        <div className="hero-model-search" id="search">
          <CombinedSearchForm
            key={formResetVersion}
            {...searchHandlers}
            initialQuery={initialQuery}
            restoreRequest={historyRestore}
            onClearResult={clearSearchState}
          />
          <ErrorBanner message={searchError} />
          <p className="search-hint"><Icon name="image" size={15} /> {t('search.hint')}</p>
        </div>

      </section>

      <ResultsSection
        data={searchResult}
        mode={searchMode}
        sectionRef={resultsRef}
        onNewSearch={startNewSearch}
      />

      {catalogError && <div className="state-message" role="alert">{catalogError}</div>}
      {catalogLoading && <CatalogSkeleton />}
      {!catalogError && !catalogLoading && categories.length === 0 && <div className="state-message">{t('catalog.empty')}</div>}

      {featured.length > 0 && (
        <section className="category-showcase">
          <div className="section-heading">
            <div><span className="eyebrow">{t('catalog.chooseEyebrow')}</span><h2>{t('catalog.exploreCategories')}</h2></div>
            <a href="/categories">{t('catalog.allCategories')} <Icon name="arrow" size={17} /></a>
          </div>
          <div className="category-showcase-grid">
            {featured.slice(0, 4).map((category) => (
              <a className="showcase-tile" key={category.name} href={categoryUrl(category.name)}>
                <div><ProductImage reference={category.image_reference} name={category.name} /></div>
                <span>{category.name}<Icon name="arrow" size={17} /></span>
              </a>
            ))}
          </div>
        </section>
      )}

      {sections.map(({ category, products }) => (
        <section className="catalog-section" key={category.name}>
          <div className="section-heading">
            <div>
              <span className="eyebrow">{t('catalog.featured')}</span>
              <h2>{category.name}</h2>
              <p>{t('catalog.productCount', { count: category.count.toLocaleString(language === 'en' ? 'en-US' : 'vi-VN') })}</p>
            </div>
            <a href={categoryUrl(category.name)}>{t('catalog.seeAll')} <Icon name="arrow" size={17} /></a>
          </div>
          <div className="product-grid product-grid--five">
            {products.map((item) => <ProductCard key={item.product_id} item={item} />)}
          </div>
        </section>
      ))}
    </>
  )
}
