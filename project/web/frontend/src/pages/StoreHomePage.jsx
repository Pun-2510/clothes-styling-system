import { useEffect, useRef, useState } from 'react'
import ErrorBanner from '../components/common/ErrorBanner'
import Icon from '../components/common/Icon'
import ProductCard from '../components/results/ProductCard'
import ProductImage from '../components/results/ProductImage'
import ResultsSection from '../components/results/ResultsSection'
import CombinedSearchForm from '../components/search/CombinedSearchForm'
import { categoryUrl, featuredCategories, getCategories, getProducts } from '../services/catalogApi'
import { clearSearchResult, loadSearchResult, saveSearchResult } from '../services/searchPersistence'

export default function StoreHomePage() {
  const [restoredSearch] = useState(loadSearchResult)
  const [categories, setCategories] = useState([])
  const [sections, setSections] = useState([])
  const [catalogError, setCatalogError] = useState('')
  const [searchMode, setSearchMode] = useState(restoredSearch?.mode || null)
  const [searchResult, setSearchResult] = useState(restoredSearch?.data || null)
  const [searchError, setSearchError] = useState('')
  const [searching, setSearching] = useState(false)
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

  const searchHandlers = {
    loading: searching,
    onSearchStart: () => { setSearching(true); clearSearchState() },
    onSearchSuccess: (data, mode) => {
      saveSearchResult(data, mode)
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
          <span className="eyebrow">CATALOG THỜI TRANG</span>
          <h1>Khám phá phong cách <em>phù hợp với bạn.</em></h1>
          <p>Duyệt theo danh mục hoặc dùng CLIP để tìm sản phẩm bằng ảnh, văn bản hay kết hợp cả hai.</p>
        </div>

        <div className="hero-model-search" id="search">
          <CombinedSearchForm
            {...searchHandlers}
            initialQuery={initialQuery}
            onClearResult={clearSearchState}
          />
          <ErrorBanner message={searchError} />
          <p className="search-hint"><Icon name="image" size={15} /> Nhập văn bản, thêm ảnh hoặc dùng cả hai trong cùng một tìm kiếm.</p>
        </div>

      </section>

      <ResultsSection data={searchResult} mode={searchMode} sectionRef={resultsRef} />

      {catalogError && <div className="state-message" role="alert">{catalogError}</div>}
      {!catalogError && categories.length === 0 && <div className="state-message">Đang tải catalog...</div>}

      {featured.length > 0 && (
        <section className="category-showcase">
          <div className="section-heading">
            <div><span className="eyebrow">CHỌN THEO DANH MỤC</span><h2>Khám phá theo danh mục</h2></div>
            <a href="/categories">Tất cả danh mục <Icon name="arrow" size={17} /></a>
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
              <span className="eyebrow">DANH MỤC NỔI BẬT</span>
              <h2>{category.name}</h2>
              <p>{category.count.toLocaleString('vi-VN')} sản phẩm trong catalog</p>
            </div>
            <a href={categoryUrl(category.name)}>Xem tất cả <Icon name="arrow" size={17} /></a>
          </div>
          <div className="product-grid product-grid--five">
            {products.map((item) => <ProductCard key={item.product_id} item={item} />)}
          </div>
        </section>
      ))}
    </>
  )
}
