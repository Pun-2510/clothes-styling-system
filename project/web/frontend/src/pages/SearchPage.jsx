import { useState } from 'react'
import ErrorBanner from '../components/common/ErrorBanner'
import CombinedSearchForm from '../components/search/CombinedSearchForm'
import ResultsSection from '../components/results/ResultsSection'

export default function SearchPage() {
  const params = new URLSearchParams(window.location.search)
  const [searchMode, setSearchMode] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  function clearSearchState() {
    setResult(null)
    setSearchMode(null)
    setError('')
  }

  const handlers = {
    loading,
    onSearchStart: () => { setLoading(true); clearSearchState() },
    onSearchSuccess: (data, mode) => { setResult(data); setSearchMode(mode); setLoading(false) },
    onSearchError: (message) => { setError(message); setLoading(false) },
  }

  return (
    <section className="search-page">
      <nav className="breadcrumbs"><a href="/">Trang chủ</a><span>/</span><span>Tìm kiếm AI</span></nav>
      <div className="page-heading">
        <span className="eyebrow">FINE-TUNED CLIP</span>
        <h1>Tìm sản phẩm</h1>
        <p>Nhập mô tả, chọn ảnh hoặc sử dụng đồng thời cả hai. Hệ thống tự nhận biết loại truy vấn.</p>
      </div>
      <div className="compact-finder">
        <CombinedSearchForm
          {...handlers}
          initialQuery={params.get('q') || ''}
          requestImageFocus={params.get('image') === '1'}
          onClearResult={clearSearchState}
        />
        <ErrorBanner message={error} />
      </div>
      <ResultsSection data={result} mode={searchMode} />
    </section>
  )
}
