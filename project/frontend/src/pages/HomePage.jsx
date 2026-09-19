import { useState } from 'react'
import ErrorBanner from '../components/common/ErrorBanner'
import Hero from '../components/home/Hero'
import HowItWorks from '../components/home/HowItWorks'
import Footer from '../components/layout/Footer'
import Header from '../components/layout/Header'
import ResultsSection from '../components/results/ResultsSection'
import ImageSearchForm from '../components/search/ImageSearchForm'
import SearchTabs from '../components/search/SearchTabs'
import TextSearchForm from '../components/search/TextSearchForm'
import useServerStatus from '../hooks/useServerStatus'

export default function HomePage() {
  const [activeTab, setActiveTab] = useState('text')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { status, checkStatus } = useServerStatus()

  function clearSearchState() {
    setResult(null)
    setError('')
  }

  function switchTab(tab) {
    setActiveTab(tab)
    setLoading(false)
    clearSearchState()
  }

  const searchHandlers = {
    loading,
    onSearchStart: () => {
      setLoading(true)
      clearSearchState()
    },
    onSearchSuccess: (data) => {
      setResult(data)
      setLoading(false)
    },
    onSearchError: (message) => {
      setError(message)
      setLoading(false)
    },
  }

  return (
    <div className="app-shell">
      <Header serverStatus={status} onRetry={checkStatus} />
      <main id="top">
        <Hero />

        <section className="finder-section">
          <SearchTabs activeTab={activeTab} onChange={switchTab} />
          {activeTab === 'text' ? (
            <TextSearchForm {...searchHandlers} />
          ) : (
            <ImageSearchForm {...searchHandlers} onClearResult={clearSearchState} />
          )}
          <ErrorBanner message={error} />
        </section>

        <ResultsSection data={result} mode={activeTab} />
        {!result && !loading && <HowItWorks />}
      </main>
      <Footer />
    </div>
  )
}
