import { useState } from 'react'
import ErrorBanner from '../components/common/ErrorBanner'
import Hero from '../components/home/Hero'
import HowItWorks from '../components/home/HowItWorks'
import Footer from '../components/layout/Footer'
import Header from '../components/layout/Header'
import ResultsSection from '../components/results/ResultsSection'
import CombinedSearchForm from '../components/search/CombinedSearchForm'
import useServerStatus from '../hooks/useServerStatus'

export default function HomePage() {
  const [searchMode, setSearchMode] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { status, checkStatus } = useServerStatus()

  function clearSearchState() {
    setResult(null)
    setSearchMode(null)
    setError('')
  }

  const searchHandlers = {
    loading,
    onSearchStart: () => {
      setLoading(true)
      clearSearchState()
    },
    onSearchSuccess: (data, mode) => {
      setResult(data)
      setSearchMode(mode)
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
          <CombinedSearchForm {...searchHandlers} onClearResult={clearSearchState} />
          <ErrorBanner message={error} />
        </section>

        <ResultsSection data={result} mode={searchMode} />
        {!result && !loading && <HowItWorks />}
      </main>
      <Footer />
    </div>
  )
}
