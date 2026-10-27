import Footer from './components/layout/Footer'
import Header from './components/layout/Header'
import LoadingBar from './components/common/LoadingBar'
import useServerStatus from './hooks/useServerStatus'
import CategoriesPage from './pages/CategoriesPage'
import CategoryPage from './pages/CategoryPage'
import ProductPage from './pages/ProductPage'
import StoreHomePage from './pages/StoreHomePage'
import DeveloperPage from './pages/DeveloperPage'
import { useLanguage } from './i18n'

function App() {
  const { t } = useLanguage()
  const { status, checkStatus } = useServerStatus()
  const path = window.location.pathname
  let page

  if (path === '/' || path === '/search') page = <StoreHomePage />
  else if (path === '/categories') page = <CategoriesPage />
  else if (path === '/developer') page = <DeveloperPage />
  else if (path.startsWith('/category/')) {
    page = <CategoryPage key={path} name={decodeURIComponent(path.slice(10))} />
  } else if (path.startsWith('/product/')) {
    page = <ProductPage key={path} productId={decodeURIComponent(path.slice(9))} />
  } else {
    page = <div className="state-message"><h1>{t('notFound.title')}</h1><a href="/">{t('notFound.home')}</a></div>
  }

  return (
    <div className="app-shell">
      <Header serverStatus={status} onRetry={checkStatus} />
      <LoadingBar />
      <main id="top">{page}</main>
      <Footer />
    </div>
  )
}

export default App
