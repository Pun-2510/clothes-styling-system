import { useEffect, useRef, useState } from 'react'
import Icon from '../common/Icon'
import ProductImage from '../results/ProductImage'
import SearchHistoryDrawer from './SearchHistoryDrawer'
import { categoryUrl, featuredCategories, getCategories } from '../../services/catalogApi'
import { clearSearchHistory, deleteSearchHistoryEntry, loadSearchHistory } from '../../services/searchHistory'
import { useLanguage } from '../../i18n'

export default function Header({ serverStatus, onRetry }) {
  const { language, setLanguage, t } = useLanguage()
  const path = window.location.pathname
  const [categories, setCategories] = useState([])
  const [categoryMenuOpen, setCategoryMenuOpen] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [historyEntries, setHistoryEntries] = useState(loadSearchHistory)
  const categoryMenuRef = useRef(null)

  useEffect(() => {
    let active = true
    getCategories()
      .then((data) => { if (active) setCategories(featuredCategories(data.categories, 12)) })
      .catch(() => {})
    return () => { active = false }
  }, [])

  useEffect(() => {
    const refreshHistory = () => setHistoryEntries(loadSearchHistory())
    window.addEventListener('thread-find:history-changed', refreshHistory)
    window.addEventListener('storage', refreshHistory)
    return () => {
      window.removeEventListener('thread-find:history-changed', refreshHistory)
      window.removeEventListener('storage', refreshHistory)
    }
  }, [])

  async function deleteHistoryItem(id) {
    await deleteSearchHistoryEntry(id)
    setHistoryEntries(loadSearchHistory())
  }

  async function clearHistoryItems() {
    await clearSearchHistory()
    setHistoryEntries([])
  }

  useEffect(() => {
    function closeMenu(event) {
      if (!categoryMenuRef.current?.contains(event.target)) setCategoryMenuOpen(false)
    }

    function closeOnEscape(event) {
      if (event.key === 'Escape') setCategoryMenuOpen(false)
    }

    document.addEventListener('pointerdown', closeMenu)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('pointerdown', closeMenu)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [])

  const categoryActive = path === '/categories' || path.startsWith('/category/')

  return (
    <header className="site-header">
      <a className="brand" href="/" aria-label={`THREAD/FIND · ${t('header.home')}`}>
        <span className="brand-mark">TF</span>
        <span>THREAD/FIND</span>
      </a>
      <nav className="main-nav" aria-label="Điều hướng chính">
        <a className={path === '/' ? 'active' : ''} href="/">{t('header.home')}</a>
        <div className="category-menu" ref={categoryMenuRef}>
          <button
            className={categoryActive || categoryMenuOpen ? 'active' : ''}
            type="button"
            aria-haspopup="menu"
            aria-expanded={categoryMenuOpen}
            onClick={() => {
              setHistoryOpen(false)
              setCategoryMenuOpen((open) => !open)
            }}
          >
            {t('header.categories')} <Icon name="chevron" size={15} />
          </button>
          {categoryMenuOpen && (
            <>
              <div className="category-menu-backdrop" aria-hidden="true" onClick={() => setCategoryMenuOpen(false)} />
              <div className="category-popover" role="menu">
                <span className="category-popover-title">{t('header.categoryMenu')}</span>
                <div className="category-popover-links">
                  {categories.map((category) => (
                    <a key={category.name} href={categoryUrl(category.name)} role="menuitem">
                      <span className="category-popover-image">
                        <ProductImage reference={category.image_reference} name={category.name} />
                      </span>
                      <span>{category.name}</span>
                    </a>
                  ))}
                </div>
                <a className="category-more-link" href="/categories" role="menuitem">
                  {t('header.allCategories')} <Icon name="arrow" size={17} />
                </a>
              </div>
            </>
          )}
        </div>
      </nav>
      <div className="header-actions">
        <div className="language-switch" role="group" aria-label={t('language.label')}>
          <button className={language === 'vi' ? 'active' : ''} type="button" onClick={() => setLanguage('vi')} aria-pressed={language === 'vi'}>VI</button>
          <button className={language === 'en' ? 'active' : ''} type="button" onClick={() => setLanguage('en')} aria-pressed={language === 'en'}>EN</button>
        </div>
        <button
          className={`history-trigger ${historyOpen ? 'active' : ''}`}
          type="button"
          aria-expanded={historyOpen}
          onClick={() => {
            setCategoryMenuOpen(false)
            setHistoryOpen((open) => !open)
          }}
        >
          <Icon name="history" size={17} />
          <span className="history-trigger-label">{t('header.history')}</span>
          {historyEntries.length > 0 && <span className="history-count">{historyEntries.length}</span>}
        </button>
        <div className={`status status--${serverStatus}`} title={t(`status.${serverStatus}`)}>
          <span className="status-dot" />
          <span className="status-label">{t(`status.${serverStatus}`)}</span>
          {serverStatus === 'offline' && (
            <button type="button" onClick={onRetry} aria-label={t('status.retry')}>
              <Icon name="retry" size={15} />
            </button>
          )}
        </div>
      </div>
      <SearchHistoryDrawer
        open={historyOpen}
        entries={historyEntries}
        onClose={() => setHistoryOpen(false)}
        onDelete={deleteHistoryItem}
        onClear={clearHistoryItems}
      />
    </header>
  )
}
