import { useEffect, useRef, useState } from 'react'
import Icon from '../common/Icon'
import ProductImage from '../results/ProductImage'
import { categoryUrl, featuredCategories, getCategories } from '../../services/catalogApi'

const STATUS_LABELS = {
  ready: 'AI sẵn sàng',
  checking: 'Đang kết nối',
  offline: 'AI chưa sẵn sàng',
}

export default function Header({ serverStatus, onRetry }) {
  const path = window.location.pathname
  const [categories, setCategories] = useState([])
  const [categoryMenuOpen, setCategoryMenuOpen] = useState(false)
  const categoryMenuRef = useRef(null)

  useEffect(() => {
    let active = true
    getCategories()
      .then((data) => { if (active) setCategories(featuredCategories(data.categories, 12)) })
      .catch(() => {})
    return () => { active = false }
  }, [])

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
      <a className="brand" href="/" aria-label="THREAD/FIND trang chủ">
        <span className="brand-mark">TF</span>
        <span>THREAD/FIND</span>
      </a>
      <nav className="main-nav" aria-label="Điều hướng chính">
        <a className={path === '/' ? 'active' : ''} href="/">Trang chủ</a>
        <div className="category-menu" ref={categoryMenuRef}>
          <button
            className={categoryActive || categoryMenuOpen ? 'active' : ''}
            type="button"
            aria-haspopup="menu"
            aria-expanded={categoryMenuOpen}
            onClick={() => setCategoryMenuOpen((open) => !open)}
          >
            Danh mục <Icon name="chevron" size={15} />
          </button>
          {categoryMenuOpen && (
            <>
              <div className="category-menu-backdrop" aria-hidden="true" onClick={() => setCategoryMenuOpen(false)} />
              <div className="category-popover" role="menu">
                <span className="category-popover-title">Tìm theo danh mục</span>
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
                  Xem tất cả danh mục sản phẩm <Icon name="arrow" size={17} />
                </a>
              </div>
            </>
          )}
        </div>
      </nav>
      <div className="header-actions">
        <div className={`status status--${serverStatus}`} title={STATUS_LABELS[serverStatus]}>
          <span className="status-dot" />
          <span className="status-label">{STATUS_LABELS[serverStatus]}</span>
          {serverStatus === 'offline' && (
            <button type="button" onClick={onRetry} aria-label="Thử kết nối lại">
              <Icon name="retry" size={15} />
            </button>
          )}
        </div>
      </div>
    </header>
  )
}
