import { useEffect, useMemo, useState } from 'react'
import ProductImage from '../components/results/ProductImage'
import { ProductGridSkeleton } from '../components/results/CatalogSkeleton'
import { categoryUrl, getCategories } from '../services/catalogApi'

const PAGE_SIZE = 24

export default function CategoriesPage() {
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
      <nav className="breadcrumbs"><a href="/">Trang chủ</a><span>/</span><span>Danh mục</span></nav>
      <div className="page-heading">
        <span className="eyebrow">KHÁM PHÁ CATALOG</span>
        <h1>Danh mục sản phẩm</h1>
        <p>Chọn một thể loại để xem toàn bộ các sản phẩm thuộc nhóm đó.</p>
      </div>
      <div className="directory-tools">
        <input
          type="search"
          value={filter}
          onChange={(event) => { setFilter(event.target.value); setPage(1) }}
          placeholder="Lọc tên danh mục..."
          aria-label="Lọc danh mục"
        />
        <span>{filtered.length} danh mục</span>
      </div>
      {error && <div className="state-message" role="alert">{error}</div>}
      {loading && <ProductGridSkeleton count={12} className="category-grid" />}
      {!error && !loading && categories.length === 0 && <div className="state-message">Catalog chưa có danh mục.</div>}
      {categories.length > 0 && filtered.length === 0 && <div className="state-message">Không có danh mục phù hợp.</div>}
      {!loading && <div className="category-grid">
        {visible.map((item) => (
          <a className="category-tile" key={item.name} href={categoryUrl(item.name)}>
            <div className="category-tile-image"><ProductImage reference={item.image_reference} name={item.name} /></div>
            <div><h2>{item.name}</h2><span>{item.count.toLocaleString('vi-VN')} sản phẩm</span></div>
          </a>
        ))}
      </div>}
      {totalPages > 1 && (
        <nav className="pagination" aria-label="Phân trang danh mục">
          <button disabled={page === 1} onClick={() => { setPage(page - 1); window.scrollTo(0, 0) }}>Trước</button>
          <span>Trang {page} / {totalPages}</span>
          <button disabled={page === totalPages} onClick={() => { setPage(page + 1); window.scrollTo(0, 0) }}>Sau</button>
        </nav>
      )}
    </section>
  )
}
