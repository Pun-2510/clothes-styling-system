import { useEffect, useState } from 'react'
import ProductCard from '../components/results/ProductCard'
import { getProducts } from '../services/catalogApi'

export default function CategoryPage({ name }) {
  const requestedPage = Number(new URLSearchParams(window.location.search).get('page'))
  const page = Number.isInteger(requestedPage) && requestedPage > 0 ? requestedPage : 1
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    getProducts({ category: name, page, pageSize: 24 })
      .then((response) => { if (active) setData(response) })
      .catch((failure) => { if (active) setError(failure.message) })
    return () => { active = false }
  }, [name, page])

  const pageUrl = (number) => `/category/${encodeURIComponent(name)}?page=${number}`

  return (
    <section className="listing-page">
      <nav className="breadcrumbs"><a href="/">Trang chủ</a><span>/</span><a href="/categories">Danh mục</a><span>/</span><span>{name}</span></nav>
      <div className="page-heading">
        <span className="eyebrow">DANH MỤC</span>
        <h1>{name}</h1>
        <p>{data ? `${data.total.toLocaleString('vi-VN')} sản phẩm` : 'Đang tải sản phẩm...'}</p>
      </div>
      {error && <div className="state-message" role="alert">{error}</div>}
      {data && <div className="product-grid category-product-grid">{data.items.map((item) => <ProductCard key={item.product_id} item={item} />)}</div>}
      {data?.total_pages > 1 && (
        <nav className="pagination" aria-label="Phân trang sản phẩm">
          {page <= 1 ? <span className="page-disabled">Trước</span> : <a href={pageUrl(page - 1)}>Trước</a>}
          <span>Trang {page} / {data.total_pages}</span>
          {page >= data.total_pages ? <span className="page-disabled">Sau</span> : <a href={pageUrl(page + 1)}>Sau</a>}
        </nav>
      )}
    </section>
  )
}
