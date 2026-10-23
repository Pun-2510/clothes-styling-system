import { useEffect, useState } from 'react'
import ProductCard from '../components/results/ProductCard'
import ProductImage from '../components/results/ProductImage'
import { categoryUrl, getProduct } from '../services/catalogApi'

export default function ProductPage({ productId }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    getProduct(productId).then((response) => { if (active) setData(response) })
      .catch((failure) => { if (active) setError(failure.message) })
    return () => { active = false }
  }, [productId])

  if (error) return <div className="state-message" role="alert">{error} <a href="/categories">Xem danh mục</a></div>
  if (!data) return <div className="state-message">Đang tải chi tiết sản phẩm...</div>

  const { product, related } = data
  return (
    <>
      <nav className="breadcrumbs">
        <a href="/">Trang chủ</a><span>/</span><a href={categoryUrl(product.category)}>{product.category}</a><span>/</span><span>{product.product_name}</span>
      </nav>
      <section className="product-detail">
        <div className="detail-visual"><ProductImage reference={product.image_reference} name={product.product_name} /></div>
        <div className="detail-copy">
          <span className="eyebrow">{product.category || 'SẢN PHẨM'}</span>
          <h1>{product.product_name}</h1>
          <span className="detail-id">Mã sản phẩm: {product.product_id}</span>
          <div className="detail-divider" />
          <h2>Thông tin sản phẩm</h2>
          <p>{product.description || 'Sản phẩm chưa có mô tả chi tiết trong catalog.'}</p>
          <a className="detail-category-link" href={categoryUrl(product.category)}>Xem thêm {product.category} →</a>
        </div>
      </section>
      {related.length > 0 && (
        <section className="catalog-section related-section">
          <div className="section-heading"><div><span className="eyebrow">GỢI Ý THÊM</span><h2>Cùng danh mục</h2></div><a href={categoryUrl(product.category)}>Xem tất cả →</a></div>
          <div className="product-grid product-grid--five">{related.map((item) => <ProductCard key={item.product_id} item={item} />)}</div>
        </section>
      )}
    </>
  )
}
