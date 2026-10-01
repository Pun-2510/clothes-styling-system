import ProductCard from './ProductCard'

export default function ResultsSection({ data, mode }) {
  if (!data) return null

  const items = data.items || []
  const elapsed = mode === 'text' ? data.elapsed_seconds : data.processing_times?.total

  return (
    <section className="results" aria-live="polite">
      <div className="results-header">
        <div>
          <span className="eyebrow">Kết quả</span>
          <h2>{items.length} sản phẩm phù hợp</h2>
        </div>
        <div className="result-meta">
          {mode === 'text' && data.query_used !== data.query && data.query_used && (
            <span>Truy vấn: “{data.query_used}”</span>
          )}
          {mode === 'image' && data.category_mode === 'soft_category' && data.predicted_category && (
            <span>Ưu tiên {data.predicted_category} · {Math.round((data.category_confidence || 0) * 100)}% tin cậy</span>
          )}
          {mode === 'image' && data.category_mode === 'no_category' && (
            <span>Tự động tìm kiếm không giới hạn danh mục</span>
          )}
          <span>{Number(elapsed || 0).toFixed(2)} giây</span>
        </div>
      </div>

      {items.length > 0 ? (
        <div className="product-grid">
          {items.map((item, index) => (
            <ProductCard key={`${item.product_id}-${index}`} item={item} index={index} mode={mode} />
          ))}
        </div>
      ) : (
        <div className="empty-result">Không tìm thấy sản phẩm phù hợp với truy vấn này.</div>
      )}
    </section>
  )
}
