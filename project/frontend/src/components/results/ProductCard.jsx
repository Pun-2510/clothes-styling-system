import Icon from '../common/Icon'
import ProductImage from './ProductImage'

export default function ProductCard({ item, index, mode }) {
  const finalScore = Number(item.final_score ?? 0)
  const detailScore = mode === 'text' ? item.text_to_image_score : item.visual_similarity

  return (
    <article className="product-card">
      <div className="product-visual">
        <ProductImage reference={item.image_reference} name={item.product_name} />
        <span className="rank">#{index + 1}</span>
        <span className="score-pill">{finalScore.toFixed(3)}</span>
      </div>
      <div className="product-copy">
        <span className="category">{item.category || 'Thời trang'}</span>
        <h3>{item.product_name || 'Sản phẩm không tên'}</h3>
        <div className="score-row">
          <span>{mode === 'text' ? 'Độ khớp hình ảnh' : 'Độ tương đồng'}</span>
          <strong>{Number(detailScore ?? finalScore).toFixed(3)}</strong>
        </div>
        {mode === 'text' && item.text_to_text_score != null && (
          <div className="score-row score-row--secondary">
            <span>Độ khớp mô tả</span>
            <strong>{Number(item.text_to_text_score).toFixed(3)}</strong>
          </div>
        )}
        {mode === 'image' && Number(item.category_bonus) > 0 && (
          <div className="bonus"><Icon name="spark" size={14} />+{Number(item.category_bonus).toFixed(2)} cùng danh mục</div>
        )}
      </div>
    </article>
  )
}
