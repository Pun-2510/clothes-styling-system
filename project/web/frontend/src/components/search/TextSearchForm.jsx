import { useState } from 'react'
import Icon from '../common/Icon'
import { RESULT_COUNTS, TEXT_EXAMPLES } from '../../data/searchOptions'
import { searchByText } from '../../services/recommendationApi'

export default function TextSearchForm({ loading, onSearchStart, onSearchSuccess, onSearchError }) {
  const [query, setQuery] = useState('')
  const [language, setLanguage] = useState('vi')
  const [topK, setTopK] = useState(5)
  const [imageWeight, setImageWeight] = useState(0.5)

  async function submit(event) {
    event.preventDefault()
    if (!query.trim() || loading) return

    onSearchStart()
    try {
      const result = await searchByText({ query, language, topK, imageWeight })
      onSearchSuccess(result)
    } catch (error) {
      onSearchError(error.message)
    }
  }

  return (
    <form className="search-panel" onSubmit={submit}>
      <div className="field-heading">
        <label htmlFor="fashion-query">Bạn đang tìm món đồ như thế nào?</label>
        <span>Mô tả màu sắc, kiểu dáng hoặc dịp sử dụng.</span>
      </div>

      <div className="query-box">
        <textarea
          id="fashion-query"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={language === 'vi' ? 'Ví dụ: váy mùa hè màu đỏ, nhẹ và nữ tính...' : 'Example: minimal white sneakers for everyday wear...'}
          maxLength={512}
          rows={3}
        />
        <span>{query.length}/512</span>
      </div>

      <div className="examples" aria-label="Gợi ý tìm kiếm">
        <span>Gợi ý:</span>
        {TEXT_EXAMPLES.map((example) => (
          <button type="button" key={example} onClick={() => { setQuery(example); setLanguage('vi') }}>
            {example}
          </button>
        ))}
      </div>

      <div className="form-controls">
        <div className="control-group">
          <span className="control-label">Ngôn ngữ</span>
          <div className="segmented-control">
            <button type="button" className={language === 'vi' ? 'active' : ''} onClick={() => setLanguage('vi')}>Tiếng Việt</button>
            <button type="button" className={language === 'en' ? 'active' : ''} onClick={() => setLanguage('en')}>English</button>
          </div>
        </div>

        <label className="control-group select-control">
          <span className="control-label">Số kết quả</span>
          <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
            {RESULT_COUNTS.map((value) => <option key={value} value={value}>{value} sản phẩm</option>)}
          </select>
        </label>

        <label className="control-group range-control">
          <span className="control-label">Trọng số hình ảnh: {Math.round(imageWeight * 100)}%</span>
          <input type="range" min="0" max="1" step="0.05" value={imageWeight} onChange={(event) => setImageWeight(Number(event.target.value))} />
        </label>
      </div>

      <button className="primary-button" type="submit" disabled={!query.trim() || loading}>
        {loading ? <span className="spinner" /> : <Icon name="search" />}
        {loading ? 'Đang tìm kiếm...' : 'Tìm sản phẩm'}
        {!loading && <Icon name="arrow" />}
      </button>
    </form>
  )
}
