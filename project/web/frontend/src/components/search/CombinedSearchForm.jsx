import { useEffect, useMemo, useRef, useState } from 'react'
import Icon from '../common/Icon'
import { ACCEPTED_IMAGE_TYPES, MAX_IMAGE_SIZE, RESULT_COUNTS } from '../../data/searchOptions'
import { searchByCombined, searchByImage, searchByText } from '../../services/recommendationApi'

export default function CombinedSearchForm({ loading, onSearchStart, onSearchSuccess, onSearchError, onClearResult }) {
  const [file, setFile] = useState(null)
  const [query, setQuery] = useState('')
  const [language, setLanguage] = useState('vi')
  const [topK, setTopK] = useState(5)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)
  const preview = useMemo(() => file ? URL.createObjectURL(file) : '', [file])
  const hasImage = Boolean(file)
  const hasText = Boolean(query.trim())
  const canSearch = hasImage || hasText

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview)
  }, [preview])

  function acceptFile(candidate) {
    if (!candidate) return
    onClearResult()
    if (!ACCEPTED_IMAGE_TYPES.includes(candidate.type)) {
      onSearchError('Chỉ hỗ trợ ảnh JPG, PNG hoặc WebP.')
      return
    }
    if (candidate.size > MAX_IMAGE_SIZE) {
      onSearchError('Ảnh không được vượt quá 5 MB.')
      return
    }
    setFile(candidate)
  }

  function updateQuery(value) {
    setQuery(value)
    onClearResult()
  }

  async function submit(event) {
    event.preventDefault()
    if (!canSearch || loading) return

    onSearchStart()
    try {
      if (hasImage && hasText) {
        const result = await searchByCombined({ file, query, language, topK })
        onSearchSuccess(result, 'combined')
      } else if (hasImage) {
        const result = await searchByImage({ file, topK })
        onSearchSuccess(result, 'image')
      } else {
        const result = await searchByText({ query, language, topK })
        onSearchSuccess(result, 'text')
      }
    } catch (error) {
      onSearchError(error.message)
    }
  }

  const submitLabel = hasImage && hasText
    ? 'Tìm bằng ảnh và mô tả'
    : hasImage
      ? 'Tìm bằng ảnh'
      : hasText
        ? 'Tìm bằng mô tả'
        : 'Nhập mô tả hoặc chọn ảnh'

  return (
    <form className="search-panel" onSubmit={submit}>
      <div className="field-heading">
        <span className="field-title">Tìm sản phẩm</span>
        <span>Nhập mô tả, tải ảnh lên hoặc sử dụng đồng thời cả hai.</span>
      </div>

      <div className="combined-inputs">
        <div>
          <input ref={inputRef} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => acceptFile(event.target.files?.[0])} />
          <div
            className={`dropzone ${dragging ? 'dragging' : ''} ${preview ? 'has-preview' : ''}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => { event.preventDefault(); setDragging(false); acceptFile(event.dataTransfer.files?.[0]) }}
          >
            {preview ? (
              <>
                <img src={preview} alt="Ảnh truy vấn đã chọn" />
                <div className="preview-meta">
                  <div><Icon name="check" size={18} /><span><strong>{file.name}</strong><small>{(file.size / 1024 / 1024).toFixed(2)} MB</small></span></div>
                  <button type="button" aria-label="Xóa ảnh" onClick={() => { setFile(null); onClearResult() }}><Icon name="close" /></button>
                </div>
              </>
            ) : (
              <button type="button" className="dropzone-action" onClick={() => inputRef.current?.click()}>
                <Icon name="upload" size={26} />
                <strong>Chọn ảnh nếu có</strong>
                <span>JPG, PNG, WebP · tối đa 5 MB</span>
              </button>
            )}
          </div>
        </div>

        <div className="combined-text">
          <div className="query-box">
            <textarea
              id="combined-fashion-query"
              value={query}
              onChange={(event) => updateQuery(event.target.value)}
              placeholder={language === 'vi' ? 'Nhập mô tả nếu có. Ví dụ: váy màu đen phù hợp đi làm...' : 'Add a description if needed. Example: a black dress suitable for work...'}
              maxLength={512}
              rows={7}
            />
            <span>{query.length}/512</span>
          </div>
        </div>
      </div>

      <div className="form-controls">
        {hasText && (
          <div className="control-group">
            <span className="control-label">Ngôn ngữ mô tả</span>
            <div className="segmented-control">
              <button type="button" className={language === 'vi' ? 'active' : ''} onClick={() => setLanguage('vi')}>Tiếng Việt</button>
              <button type="button" className={language === 'en' ? 'active' : ''} onClick={() => setLanguage('en')}>English</button>
            </div>
          </div>
        )}

        <label className="control-group select-control">
          <span className="control-label">Số kết quả</span>
          <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
            {RESULT_COUNTS.map((value) => <option key={value} value={value}>{value} sản phẩm</option>)}
          </select>
        </label>

      </div>

      <button className="primary-button" type="submit" disabled={!canSearch || loading}>
        {loading ? <span className="spinner" /> : <Icon name="spark" />}
        {loading ? 'Đang tìm kiếm...' : submitLabel}
        {!loading && canSearch && <Icon name="arrow" />}
      </button>
    </form>
  )
}
