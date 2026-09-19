import { useEffect, useMemo, useRef, useState } from 'react'
import Icon from '../common/Icon'
import { ACCEPTED_IMAGE_TYPES, CATEGORY_MODES, MAX_IMAGE_SIZE, RESULT_COUNTS } from '../../data/searchOptions'
import { searchByImage } from '../../services/recommendationApi'

export default function ImageSearchForm({ loading, onSearchStart, onSearchSuccess, onSearchError, onClearResult }) {
  const [file, setFile] = useState(null)
  const [topK, setTopK] = useState(5)
  const [categoryMode, setCategoryMode] = useState('soft_category')
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)
  const preview = useMemo(() => file ? URL.createObjectURL(file) : '', [file])

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

  async function submit(event) {
    event.preventDefault()
    if (!file || loading) return

    onSearchStart()
    try {
      const result = await searchByImage({ file, topK, categoryMode })
      onSearchSuccess(result)
    } catch (error) {
      onSearchError(error.message)
    }
  }

  return (
    <form className="search-panel" onSubmit={submit}>
      <div className="field-heading">
        <span className="field-title">Tải lên món đồ bạn yêu thích</span>
        <span>Hệ thống sẽ tìm các sản phẩm có phong cách tương đồng.</span>
      </div>

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
            <strong>Kéo thả hoặc chọn ảnh</strong>
            <span>JPG, PNG, WebP · tối đa 5 MB</span>
          </button>
        )}
      </div>

      <div className="mode-picker">
        <span className="control-label">Cách lọc danh mục</span>
        <div className="mode-options">
          {CATEGORY_MODES.map((mode) => (
            <button type="button" key={mode.value} className={categoryMode === mode.value ? 'active' : ''} onClick={() => setCategoryMode(mode.value)}>
              <span>{mode.label}</span>
              <small>{mode.hint}</small>
            </button>
          ))}
        </div>
      </div>

      <div className="image-actions">
        <label className="control-group select-control">
          <span className="control-label">Số kết quả</span>
          <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
            {RESULT_COUNTS.map((value) => <option key={value} value={value}>{value} sản phẩm</option>)}
          </select>
        </label>
        <button className="primary-button" type="submit" disabled={!file || loading}>
          {loading ? <span className="spinner" /> : <Icon name="spark" />}
          {loading ? 'Đang phân tích...' : 'Tìm sản phẩm tương tự'}
          {!loading && <Icon name="arrow" />}
        </button>
      </div>
    </form>
  )
}
