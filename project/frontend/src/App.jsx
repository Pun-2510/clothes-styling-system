import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''
const MAX_IMAGE_SIZE = 5 * 1024 * 1024

const textExamples = [
  'váy nữ màu đỏ có họa tiết hoa',
  'áo sơ mi nam màu xanh thanh lịch',
  'giày thể thao trắng tối giản',
]

const categoryModes = [
  { value: 'no_category', label: 'Tự do', hint: 'Chỉ dùng độ tương đồng hình ảnh' },
  { value: 'soft_category', label: 'Cân bằng', hint: 'Ưu tiên nhẹ sản phẩm cùng loại' },
  { value: 'hard_category', label: 'Chính xác', hint: 'Chỉ lấy sản phẩm cùng loại' },
]

function Icon({ name, size = 20 }) {
  const paths = {
    search: <><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>,
    image: <><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9" r="1.5"/><path d="m21 15-5-5L5 20"/></>,
    upload: <><path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M5 20h14"/></>,
    spark: <><path d="m12 3 1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3Z"/><path d="m19 15 .7 2.3L22 18l-2.3.7L19 21l-.7-2.3L16 18l2.3-.7L19 15Z"/></>,
    arrow: <><path d="M5 12h14"/><path d="m14 7 5 5-5 5"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    retry: <><path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 5v6h-6"/></>,
    close: <><path d="m6 6 12 12"/><path d="m18 6-12 12"/></>,
  }
  return (
    <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {paths[name]}
    </svg>
  )
}

function getErrorMessage(error) {
  if (error instanceof TypeError) {
    return 'Không thể kết nối tới máy chủ. Hãy kiểm tra backend đang chạy.'
  }
  return error.message || 'Đã có lỗi xảy ra. Vui lòng thử lại.'
}

async function readResponse(response) {
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = data?.detail
    throw new Error(detail?.message || detail?.reason || `Yêu cầu thất bại (${response.status})`)
  }
  return data
}

function ProductImage({ reference, name }) {
  const [failed, setFailed] = useState(false)
  const source = reference
    ? `${API_BASE}/catalog-images/${encodeURIComponent(reference)}`
    : null

  if (!source || failed) {
    return (
      <div className="product-image product-image--fallback" aria-label="Không có ảnh sản phẩm">
        <Icon name="image" size={28} />
        <span>{reference || 'No image'}</span>
      </div>
    )
  }

  return (
    <div className="product-image">
      <img src={source} alt={name || 'Sản phẩm thời trang'} loading="lazy" onError={() => setFailed(true)} />
    </div>
  )
}

function ProductCard({ item, index, mode }) {
  const finalScore = Number(item.final_score ?? 0)
  const detailScore = mode === 'text'
    ? item.text_to_image_score
    : item.visual_similarity

  return (
    <article className="product-card" style={{ '--delay': `${index * 55}ms` }}>
      <div className="product-visual">
        <ProductImage reference={item.image_reference} name={item.product_name} />
        <span className="rank">{String(index + 1).padStart(2, '0')}</span>
        <span className="score-pill">{finalScore.toFixed(3)}</span>
      </div>
      <div className="product-copy">
        <span className="category">{item.category || 'Thời trang'}</span>
        <h3>{item.product_name || 'Sản phẩm không tên'}</h3>
        <div className="score-row">
          <span>{mode === 'text' ? 'Text → image' : 'Visual match'}</span>
          <strong>{Number(detailScore ?? finalScore).toFixed(3)}</strong>
        </div>
        {mode === 'text' && item.text_to_text_score !== null && item.text_to_text_score !== undefined && (
          <div className="score-row score-row--muted">
            <span>Text → product</span>
            <strong>{Number(item.text_to_text_score).toFixed(3)}</strong>
          </div>
        )}
        {mode === 'image' && Number(item.category_bonus) > 0 && (
          <div className="bonus"><Icon name="spark" size={14} /> +{Number(item.category_bonus).toFixed(2)} cùng danh mục</div>
        )}
      </div>
    </article>
  )
}

function Results({ data, mode }) {
  if (!data) return null

  const items = data.items || []
  const elapsed = mode === 'text'
    ? data.elapsed_seconds
    : data.processing_times?.total

  return (
    <section className="results" aria-live="polite">
      <div className="results-head">
        <div>
          <span className="eyebrow">Kết quả tuyển chọn</span>
          <h2>{items.length} gợi ý gần nhất</h2>
        </div>
        <div className="result-facts">
          {mode === 'text' && data.query_used !== data.query && (
            <span className="translated">“{data.query_used}”</span>
          )}
          {mode === 'image' && data.predicted_category && (
            <span><b>{data.predicted_category}</b> · {Math.round((data.category_confidence || 0) * 100)}% tin cậy</span>
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

function TextSearch({ onResult, onError, loading, setLoading }) {
  const [query, setQuery] = useState('')
  const [language, setLanguage] = useState('vi')
  const [topK, setTopK] = useState(5)
  const [imageWeight, setImageWeight] = useState(0.5)

  async function submit(event) {
    event.preventDefault()
    if (!query.trim() || loading) return

    setLoading(true)
    onError('')
    onResult(null)
    try {
      const response = await fetch(`${API_BASE}/api/recommendations/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), language, top_k: topK, image_weight: imageWeight }),
      })
      onResult(await readResponse(response))
    } catch (error) {
      onError(getErrorMessage(error))
    } finally {
      setLoading(false)
    }
  }

  return (
    <form className="search-panel" onSubmit={submit}>
      <div className="field-label">
        <label htmlFor="fashion-query">Bạn đang tìm món đồ như thế nào?</label>
        <span>Mô tả màu sắc, kiểu dáng hoặc dịp sử dụng</span>
      </div>
      <div className="query-box">
        <textarea
          id="fashion-query"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={language === 'vi' ? 'Ví dụ: một chiếc váy mùa hè màu đỏ, nhẹ và nữ tính...' : 'Example: a minimal white sneaker for everyday wear...'}
          maxLength={512}
          rows={3}
        />
        <span className="character-count">{query.length}/512</span>
      </div>

      <div className="examples" aria-label="Gợi ý tìm kiếm">
        <span>Thử nhanh</span>
        {textExamples.map((example) => (
          <button type="button" key={example} onClick={() => { setQuery(example); setLanguage('vi') }}>
            {example}
          </button>
        ))}
      </div>

      <div className="controls-row">
        <div className="control-group">
          <span className="control-label">Ngôn ngữ</span>
          <div className="segmented">
            <button type="button" className={language === 'vi' ? 'active' : ''} onClick={() => setLanguage('vi')}>Tiếng Việt</button>
            <button type="button" className={language === 'en' ? 'active' : ''} onClick={() => setLanguage('en')}>English</button>
          </div>
        </div>

        <label className="control-group compact-control">
          <span className="control-label">Số kết quả</span>
          <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
            {[3, 5, 6, 8, 10].map((value) => <option key={value} value={value}>{value} sản phẩm</option>)}
          </select>
        </label>

        <label className="control-group weight-control">
          <span className="control-label">Thiên về hình ảnh <b>{Math.round(imageWeight * 100)}%</b></span>
          <input type="range" min="0" max="1" step="0.05" value={imageWeight} onChange={(event) => setImageWeight(Number(event.target.value))} />
        </label>
      </div>

      <button className="primary-button" type="submit" disabled={!query.trim() || loading}>
        {loading ? <span className="spinner" /> : <Icon name="search" />}
        {loading ? 'Đang tìm phong cách phù hợp...' : 'Tìm sản phẩm phù hợp'}
        {!loading && <Icon name="arrow" />}
      </button>
    </form>
  )
}

function ImageSearch({ onResult, onError, loading, setLoading }) {
  const [file, setFile] = useState(null)
  const [topK, setTopK] = useState(5)
  const [categoryMode, setCategoryMode] = useState('soft_category')
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)
  const preview = useMemo(() => file ? URL.createObjectURL(file) : '', [file])

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview)
    }
  }, [preview])

  function acceptFile(candidate) {
    if (!candidate) return
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(candidate.type)) {
      onError('Chỉ hỗ trợ ảnh JPG, PNG hoặc WebP.')
      return
    }
    if (candidate.size > MAX_IMAGE_SIZE) {
      onError('Ảnh không được vượt quá 5 MB.')
      return
    }
    onError('')
    onResult(null)
    setFile(candidate)
  }

  async function submit(event) {
    event.preventDefault()
    if (!file || loading) return

    const payload = new FormData()
    payload.append('file', file)
    payload.append('top_k', String(topK))
    payload.append('category_mode', categoryMode)

    setLoading(true)
    onError('')
    onResult(null)
    try {
      const response = await fetch(`${API_BASE}/api/recommendations/image`, { method: 'POST', body: payload })
      onResult(await readResponse(response))
    } catch (error) {
      onError(getErrorMessage(error))
    } finally {
      setLoading(false)
    }
  }

  return (
    <form className="search-panel" onSubmit={submit}>
      <div className="field-label">
        <span className="label-title">Tải lên một món đồ bạn yêu thích</span>
        <span>AI sẽ tìm các sản phẩm có phong cách tương đồng</span>
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
              <div><Icon name="check" size={18} /><span><b>{file.name}</b><small>{(file.size / 1024 / 1024).toFixed(2)} MB</small></span></div>
              <button type="button" aria-label="Xóa ảnh" onClick={() => setFile(null)}><Icon name="close" /></button>
            </div>
          </>
        ) : (
          <button type="button" className="dropzone-action" onClick={() => inputRef.current?.click()}>
            <span className="upload-icon"><Icon name="upload" size={26} /></span>
            <b>Kéo thả ảnh vào đây</b>
            <span>hoặc nhấn để chọn ảnh · JPG, PNG, WebP · tối đa 5 MB</span>
          </button>
        )}
      </div>

      <div className="mode-picker">
        <span className="control-label">Cách lọc danh mục</span>
        <div className="mode-options">
          {categoryModes.map((mode) => (
            <button type="button" key={mode.value} className={categoryMode === mode.value ? 'active' : ''} onClick={() => setCategoryMode(mode.value)}>
              <span>{mode.label}</span><small>{mode.hint}</small>
            </button>
          ))}
        </div>
      </div>

      <div className="image-actions">
        <label className="control-group compact-control">
          <span className="control-label">Số kết quả</span>
          <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
            {[3, 5, 6, 8, 10].map((value) => <option key={value} value={value}>{value} sản phẩm</option>)}
          </select>
        </label>
        <button className="primary-button" type="submit" disabled={!file || loading}>
          {loading ? <span className="spinner" /> : <Icon name="spark" />}
          {loading ? 'Đang phân tích hình ảnh...' : 'Khám phá sản phẩm tương tự'}
          {!loading && <Icon name="arrow" />}
        </button>
      </div>
    </form>
  )
}

function App() {
  const [activeTab, setActiveTab] = useState('text')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [serverStatus, setServerStatus] = useState('checking')

  async function checkServer() {
    setServerStatus('checking')
    try {
      const response = await fetch(`${API_BASE}/api/health/ready`)
      setServerStatus(response.ok ? 'ready' : 'offline')
    } catch {
      setServerStatus('offline')
    }
  }

  useEffect(() => {
    let active = true
    fetch(`${API_BASE}/api/health/ready`)
      .then((response) => {
        if (active) setServerStatus(response.ok ? 'ready' : 'offline')
      })
      .catch(() => {
        if (active) setServerStatus('offline')
      })
    return () => { active = false }
  }, [])

  function switchTab(tab) {
    setActiveTab(tab)
    setResult(null)
    setError('')
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Thread Find trang chủ">
          <span className="brand-mark">TF</span>
          <span>THREAD<span>/</span>FIND</span>
        </a>
        <div className={`status status--${serverStatus}`}>
          <span className="status-dot" />
          {serverStatus === 'ready' ? 'AI sẵn sàng' : serverStatus === 'checking' ? 'Đang kết nối' : 'Backend ngoại tuyến'}
          {serverStatus === 'offline' && <button onClick={checkServer} aria-label="Thử kết nối lại"><Icon name="retry" size={15} /></button>}
        </div>
      </header>

      <main id="top">
        <section className="hero-section">
          <div className="hero-kicker"><span>01</span> Multimodal fashion discovery</div>
          <h1>Tìm đúng phong cách.<br/><em>Theo cách của bạn.</em></h1>
          <p>Mô tả bằng lời hoặc đưa cho chúng tôi một hình ảnh. Công nghệ CLIP sẽ đọc gu thẩm mỹ và tìm ra những lựa chọn gần nhất.</p>
          <div className="hero-note"><Icon name="spark" size={16} /> Powered by fine-tuned CLIP</div>
        </section>

        <section className="finder-section">
          <div className="tabs" role="tablist" aria-label="Phương thức tìm kiếm">
            <button role="tab" aria-selected={activeTab === 'text'} className={activeTab === 'text' ? 'active' : ''} onClick={() => switchTab('text')}>
              <span>01</span><Icon name="search" />Tìm bằng mô tả
            </button>
            <button role="tab" aria-selected={activeTab === 'image'} className={activeTab === 'image' ? 'active' : ''} onClick={() => switchTab('image')}>
              <span>02</span><Icon name="image" />Tìm bằng hình ảnh
            </button>
          </div>

          {activeTab === 'text' ? (
            <TextSearch onResult={setResult} onError={setError} loading={loading} setLoading={setLoading} />
          ) : (
            <ImageSearch onResult={setResult} onError={setError} loading={loading} setLoading={setLoading} />
          )}

          {error && <div className="error-banner" role="alert"><span>!</span><div><b>Chưa thể hoàn tất tìm kiếm</b><p>{error}</p></div></div>}
        </section>

        <Results data={result} mode={activeTab} />

        {!result && !loading && (
          <section className="how-it-works">
            <div><span className="step-number">01</span><h3>Đưa ra tín hiệu</h3><p>Viết điều bạn muốn hoặc tải lên một hình ảnh tham khảo.</p></div>
            <div><span className="step-number">02</span><h3>AI đọc phong cách</h3><p>CLIP chuyển hình ảnh và ngôn ngữ vào cùng một không gian ý nghĩa.</p></div>
            <div><span className="step-number">03</span><h3>Nhận gợi ý</h3><p>Các lựa chọn phù hợp nhất được xếp hạng trong vài giây.</p></div>
          </section>
        )}
      </main>

      <footer>
        <span>THREAD/FIND © 2026</span>
        <span>Fashion recommendation · Fine-tuned CLIP</span>
      </footer>
    </div>
  )
}

export default App
