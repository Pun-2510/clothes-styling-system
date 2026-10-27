import { useCallback, useEffect, useMemo, useState } from 'react'
import Icon from '../components/common/Icon'
import RankingEquation from '../components/developer/RankingEquation'
import { getDeveloperSnapshot } from '../services/developerApi'
import { useLanguage } from '../i18n'

const COPY = {
  vi: {
    breadcrumb: 'Developer', eyebrow: 'DEVELOPER WORKSPACE', title: 'Bảng điều khiển hệ thống',
    description: 'Theo dõi runtime, kiểm tra artifact và xem chính xác pipeline truy xuất đang được website sử dụng.',
    refresh: 'Làm mới trạng thái', refreshing: 'Đang kiểm tra...', updated: 'Cập nhật',
    overview: 'Tổng quan', health: 'System Health', backend: 'Backend hiện tại', pipeline: 'Quy trình hệ thống', formulas: 'Công thức xếp hạng', commands: 'Quy trình khởi chạy',
    healthy: 'Hoạt động', degraded: 'Chưa sẵn sàng', unavailable: 'Không kết nối', waiting: 'Đang kiểm tra',
    loadBalancer: 'Nginx load balancer', model: 'Fine-tuned CLIP', imageEmbeddings: 'Image embeddings', textEmbeddings: 'Text embeddings', translator: 'Bộ dịch VI → EN',
    activeInstance: 'Instance đang phản hồi', observedUpstream: 'Upstream Nginx', device: 'Thiết bị suy luận', products: 'Sản phẩm catalog', apiVersion: 'Phiên bản API', activeModel: 'Checkpoint đang dùng',
    topology: 'Kiến trúc phục vụ', topologyText: 'Trình duyệt → React → Nginx (least_conn) → backend / backend_2 → fine-tuned CLIP và catalog embeddings',
    offline: 'Offline pipeline', online: 'Online retrieval pipeline',
    offlineSteps: ['Chuẩn bị catalog 5.000 sản phẩm và audit category.', 'Chia train/validation/test và fine-tune CLIP bằng category-balanced sampling.', 'Chọn checkpoint tốt nhất trên validation set.', 'Sinh image/text embeddings từ cùng checkpoint và catalog.', 'Chọn trọng số retrieval trên validation set rồi khởi động Docker.'],
    onlineSteps: ['Tự nhận biết truy vấn text-only, image-only hoặc combined.', 'Mã hóa query bằng fine-tuned CLIP; tiếng Việt được dịch sang tiếng Anh trước khi encode.', 'Tính raw cosine similarity với toàn bộ catalog embeddings.', 'Kết hợp text, image, attribute và category theo loại truy vấn.', 'Xếp hạng Top-K và trả kết quả qua Nginx load balancer.'],
    scoreLegend: 'Các score đều bắt đầu từ cosine similarity; percentile không được sử dụng trong pipeline hiện tại.',
    cosine: 'Cosine similarity', imageOnly: 'Image-only', textOnly: 'Text-only', combined: 'Image + text', categoryConfidence: 'Category confidence',
    cosineNote: 'Độ tương đồng giữa hai embedding đã chuẩn hóa; giá trị cao hơn nghĩa là hai biểu diễn gần nhau hơn.',
    imageOnlyNote: 'Ảnh được so với image embeddings. Category dự đoán chỉ cộng bonus theo confidence, không lọc cứng catalog.',
    textOnlyNote: 'Text embedding được so với cả image embeddings và text embeddings, sau đó trộn bằng β.',
    combinedNote: 'Ảnh giữ ngữ cảnh sản phẩm; văn bản tinh chỉnh yêu cầu. Attribute và category được áp dụng sau raw cosine fusion.',
    categoryNote: 'g(h) đưa confidence về khoảng 0–1 giữa floor và threshold. Category nêu rõ trong text sẽ giới hạn candidate pool.',
    symbols: 'Ký hiệu', symbolText: 'SII: image→image · STI: text→image · STT: text→text · A: attribute match · C: category match.',
    liveWeights: 'Trọng số runtime', textImageWeight: 'β · text→image', combinedImageWeight: 'ωI · image', attributeWeight: 'ωA · attribute', categoryWeight: 'λC · combined category', imageCategoryBonus: 'λI · image category', confidenceRange: 'Khoảng confidence', topPerClass: 'Top láng giềng/category',
    quickStart: 'Các lệnh chính', prepare: '1. Chuẩn bị dữ liệu và audit', train: '2. Fine-tune CLIP', embeddings: '3. Sinh embeddings', serve: '4. Khởi động website',
    noRuntime: 'Chưa nhận được dữ liệu runtime từ backend.', localOnly: 'Trang dành cho quan sát kỹ thuật; không thay đổi model hoặc dữ liệu.',
  },
  en: {
    breadcrumb: 'Developer', eyebrow: 'DEVELOPER WORKSPACE', title: 'System dashboard',
    description: 'Monitor the runtime, inspect artifacts, and review the exact retrieval pipeline used by the website.',
    refresh: 'Refresh status', refreshing: 'Checking...', updated: 'Updated',
    overview: 'Overview', health: 'System Health', backend: 'Current backend', pipeline: 'System pipeline', formulas: 'Ranking formulas', commands: 'Run workflow',
    healthy: 'Healthy', degraded: 'Not ready', unavailable: 'Unavailable', waiting: 'Checking',
    loadBalancer: 'Nginx load balancer', model: 'Fine-tuned CLIP', imageEmbeddings: 'Image embeddings', textEmbeddings: 'Text embeddings', translator: 'VI → EN translator',
    activeInstance: 'Responding instance', observedUpstream: 'Nginx upstream', device: 'Inference device', products: 'Catalog products', apiVersion: 'API version', activeModel: 'Active checkpoint',
    topology: 'Serving architecture', topologyText: 'Browser → React → Nginx (least_conn) → backend / backend_2 → fine-tuned CLIP and catalog embeddings',
    offline: 'Offline pipeline', online: 'Online retrieval pipeline',
    offlineSteps: ['Prepare a 5,000-product catalog and audit category distribution.', 'Create train/validation/test splits and fine-tune CLIP with category-balanced sampling.', 'Select the best checkpoint on the validation set.', 'Generate image/text embeddings with the same checkpoint and catalog.', 'Select retrieval weights on validation data and start Docker.'],
    onlineSteps: ['Detect text-only, image-only, or combined input automatically.', 'Encode the query with fine-tuned CLIP; Vietnamese text is translated before encoding.', 'Calculate raw cosine similarity against all catalog embeddings.', 'Combine text, image, attribute, and category signals for the detected query type.', 'Rank Top-K products and return them through the Nginx load balancer.'],
    scoreLegend: 'Every score starts from cosine similarity; percentile normalization is not used by the current pipeline.',
    cosine: 'Cosine similarity', imageOnly: 'Image-only', textOnly: 'Text-only', combined: 'Image + text', categoryConfidence: 'Category confidence',
    cosineNote: 'Similarity between two normalized embeddings; a higher value means the representations are closer.',
    imageOnlyNote: 'The query image is compared with image embeddings. The predicted category adds a confidence-weighted bonus without hard filtering.',
    textOnlyNote: 'The text embedding is compared with both image and text embeddings, then mixed with β.',
    combinedNote: 'The image supplies product context while text refines the request. Attribute and category signals follow raw cosine fusion.',
    categoryNote: 'g(h) maps confidence to 0–1 between the floor and threshold. A category explicitly stated in text restricts the candidate pool.',
    symbols: 'Symbols', symbolText: 'SII: image→image · STI: text→image · STT: text→text · A: attribute match · C: category match.',
    liveWeights: 'Runtime weights', textImageWeight: 'β · text→image', combinedImageWeight: 'ωI · image', attributeWeight: 'ωA · attribute', categoryWeight: 'λC · combined category', imageCategoryBonus: 'λI · image category', confidenceRange: 'Confidence range', topPerClass: 'Top neighbours/category',
    quickStart: 'Main commands', prepare: '1. Prepare data and audit', train: '2. Fine-tune CLIP', embeddings: '3. Generate embeddings', serve: '4. Start the website',
    noRuntime: 'Runtime data has not been received from the backend.', localOnly: 'This page is read-only and does not modify the model or data.',
  },
}

const COMMANDS = {
  prepare: `.\\.venv\\Scripts\\python.exe -m src.prepare_dataset \`\n  --csv data/data.csv --image-dir data/data \`\n  --num-products 5000 --min-per-category 10 --seed 42 \`\n  --output data/processed/products.csv`,
  train: `.\\.venv\\Scripts\\python.exe -m src.finetune_clip \`\n  --csv data/processed/products.csv \`\n  --run-dir runs/clip_finetune_3epochs --trainable full \`\n  --epochs 3 --batch-size 16 --learning-rate 1e-6 --seed 42`,
  embeddings: `.\\.venv\\Scripts\\python.exe -m src.generate_clip_embeddings \`\n  --csv data/processed/products.csv \`\n  --model runs/clip_finetune_3epochs/best \`\n  --output-dir embeddings --batch-size 16`,
  serve: 'docker compose up -d --build',
}

function StatusBadge({ ready, copy }) {
  const state = ready === null ? 'waiting' : ready ? 'healthy' : 'degraded'
  return <span className={`dev-status dev-status--${state}`}><i />{copy[state]}</span>
}

function HealthCard({ label, ready, detail, copy }) {
  return (
    <article className="dev-health-card">
      <div className="dev-health-card-top"><span>{label}</span><StatusBadge ready={ready} copy={copy} /></div>
      <strong>{detail || '—'}</strong>
    </article>
  )
}

function PipelinePanel({ title, steps, tone }) {
  return (
    <article className={`dev-pipeline dev-pipeline--${tone}`}>
      <h3>{title}</h3>
      <ol>{steps.map((step, index) => <li key={step}><span>{String(index + 1).padStart(2, '0')}</span><p>{step}</p></li>)}</ol>
    </article>
  )
}

function FormulaCard({ title, type, note, wide = false }) {
  return (
    <article className={`dev-formula-card${wide ? ' dev-formula-card--wide' : ''}`}>
      <span>{title}</span>
      <div className={`dev-equation dev-equation--${type}`}><RankingEquation type={type} /></div>
      <p>{note}</p>
    </article>
  )
}

export default function DeveloperPage() {
  const { language, t } = useLanguage()
  const copy = COPY[language]
  const [snapshot, setSnapshot] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const refresh = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      setSnapshot(await getDeveloperSnapshot())
    } catch (failure) {
      setError(failure.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    const timer = window.setInterval(refresh, 15000)
    return () => window.clearInterval(timer)
  }, [refresh])

  const backend = snapshot?.backend
  const retrieval = backend?.retrieval
  const locale = language === 'en' ? 'en-US' : 'vi-VN'
  const updatedAt = snapshot?.checkedAt
    ? new Intl.DateTimeFormat(locale, { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(new Date(snapshot.checkedAt))
    : '—'
  const formatWeight = (value) => value == null ? '—' : Number(value).toFixed(2)

  const weights = useMemo(() => [
    [copy.textImageWeight, formatWeight(retrieval?.text_retrieval_image_weight)],
    [copy.combinedImageWeight, formatWeight(retrieval?.combined_image_weight)],
    [copy.attributeWeight, formatWeight(retrieval?.combined_attribute_weight)],
    [copy.categoryWeight, formatWeight(retrieval?.combined_category_weight)],
    [copy.imageCategoryBonus, formatWeight(retrieval?.image_category_bonus)],
    [copy.confidenceRange, retrieval ? `${formatWeight(retrieval.category_confidence_floor)} → ${formatWeight(retrieval.category_confidence_threshold)}` : '—'],
    [copy.topPerClass, retrieval?.category_top_per_class ?? '—'],
  ], [copy, retrieval])

  return (
    <section className="developer-page">
      <nav className="breadcrumbs"><a href="/">{t('categories.home')}</a><span>/</span><span>{copy.breadcrumb}</span></nav>

      <header className="developer-hero">
        <div><span className="eyebrow">{copy.eyebrow}</span><h1>{copy.title}</h1><p>{copy.description}</p></div>
        <div className="developer-hero-actions">
          <span>{copy.updated}: {updatedAt}</span>
          <button type="button" onClick={refresh} disabled={loading}><Icon name="retry" size={16} />{loading ? copy.refreshing : copy.refresh}</button>
        </div>
      </header>

      <div className="developer-layout">
        <aside className="developer-nav" aria-label="Developer sections">
          <a href="#overview">01 <span>{copy.overview}</span></a>
          <a href="#health">02 <span>{copy.health}</span></a>
          <a href="#pipeline">03 <span>{copy.pipeline}</span></a>
          <a href="#formulas">04 <span>{copy.formulas}</span></a>
          <a href="#commands">05 <span>{copy.commands}</span></a>
        </aside>

        <div className="developer-content">
          {error && <div className="dev-alert">{error}</div>}

          <section id="overview" className="dev-section">
            <div className="dev-section-heading"><span>01</span><div><small>{copy.overview}</small><h2>{copy.topology}</h2></div></div>
            <div className="dev-topology"><span>Browser</span><b>→</b><span>React :5173</span><b>→</b><span>Nginx :8000</span><b>→</b><span>backend × 2</span><b>→</b><span>CLIP</span></div>
            <p className="dev-section-note">{copy.topologyText}</p>
          </section>

          <section id="health" className="dev-section">
            <div className="dev-section-heading"><span>02</span><div><small>LIVE TELEMETRY</small><h2>{copy.health}</h2></div></div>
            <div className="dev-health-grid">
              <HealthCard label={copy.loadBalancer} ready={snapshot ? snapshot.loadBalancer.ready : null} detail="least_conn · rate limit" copy={copy} />
              <HealthCard label={copy.model} ready={snapshot ? Boolean(backend?.model_ready) : null} detail={backend?.status || copy.waiting} copy={copy} />
              <HealthCard label={copy.imageEmbeddings} ready={snapshot ? Boolean(backend?.image_embeddings_ready) : null} detail="image_embeddings.npy" copy={copy} />
              <HealthCard label={copy.textEmbeddings} ready={snapshot ? Boolean(backend?.text_embeddings_ready) : null} detail="text_embeddings.npy" copy={copy} />
              <HealthCard label={copy.translator} ready={snapshot ? Boolean(backend?.translator_ready) : null} detail="Helsinki-NLP/opus-mt-vi-en" copy={copy} />
            </div>

            <div className="dev-runtime-grid">
              <article className="dev-runtime-card">
                <div className="dev-card-title"><span className="eyebrow">RUNTIME</span><StatusBadge ready={snapshot ? backend?.status === 'ready' : null} copy={copy} /></div>
                {backend ? <dl>
                  <div><dt>{copy.activeInstance}</dt><dd>{backend.instance}</dd></div>
                  <div><dt>{copy.observedUpstream}</dt><dd>{snapshot.upstream || '—'}</dd></div>
                  <div><dt>{copy.device}</dt><dd>{backend.device || '—'}</dd></div>
                  <div><dt>{copy.products}</dt><dd>{Number(backend.products || 0).toLocaleString(locale)}</dd></div>
                  <div><dt>{copy.apiVersion}</dt><dd>{backend.api_version}</dd></div>
                  <div><dt>{copy.activeModel}</dt><dd className="dev-path">{backend.active_model}</dd></div>
                </dl> : <p>{copy.noRuntime}</p>}
              </article>
              <article className="dev-runtime-card">
                <div className="dev-card-title"><span className="eyebrow">CONFIG</span><strong>{copy.liveWeights}</strong></div>
                <div className="dev-weight-list">{weights.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
              </article>
            </div>
          </section>

          <section id="pipeline" className="dev-section">
            <div className="dev-section-heading"><span>03</span><div><small>DATA → MODEL → SERVING</small><h2>{copy.pipeline}</h2></div></div>
            <div className="dev-pipeline-grid">
              <PipelinePanel title={copy.offline} steps={copy.offlineSteps} tone="offline" />
              <PipelinePanel title={copy.online} steps={copy.onlineSteps} tone="online" />
            </div>
          </section>

          <section id="formulas" className="dev-section">
            <div className="dev-section-heading"><span>04</span><div><small>RAW COSINE FUSION</small><h2>{copy.formulas}</h2></div></div>
            <p className="dev-section-note">{copy.scoreLegend}</p>
            <div className="dev-formula-grid">
              <FormulaCard title={copy.cosine} type="cosine" note={copy.cosineNote} />
              <FormulaCard title={copy.imageOnly} type="image" note={copy.imageOnlyNote} />
              <FormulaCard title={copy.textOnly} type="text" note={copy.textOnlyNote} />
              <FormulaCard title={copy.combined} type="combined" note={copy.combinedNote} />
              <FormulaCard title={copy.categoryConfidence} type="category" note={copy.categoryNote} wide />
            </div>
            <div className="dev-symbols"><strong>{copy.symbols}</strong><span>{copy.symbolText}</span></div>
          </section>

          <section id="commands" className="dev-section">
            <div className="dev-section-heading"><span>05</span><div><small>POWERSHELL</small><h2>{copy.quickStart}</h2></div></div>
            <div className="dev-command-list">
              {Object.entries(COMMANDS).map(([key, command]) => <article key={key}><h3>{copy[key]}</h3><pre><code>{command}</code></pre></article>)}
            </div>
            <p className="dev-readonly"><Icon name="lock" size={14} />{copy.localOnly}</p>
          </section>
        </div>
      </div>
    </section>
  )
}
