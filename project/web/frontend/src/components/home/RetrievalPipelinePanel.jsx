import { useState } from 'react'
import { useLanguage } from '../../i18n'

const COPY = {
  vi: {
    eyebrow: 'ONLINE RETRIEVAL PIPELINE',
    title: 'Ba luồng truy vấn trong cùng một hệ thống',
    subtitle: 'Chọn một loại truy vấn để xem cách hệ thống mã hóa, tính điểm và sử dụng category evidence trước khi trả về Top-K.',
    serving: ['REACT', 'NGINX', 'BACKEND ×2', 'FINE-TUNED CLIP', 'CATALOG INDEX'],
    modes: {
      image: {
        tab: 'IMAGE QUERY',
        summary: 'Ảnh được mã hóa bằng CLIP và so với catalog. Category evidence dự đoán danh mục; hệ thống chỉ xếp hạng sản phẩm trong danh mục đó.',
        steps: [
          ['01', 'CHỌN VÙNG ẢNH', 'Chọn trang phục cần tìm'],
          ['02', 'CLIP VISION', 'Tạo image embedding'],
          ['03', 'COSINE SII', 'So sánh ảnh với catalog'],
          ['04', 'DỰ ĐOÁN DANH MỤC', 'Láng giềng cân bằng và CLIP prompt'],
          ['05', 'HARD CATEGORY + TOP-K', 'Chỉ xếp hạng trong danh mục dự đoán'],
        ],
        evidenceTitle: 'Cách dùng category evidence',
        evidence: [
          ['Dự đoán danh mục', 'Kết hợp láng giềng ảnh và CLIP prompt để chọn một danh mục.'],
          ['Lọc theo danh mục', 'Chỉ sản phẩm thuộc danh mục được chọn mới vào Top-K.'],
          ['Nếu kết quả chưa đúng', 'Thêm mô tả loại sản phẩm hoặc chọn lại vùng ảnh để tìm kết hợp.'],
        ],
      },
      text: {
        tab: 'TEXT QUERY',
        summary: 'Văn bản tạo text-to-image và text-to-text score; TF-IDF bổ sung mức khớp với tên và mô tả catalog. Category chỉ giới hạn ứng viên khi truy vấn nêu rõ đúng một loại sản phẩm.',
        steps: [
          ['01', 'NORMALIZE', 'Dịch VI → EN khi cần'],
          ['02', 'CLIP TEXT', 'Tạo text embedding'],
          ['03', 'DUAL SCORES', 'Tính STI và STT'],
          ['04', 'TF-IDF METADATA', 'Khớp từ khóa với tên và mô tả'],
          ['05', 'CATEGORY + TOP-K', 'Chọn candidate pool rồi xếp hạng'],
        ],
        evidenceTitle: 'Ba nhánh xử lý category từ text',
        evidence: [
          ['Đúng một category', 'Giới hạn candidate pool theo category rồi xếp hạng bằng text score.'],
          ['Không có category', 'Giữ toàn bộ catalog và xếp hạng bằng semantic similarity.'],
          ['Nhiều loại / sai từ', 'Không áp dụng hard filter; giao diện yêu cầu chọn hoặc sửa truy vấn.'],
        ],
      },
      combined: {
        tab: 'IMAGE + TEXT',
        summary: 'Ảnh và văn bản cùng tạo điểm CLIP. Nếu văn bản nêu rõ một danh mục, dùng danh mục đó; nếu không, dùng danh mục ảnh dự đoán. TF-IDF hỗ trợ khớp thuộc tính trong danh mục đã chọn.',
        steps: [
          ['01', 'ẢNH + MÔ TẢ', 'Ảnh đã chọn vùng và văn bản'],
          ['02', 'CLIP ENCODERS', 'Tạo image và text embeddings'],
          ['03', 'RAW FUSION', 'Kết hợp điểm ảnh và văn bản'],
          ['04', 'SELECT CATEGORY', 'Ưu tiên danh mục rõ trong text, còn lại dùng ảnh'],
          ['05', 'TF-IDF + FILTER + TOP-K', 'Tinh chỉnh và xếp hạng trong danh mục'],
        ],
        evidenceTitle: 'Chọn danh mục cho truy vấn kết hợp',
        evidence: [
          ['Text nêu rõ một danh mục', 'Dùng đúng danh mục người dùng yêu cầu.'],
          ['Text không nêu danh mục', 'Dùng danh mục được dự đoán từ ảnh.'],
          ['Text nêu nhiều danh mục', 'Dùng danh mục ảnh dự đoán và gợi ý người dùng làm rõ.'],
        ],
      },
    },
  },
  en: {
    eyebrow: 'ONLINE RETRIEVAL PIPELINE',
    title: 'Three query paths in one retrieval system',
    subtitle: 'Select a query type to inspect how the system encodes, scores, and applies category evidence before returning Top-K.',
    serving: ['REACT', 'NGINX', 'BACKEND ×2', 'FINE-TUNED CLIP', 'CATALOG INDEX'],
    modes: {
      image: {
        tab: 'IMAGE QUERY',
        summary: 'CLIP encodes the image and compares it with the catalog. Category evidence predicts one category; only products in that category are ranked.',
        steps: [
          ['01', 'SELECT IMAGE REGION', 'Choose the target garment'],
          ['02', 'CLIP VISION', 'Create an image embedding'],
          ['03', 'COSINE SII', 'Compare with catalog images'],
          ['04', 'PREDICT CATEGORY', 'Balanced neighbours and CLIP prompts'],
          ['05', 'HARD CATEGORY + TOP-K', 'Rank only the predicted category'],
        ],
        evidenceTitle: 'How category evidence is used',
        evidence: [
          ['Predict category', 'Combine visual neighbours and CLIP prompts to select one category.'],
          ['Restrict results', 'Only products in the selected category may enter Top-K.'],
          ['If results are wrong', 'Add a product description or select a different image region.'],
        ],
      },
      text: {
        tab: 'TEXT QUERY',
        summary: 'Text produces text-to-image and text-to-text scores; TF-IDF adds lexical matching against catalog names and descriptions. Candidate restriction is used only when exactly one category is explicit.',
        steps: [
          ['01', 'NORMALIZE', 'Translate VI → EN when needed'],
          ['02', 'CLIP TEXT', 'Create text embedding'],
          ['03', 'DUAL SCORES', 'Calculate STI and STT'],
          ['04', 'TF-IDF METADATA', 'Match catalog names and descriptions'],
          ['05', 'CATEGORY + TOP-K', 'Choose candidate pool and rank'],
        ],
        evidenceTitle: 'Three text category routes',
        evidence: [
          ['Exactly one category', 'Restrict the candidate pool, then rank it with the text score.'],
          ['No category', 'Keep the complete catalog and rank by semantic similarity.'],
          ['Multiple / misspelled', 'Do not hard-filter; ask the user to select or correct the query.'],
        ],
      },
      combined: {
        tab: 'IMAGE + TEXT',
        summary: 'Image and text contribute CLIP scores. One explicit text category takes precedence; otherwise the image prediction selects the category. TF-IDF refines attributes within that category.',
        steps: [
          ['01', 'IMAGE + DESCRIPTION', 'Selected image region and text'],
          ['02', 'CLIP ENCODERS', 'Create image and text embeddings'],
          ['03', 'RAW FUSION', 'Combine image and text scores'],
          ['04', 'SELECT CATEGORY', 'Explicit text category or image prediction'],
          ['05', 'TF-IDF + FILTER + TOP-K', 'Refine and rank within the category'],
        ],
        evidenceTitle: 'Select a category for combined retrieval',
        evidence: [
          ['One explicit text category', 'Use the category requested by the user.'],
          ['No text category', 'Use the predicted image category.'],
          ['Multiple text categories', 'Use the image prediction and ask the user to clarify.'],
        ],
      },
    },
  },
}

function Formula({ mode }) {
  if (mode === 'image') {
    return <><i>S</i><sub>final</sub>(i) = <i>S</i><sub>II</sub>(i), c<sub>i</sub> = ĉ</>
  }
  if (mode === 'text') {
    return <><i>S</i><sub>text</sub>(i) = (1 − λ<sub>M</sub>)<i>S</i><sub>T</sub>(i) + λ<sub>M</sub><i>S</i><sub>M</sub>(i)</>
  }
  return <><i>S</i><sub>final</sub>(i) = <i>S</i><sub>base</sub>(i) + λ<sub>M</sub><i>S</i><sub>M</sub>(i), c<sub>i</sub> = ĉ</>
}

export default function RetrievalPipelinePanel() {
  const { language } = useLanguage()
  const [mode, setMode] = useState('combined')
  const copy = COPY[language] || COPY.en
  const selected = copy.modes[mode]

  return (
    <section className={`hero-pipeline-visual pipeline-mode--${mode}`} aria-labelledby="pipeline-title">
      <div className="pipeline-visual-head">
        <div>
          <span>{copy.eyebrow}</span>
          <h2 id="pipeline-title">{copy.title}</h2>
        </div>
        <small>01 — 05</small>
      </div>

      <p className="pipeline-introduction">{copy.subtitle}</p>

      <div className="pipeline-mode-tabs" role="tablist" aria-label={copy.title}>
        {Object.entries(copy.modes).map(([key, item]) => (
          <button
            key={key}
            type="button"
            role="tab"
            aria-selected={mode === key}
            className={mode === key ? 'active' : ''}
            onClick={() => setMode(key)}
          >
            <span>{key === 'image' ? 'I' : key === 'text' ? 'T' : 'I+T'}</span>
            {item.tab}
          </button>
        ))}
      </div>

      <div className="pipeline-mode-content" role="tabpanel">
        <div className="pipeline-mode-summary">
          <p>{selected.summary}</p>
          <code><Formula mode={mode} /></code>
        </div>

        <div className="pipeline-flow">
          {selected.steps.map(([number, title, description], index) => (
            <div className="pipeline-step-wrap" key={`${mode}-${number}`}>
              <article className={`pipeline-stage ${index === selected.steps.length - 1 ? 'pipeline-stage--result' : ''}`}>
                <b>{number}</b>
                <strong>{title}</strong>
                <small>{description}</small>
              </article>
              {index < selected.steps.length - 1 && <span className="pipeline-connector" aria-hidden="true">→</span>}
            </div>
          ))}
        </div>

        <div className="pipeline-evidence">
          <div className="pipeline-evidence-heading">
            <span>DECISION LOGIC</span>
            <strong>{selected.evidenceTitle}</strong>
          </div>
          <div className="pipeline-evidence-grid">
            {selected.evidence.map(([title, description], index) => (
              <article key={`${mode}-evidence-${title}`}>
                <b>{String(index + 1).padStart(2, '0')}</b>
                <div><strong>{title}</strong><p>{description}</p></div>
              </article>
            ))}
          </div>
        </div>
      </div>

      <div className="pipeline-serving">
        {copy.serving.map((item, index) => (
          <span key={item}>{item}{index < copy.serving.length - 1 && <i>→</i>}</span>
        ))}
      </div>
    </section>
  )
}
