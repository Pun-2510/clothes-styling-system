import { useState } from 'react'
import { useLanguage } from '../../i18n'

const COPY = {
  vi: {
    eyebrow: 'ONLINE RETRIEVAL PIPELINE',
    title: 'Ba luồng truy vấn trong cùng một hệ thống',
    subtitle: 'Chọn một loại truy vấn để xem cách hệ thống mã hóa, tính điểm và áp dụng category evidence trước khi trả về Top-K.',
    serving: ['REACT', 'NGINX', 'BACKEND ×2', 'FINE-TUNED CLIP', 'CATALOG 5K'],
    modes: {
      image: {
        tab: 'IMAGE QUERY',
        summary: 'Ảnh được so sánh với catalog trước; category dự đoán chỉ bổ sung điểm theo độ tin cậy và không loại sản phẩm khác.',
        steps: [
          ['01', 'CROP ẢNH', 'Chọn đúng vùng trang phục'],
          ['02', 'CLIP VISION', 'Tạo image embedding'],
          ['03', 'COSINE SII', 'So với 5.000 ảnh catalog'],
          ['04', 'CATEGORY', 'Láng giềng cân bằng → r₁, r₂'],
          ['05', 'WEIGHTED TOP-K', 'g(h) điều khiển category bonus'],
        ],
        evidenceTitle: 'Category prediction quay lại ranking',
        evidence: [
          ['Dự đoán', 'Mỗi category đóng góp cùng số láng giềng mạnh nhất.'],
          ['Confidence', 'h = r₁ / (r₁ + r₂), sau đó chuẩn hóa thành g(h).'],
          ['Áp dụng lại', 'Cộng λI · g(h) · C(i) vào image score trước khi xếp hạng.'],
        ],
      },
      text: {
        tab: 'TEXT QUERY',
        summary: 'Văn bản tạo cả text-to-image và text-to-text score; category chỉ lọc cứng khi câu nêu đúng một loại sản phẩm.',
        steps: [
          ['01', 'NORMALIZE', 'Dịch VI → EN khi cần'],
          ['02', 'CLIP TEXT', 'Tạo text embedding'],
          ['03', 'DUAL SCORES', 'STI và STT'],
          ['04', 'TEXT EVIDENCE', 'Tên category + alias chính xác'],
          ['05', 'ROUTING TOP-K', 'Chọn candidate pool rồi xếp hạng'],
        ],
        evidenceTitle: 'Ba nhánh xử lý category từ text',
        evidence: [
          ['Đúng 1 category', 'Hard filter category rồi xếp hạng bằng text score.'],
          ['Không có category', 'Giữ toàn bộ catalog, không áp đặt category.'],
          ['Nhiều loại / sai từ', 'Không hard filter; giao diện yêu cầu chọn hoặc sửa truy vấn.'],
        ],
      },
      combined: {
        tab: 'IMAGE + TEXT',
        summary: 'Ảnh giữ ngữ cảnh sản phẩm, text tinh chỉnh yêu cầu; attributes và category evidence được áp dụng sau raw cosine fusion.',
        steps: [
          ['01', 'DUAL INPUT', 'Ảnh đã crop + mô tả'],
          ['02', 'CLIP ENCODERS', 'Vision và text embeddings'],
          ['03', 'RAW FUSION', 'SII kết hợp ST'],
          ['04', 'ATTRIBUTES', 'Màu sắc và họa tiết A(i)'],
          ['05', 'CATEGORY TOP-K', 'Text hard hoặc image soft'],
        ],
        evidenceTitle: 'Kết hợp hai nguồn category evidence',
        evidence: [
          ['Text rõ ràng', 'Một category chính xác tạo candidate pool; có thể tinh chỉnh subtype phù hợp với ảnh.'],
          ['Text mơ hồ', 'Nhiều category không được hard filter; người dùng nhận gợi ý chọn lại.'],
          ['Chỉ evidence từ ảnh', 'Dùng confidence-weighted bonus, không loại sản phẩm ngoài category.'],
        ],
      },
    },
  },
  en: {
    eyebrow: 'ONLINE RETRIEVAL PIPELINE',
    title: 'Three query paths in one retrieval system',
    subtitle: 'Select a query type to inspect how the system encodes, scores, and applies category evidence before returning Top-K.',
    serving: ['REACT', 'NGINX', 'BACKEND ×2', 'FINE-TUNED CLIP', 'CATALOG 5K'],
    modes: {
      image: {
        tab: 'IMAGE QUERY',
        summary: 'The image is compared with the catalog first; the predicted category only adds a confidence-weighted bonus and never removes other products.',
        steps: [
          ['01', 'IMAGE CROP', 'Select the target garment'],
          ['02', 'CLIP VISION', 'Create image embedding'],
          ['03', 'COSINE SII', 'Compare with 5,000 catalog images'],
          ['04', 'CATEGORY', 'Balanced neighbours → r₁, r₂'],
          ['05', 'WEIGHTED TOP-K', 'g(h) controls category bonus'],
        ],
        evidenceTitle: 'Category prediction feeds back into ranking',
        evidence: [
          ['Prediction', 'Every category contributes the same number of strongest neighbours.'],
          ['Confidence', 'h = r₁ / (r₁ + r₂), then mapped to the strength g(h).'],
          ['Feedback', 'Add λI · g(h) · C(i) to the image score before ranking.'],
        ],
      },
      text: {
        tab: 'TEXT QUERY',
        summary: 'Text produces both text-to-image and text-to-text scores; hard filtering is used only when exactly one product category is named.',
        steps: [
          ['01', 'NORMALIZE', 'Translate VI → EN when needed'],
          ['02', 'CLIP TEXT', 'Create text embedding'],
          ['03', 'DUAL SCORES', 'STI and STT'],
          ['04', 'TEXT EVIDENCE', 'Exact category name + aliases'],
          ['05', 'ROUTING TOP-K', 'Choose candidate pool and rank'],
        ],
        evidenceTitle: 'Three text category routes',
        evidence: [
          ['Exactly one category', 'Hard-filter that category, then rank with the text score.'],
          ['No category', 'Keep the complete catalog without imposing a category.'],
          ['Multiple / misspelled', 'Do not hard-filter; ask the user to select or correct the query.'],
        ],
      },
      combined: {
        tab: 'IMAGE + TEXT',
        summary: 'The image supplies product context while text refines the request; attributes and category evidence follow raw cosine fusion.',
        steps: [
          ['01', 'DUAL INPUT', 'Cropped image + description'],
          ['02', 'CLIP ENCODERS', 'Vision and text embeddings'],
          ['03', 'RAW FUSION', 'Combine SII with ST'],
          ['04', 'ATTRIBUTES', 'Colour and pattern A(i)'],
          ['05', 'CATEGORY TOP-K', 'Text hard or image soft'],
        ],
        evidenceTitle: 'Combine two category evidence sources',
        evidence: [
          ['Explicit text', 'One exact category creates the candidate pool and may use a compatible image subtype.'],
          ['Ambiguous text', 'Multiple categories do not trigger a hard filter; the interface asks for a choice.'],
          ['Image evidence only', 'Use a confidence-weighted bonus without removing products outside the category.'],
        ],
      },
    },
  },
}

function Formula({ mode }) {
  if (mode === 'image') {
    return <><i>S</i><sub>image</sub>(i) = <i>S</i><sub>II</sub>(i) + λ<sub>I</sub> · g(h) · C(i)</>
  }
  if (mode === 'text') {
    return <><i>S</i><sub>T</sub>(i) = β<i>S</i><sub>TI</sub>(i) + (1 − β)<i>S</i><sub>TT</sub>(i)</>
  }
  return <><i>S</i><sub>base</sub>(i) = ω<sub>I</sub><i>S</i><sub>II</sub>(i) + (1 − ω<sub>I</sub>)<i>S</i><sub>T</sub>(i)</>
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
