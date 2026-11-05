import { useState } from 'react'
import { useLanguage } from '../../i18n'

const COPY = {
  vi: {
    eyebrow: 'ONLINE RETRIEVAL PIPELINE',
    title: 'Ba luồng truy vấn trong cùng một hệ thống',
    subtitle: 'Chọn một loại truy vấn để xem cách hệ thống mã hóa, tính điểm và sử dụng category evidence trước khi trả về Top-K.',
    serving: ['REACT', 'NGINX', 'BACKEND ×2', 'FINE-TUNED CLIP', 'CATALOG 5K'],
    modes: {
      image: {
        tab: 'IMAGE QUERY',
        summary: 'Ảnh luôn được so với toàn bộ catalog. Category được suy ra từ láng giềng ảnh và CLIP prototype; quan hệ Top-2 chỉ điều chỉnh bonus, không lọc cứng sản phẩm.',
        steps: [
          ['01', 'CROP ẢNH', 'Chọn vùng trang phục cần tìm'],
          ['02', 'CLIP + SII', 'Mã hóa ảnh và tính cosine với catalog'],
          ['03', 'CATEGORY EVIDENCE', 'Láng giềng cân bằng + CLIP prototype'],
          ['04', 'CONFIDENCE ROUTE', 'Đánh giá h và quan hệ Top-2'],
          ['05', 'SOFT TOP-K', 'Cộng bonus có trọng số rồi xếp hạng'],
        ],
        evidenceTitle: 'Category evidence quay lại bước xếp hạng',
        evidence: [
          ['Confidence thấp', 'h ≤ h₀: không cộng category bonus; giữ nguyên cosine ranking.'],
          ['Confidence trung bình', 'Top-2 liên quan: ưu tiên cả hai với mức khác nhau. Không liên quan: chỉ cộng tín hiệu Top-1 rất nhỏ.'],
          ['Confidence cao', 'h ≥ h₁: dùng soft bonus đầy đủ cho Top-1; toàn bộ catalog vẫn được giữ lại.'],
        ],
      },
      text: {
        tab: 'TEXT QUERY',
        summary: 'Văn bản tạo cả text-to-image và text-to-text score; category chỉ lọc cứng khi truy vấn nêu rõ đúng một loại sản phẩm.',
        steps: [
          ['01', 'NORMALIZE', 'Dịch VI → EN khi cần'],
          ['02', 'CLIP TEXT', 'Tạo text embedding'],
          ['03', 'DUAL SCORES', 'Tính STI và STT'],
          ['04', 'TEXT EVIDENCE', 'Kiểm tra category và alias'],
          ['05', 'ROUTING TOP-K', 'Chọn candidate pool rồi xếp hạng'],
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
        summary: 'Ảnh giữ ngữ cảnh sản phẩm, text tinh chỉnh yêu cầu; attributes và category evidence được áp dụng sau raw cosine fusion.',
        steps: [
          ['01', 'DUAL INPUT', 'Ảnh đã crop + mô tả'],
          ['02', 'CLIP ENCODERS', 'Vision và text embeddings'],
          ['03', 'RAW FUSION', 'Kết hợp SII với ST'],
          ['04', 'ATTRIBUTES', 'Màu sắc và họa tiết A(i)'],
          ['05', 'CATEGORY TOP-K', 'Text rõ: hard · ảnh: soft'],
        ],
        evidenceTitle: 'Kết hợp category evidence từ ảnh và văn bản',
        evidence: [
          ['Text rõ ràng', 'Một category chính xác tạo candidate pool và có thể tinh chỉnh subtype phù hợp với ảnh.'],
          ['Text mơ hồ', 'Nhiều category không kích hoạt hard filter; người dùng nhận gợi ý chọn lại.'],
          ['Chỉ evidence từ ảnh', 'Dùng confidence-weighted soft bonus, không loại sản phẩm ngoài category.'],
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
        summary: 'The image is always compared with the full catalog. Category evidence combines visual neighbours and CLIP prototypes; the Top-2 relation only adjusts bonuses and never hard-filters products.',
        steps: [
          ['01', 'IMAGE CROP', 'Select the target garment'],
          ['02', 'CLIP + SII', 'Encode the image and score the catalog'],
          ['03', 'CATEGORY EVIDENCE', 'Balanced neighbours + CLIP prototype'],
          ['04', 'CONFIDENCE ROUTE', 'Evaluate h and the Top-2 relation'],
          ['05', 'SOFT TOP-K', 'Apply weighted bonuses and rank'],
        ],
        evidenceTitle: 'Category evidence feeds back into ranking',
        evidence: [
          ['Low confidence', 'h ≤ h₀: apply no category bonus and preserve cosine ranking.'],
          ['Medium confidence', 'Related Top-2 labels receive unequal bonuses; otherwise only a very weak Top-1 signal is used.'],
          ['High confidence', 'h ≥ h₁: apply the full Top-1 soft bonus while retaining the complete catalog.'],
        ],
      },
      text: {
        tab: 'TEXT QUERY',
        summary: 'Text produces both text-to-image and text-to-text scores; hard filtering is used only when exactly one product category is explicitly identified.',
        steps: [
          ['01', 'NORMALIZE', 'Translate VI → EN when needed'],
          ['02', 'CLIP TEXT', 'Create text embedding'],
          ['03', 'DUAL SCORES', 'Calculate STI and STT'],
          ['04', 'TEXT EVIDENCE', 'Check category names and aliases'],
          ['05', 'ROUTING TOP-K', 'Choose candidate pool and rank'],
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
        summary: 'The image supplies product context while text refines the request; attributes and category evidence follow raw cosine fusion.',
        steps: [
          ['01', 'DUAL INPUT', 'Cropped image + description'],
          ['02', 'CLIP ENCODERS', 'Vision and text embeddings'],
          ['03', 'RAW FUSION', 'Combine SII with ST'],
          ['04', 'ATTRIBUTES', 'Colour and pattern A(i)'],
          ['05', 'CATEGORY TOP-K', 'Explicit text: hard · image: soft'],
        ],
        evidenceTitle: 'Combine image and text category evidence',
        evidence: [
          ['Explicit text', 'One exact category creates the candidate pool and may refine a compatible image subtype.'],
          ['Ambiguous text', 'Multiple categories do not trigger a hard filter; the interface asks for a choice.'],
          ['Image evidence only', 'Use a confidence-weighted soft bonus without removing products outside the category.'],
        ],
      },
    },
  },
}

function Formula({ mode }) {
  if (mode === 'image') {
    return <><i>S</i><sub>image</sub>(i) = <i>S</i><sub>II</sub>(i) + λ<sub>I</sub> · g(h) · w(c<sub>i</sub>)</>
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
