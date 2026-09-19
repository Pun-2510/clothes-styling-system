const STEPS = [
  ['01', 'Đưa ra gợi ý', 'Nhập mô tả hoặc tải lên một ảnh tham khảo.'],
  ['02', 'AI phân tích', 'Hệ thống nhận diện đặc điểm và phong cách phù hợp.'],
  ['03', 'Xem kết quả', 'Các sản phẩm gần nhất được xếp hạng cho bạn.'],
]

export default function HowItWorks() {
  return (
    <section className="how-it-works" aria-label="Cách hệ thống hoạt động">
      {STEPS.map(([number, title, description]) => (
        <article key={number}>
          <span>{number}</span>
          <div>
            <h3>{title}</h3>
            <p>{description}</p>
          </div>
        </article>
      ))}
    </section>
  )
}
