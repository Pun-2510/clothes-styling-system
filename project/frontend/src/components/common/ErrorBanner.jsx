export default function ErrorBanner({ message }) {
  if (!message) return null

  return (
    <div className="error-banner" role="alert">
      <span aria-hidden="true">!</span>
      <div>
        <strong>Chưa thể hoàn tất tìm kiếm</strong>
        <p>{message}</p>
      </div>
    </div>
  )
}
