import { useState } from 'react'
import '../styles/auth.css'

function Eye({ visible }) {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" /><circle cx="12" cy="12" r="3" />{visible && <path d="m3 3 18 18" />}</svg>
}

function AuthForm({ mode, onSwitch }) {
  const register = mode === 'register'
  const [visible, setVisible] = useState(false)
  const [notice, setNotice] = useState('')

  function submit(event) {
    event.preventDefault()
    const form = event.currentTarget
    const values = new FormData(form)
    if (register && values.get('password') !== values.get('confirmPassword')) {
      form.elements.confirmPassword.setCustomValidity('Mật khẩu xác nhận chưa khớp.')
      form.elements.confirmPassword.reportValidity()
      return
    }
    setNotice('Đây là bản xem trước giao diện. Chức năng tài khoản sẽ được kết nối sau; thông tin của bạn chưa được gửi hoặc lưu.')
  }

  return (
    <div className="auth-page">
      <section className="auth-story" aria-label="Giới thiệu clothes styling">
        <div className="auth-brand" aria-label="clothes styling">
          <span className="auth-brand-icon">✳</span> clothes<span className="auth-brand-light"> styling</span><span className="auth-brand-period">.</span>
        </div>
        <div className="auth-story-copy">
          <span className="auth-kicker"><span /> YOUR STYLE, REIMAGINED</span>
          <h1>Phong cách của bạn.<br /><em>Cảm hứng mỗi ngày.</em></h1>
          <p>Từ một ý tưởng nhỏ đến món đồ bạn yêu thích.<br />Khám phá thời trang theo cách riêng của bạn.</p>
        </div>
        <div className="auth-art" aria-hidden="true">
          <div className="auth-orbit auth-orbit-one" /><div className="auth-orbit auth-orbit-two" />
          <span className="auth-art-star">✦</span>
          <div className="auth-fashion-card">
            <div className="auth-card-top"><span>THE EVERYDAY EDIT</span><span>↗</span></div>
            <svg className="auth-shirt" viewBox="0 0 260 250" fill="none"><defs><linearGradient id="shirt" x1="50" y1="20" x2="215" y2="230" gradientUnits="userSpaceOnUse"><stop stopColor="#fefcff" /><stop offset="1" stopColor="#d6c5fa" /></linearGradient></defs><path d="m83 37 27-12c10 13 30 13 40 0l27 12 49 46-30 37-22-17 6 119H80l6-119-22 17-30-37 49-46Z" fill="url(#shirt)" stroke="#fff" strokeWidth="2" /><path d="M110 25c-2 30 43 30 40 0M85 101l7-42m82 42-7-42M86 211h87" stroke="#b6a0db" strokeWidth="2" /><path d="m128 65-9 130m32-72 7 71" stroke="#e6dcf7" strokeWidth="3" /></svg>
            <div className="auth-card-bottom"><div><strong>Less, but better.</strong><span>Những lựa chọn mang dấu ấn riêng</span></div><span className="auth-card-heart">♡</span></div>
          </div>
          <div className="auth-floating-label"><span>✧</span><div><strong>Một chút cảm hứng</strong><small>Một phong cách rất bạn</small></div></div>
          <div className="auth-color-label"><span /><span /><span /><small>YOUR PALETTE</small></div>
        </div>
        <div className="auth-story-footer"><span>Khám phá. Yêu thích. Là chính mình.</span><span>01 — 03</span></div>
      </section>

      <section className="auth-panel" aria-labelledby="auth-title">
        <div className="auth-top-note">{register ? 'Đã có tài khoản?' : 'Lần đầu đến đây?'} <button type="button" onClick={onSwitch}>{register ? 'Đăng nhập' : 'Tạo tài khoản'} <span aria-hidden="true">↗</span></button></div>
        <div className="auth-form-wrap">
          <div className="auth-welcome-icon" aria-hidden="true">{register ? '✦' : '✧'}</div>
          <span className="auth-form-kicker">YOUR NEXT FAVORITE STARTS HERE</span>
          <h2 id="auth-title">{register ? 'Bắt đầu câu chuyện của bạn.' : 'Chào mừng trở lại.'}</h2>
          <p className="auth-subtitle">{register ? 'Tạo tài khoản và tìm cảm hứng cho phong cách riêng.' : 'Đăng nhập để tiếp tục hành trình tìm phong cách của bạn.'}</p>
          <form className="auth-form" onSubmit={submit} onChange={() => setNotice('')}>
            {register && <label className="auth-field">Họ và tên<input name="name" autoComplete="name" placeholder="Nguyễn Minh Anh" required maxLength={100} /></label>}
            <label className="auth-field">Địa chỉ email<input name="email" type="email" autoComplete="email" placeholder="ban@example.com" required maxLength={254} /></label>
            <label className="auth-field">Mật khẩu<div className="auth-password"><input name="password" type={visible ? 'text' : 'password'} autoComplete={register ? 'new-password' : 'current-password'} placeholder={register ? 'Tối thiểu 8 ký tự' : 'Nhập mật khẩu của bạn'} minLength={register ? 8 : undefined} required onChange={event => { const field = event.currentTarget.form.elements.confirmPassword; if (field) field.setCustomValidity('') }} /><button type="button" aria-label={visible ? 'Ẩn mật khẩu' : 'Hiện mật khẩu'} aria-pressed={visible} onClick={() => setVisible(!visible)}><Eye visible={visible} /></button></div></label>
            {register && <label className="auth-field">Xác nhận mật khẩu<input name="confirmPassword" type={visible ? 'text' : 'password'} autoComplete="new-password" placeholder="Nhập lại mật khẩu" required onChange={event => event.currentTarget.setCustomValidity('')} /></label>}
            {!register && <div className="auth-options"><label><input type="checkbox" name="remember" />Ghi nhớ đăng nhập</label><button type="button" onClick={() => setNotice('Chức năng khôi phục mật khẩu chưa được kết nối trong bản xem trước.')}>Quên mật khẩu?</button></div>}
            {register && <p className="auth-register-note">Một không gian dành cho cảm hứng thời trang của riêng bạn.</p>}
            <button className="auth-submit" type="submit">{register ? 'Tạo tài khoản' : 'Đăng nhập'}<span aria-hidden="true">→</span></button>
            {notice && <p className="auth-notice" role="status">{notice}</p>}
          </form>
          <div className="auth-divider"><span />Phong cách bắt đầu từ bạn<span /></div>
          <p className="auth-bottom-note">{register ? 'Đã tìm thấy phong cách của mình?' : 'Chưa có tài khoản?'} <button type="button" onClick={onSwitch}>{register ? 'Đăng nhập ngay' : 'Đăng ký miễn phí'}</button></p>
        </div>
        <footer className="auth-footer"><span>© {new Date().getFullYear()} clothes styling</span><span>Made for your kind of style <span aria-hidden="true">✦</span></span></footer>
      </section>
    </div>
  )
}

export default function LoginPage() {
  const [mode, setMode] = useState('login')
  return <AuthForm key={mode} mode={mode} onSwitch={() => setMode(mode === 'login' ? 'register' : 'login')} />
}
