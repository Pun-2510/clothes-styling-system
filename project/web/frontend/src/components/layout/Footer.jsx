import { useLanguage } from '../../i18n'

export default function Footer() {
  const { t } = useLanguage()

  return (
    <footer className="site-footer">
      <div className="footer-brand">
        <span className="brand-mark">TF</span>
        <div><strong>THREAD/FIND</strong><p>{t('footer.description')}</p></div>
      </div>
      <nav aria-label="Footer">
        <a href="/">Home</a>
        <a href="/categories">Categories</a>
        <a href="/developer">Developer</a>
      </nav>
      <div className="footer-technical">
        <span>FINE-TUNED CLIP</span><span>CATEGORY-AWARE</span><span>© 2026</span>
      </div>
    </footer>
  )
}
