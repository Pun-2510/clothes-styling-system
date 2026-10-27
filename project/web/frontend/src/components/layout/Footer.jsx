import { useLanguage } from '../../i18n'

export default function Footer() {
  const { t } = useLanguage()

  return (
    <footer className="site-footer">
      <span>THREAD/FIND © 2026</span>
      <span>{t('footer.description')}</span>
    </footer>
  )
}
