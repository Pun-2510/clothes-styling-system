import { useLanguage } from '../../i18n'

export default function ErrorBanner({ message }) {
  const { t } = useLanguage()
  if (!message) return null

  return (
    <div className="error-banner" role="alert">
      <span aria-hidden="true">!</span>
      <div>
        <strong>{t('error.searchTitle')}</strong>
        <p>{message}</p>
      </div>
    </div>
  )
}
