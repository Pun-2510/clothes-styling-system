import { useEffect, useMemo, useRef, useState } from 'react'
import Icon from '../common/Icon'
import { ACCEPTED_IMAGE_TYPES, MAX_IMAGE_SIZE, RESULT_COUNTS } from '../../data/searchOptions'
import { searchByCombined, searchByImage, searchByText } from '../../services/recommendationApi'
import { clearSearchFile, loadSearchFile, loadSearchFormState, saveSearchFile, saveSearchFormState } from '../../services/searchPersistence'
import { useLanguage } from '../../i18n'
import ImageCropDialog from './ImageCropDialog'

export default function CombinedSearchForm({ loading, onSearchStart, onSearchSuccess, onSearchError, onClearResult, initialQuery = '', restoreRequest = null, querySuggestion = null, refinementRequest = null }) {
  const { t } = useLanguage()
  const [savedForm] = useState(loadSearchFormState)
  const [file, setFile] = useState(null)
  const [pendingCropFile, setPendingCropFile] = useState(null)
  const [query, setQuery] = useState(initialQuery || savedForm.query || '')
  const [language, setLanguage] = useState(savedForm.language === 'en' ? 'en' : 'vi')
  const [topK, setTopK] = useState(RESULT_COUNTS.includes(savedForm.topK) ? savedForm.topK : 5)
  const [audience, setAudience] = useState(['women', 'men', 'kids', 'unisex'].includes(savedForm.audience) ? savedForm.audience : '')
  const inputRef = useRef(null)
  const queryInputRef = useRef(null)
  const onClearResultRef = useRef(onClearResult)
  const preview = useMemo(() => file ? URL.createObjectURL(file) : '', [file])
  const hasImage = Boolean(file)
  const hasText = Boolean(query.trim())
  const canSearch = hasImage || hasText
  const settingsSummary = [
    hasText ? (language === 'vi' ? 'Tiếng Việt' : 'English') : null,
    t('search.products', { count: topK }),
    audience ? t(`audience.${audience}`) : t('audience.any'),
  ].filter(Boolean).join(' · ')

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview)
  }, [preview])

  useEffect(() => {
    onClearResultRef.current = onClearResult
  }, [onClearResult])

  useEffect(() => {
    let active = true
    loadSearchFile().then((savedFile) => {
      if (!active || !savedFile) return
      if (ACCEPTED_IMAGE_TYPES.includes(savedFile.type) && savedFile.size <= MAX_IMAGE_SIZE) setFile(savedFile)
      else clearSearchFile()
    })
    return () => { active = false }
  }, [])

  useEffect(() => {
    saveSearchFormState({ query, language, topK, audience })
  }, [query, language, topK, audience])

  useEffect(() => {
    if (!restoreRequest) return

    const restoredFile = restoreRequest.file || null
    setQuery(restoreRequest.query || '')
    setLanguage(restoreRequest.language === 'en' ? 'en' : 'vi')
    setTopK(RESULT_COUNTS.includes(restoreRequest.topK) ? restoreRequest.topK : 5)
    setAudience(['women', 'men', 'kids', 'unisex'].includes(restoreRequest.audience) ? restoreRequest.audience : '')
    setFile(restoredFile)
    if (restoredFile) saveSearchFile(restoredFile)
    else clearSearchFile()
    if (inputRef.current) inputRef.current.value = ''
    onClearResultRef.current?.()
  }, [restoreRequest])

  useEffect(() => {
    if (!querySuggestion?.query) return
    setQuery(querySuggestion.query)
    setLanguage(querySuggestion.language === 'vi' ? 'vi' : 'en')
    setFile(null)
    clearSearchFile()
    if (inputRef.current) inputRef.current.value = ''
    onClearResultRef.current?.()
  }, [querySuggestion])

  useEffect(() => {
    if (!refinementRequest?.id) return undefined
    const frame = window.requestAnimationFrame(() => {
      queryInputRef.current?.focus({ preventScroll: true })
      const cursorPosition = queryInputRef.current?.value.length || 0
      queryInputRef.current?.setSelectionRange(cursorPosition, cursorPosition)
    })
    return () => window.cancelAnimationFrame(frame)
  }, [refinementRequest])

  function acceptFile(candidate) {
    if (!candidate) return
    onClearResult()
    if (!ACCEPTED_IMAGE_TYPES.includes(candidate.type)) {
      onSearchError(t('search.invalidType'))
      return
    }
    if (candidate.size > MAX_IMAGE_SIZE) {
      onSearchError(t('search.invalidSize'))
      return
    }
    setPendingCropFile(candidate)
  }

  function commitImage(candidate) {
    setPendingCropFile(null)
    setFile(candidate)
    saveSearchFile(candidate)
    onClearResult()
  }

  function updateQuery(value) {
    setQuery(value)
    onClearResult()
  }

  async function submit(event) {
    event.preventDefault()
    if (!canSearch || loading) return

    onSearchStart()
    const requestContext = {
      query: query.trim(),
      language,
      topK,
      file,
      audience,
    }
    try {
      if (hasImage && hasText) {
        const result = await searchByCombined({ file, query, language, topK, audience })
        onSearchSuccess(result, 'combined', requestContext)
      } else if (hasImage) {
        const result = await searchByImage({ file, topK, audience })
        onSearchSuccess(result, 'image', requestContext)
      } else {
        const result = await searchByText({ query, language, topK, audience })
        onSearchSuccess(result, 'text', requestContext)
      }
    } catch (error) {
      onSearchError(error.message)
    }
  }

  const submitLabel = hasImage && hasText
    ? t('search.combined')
    : hasImage
      ? t('search.byImage')
      : t('search.submit')

  return (
    <form className="search-panel compact-search-panel" onSubmit={submit}>
      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={(event) => acceptFile(event.target.files?.[0])}
      />
      {pendingCropFile && (
        <ImageCropDialog
          file={pendingCropFile}
          onApply={commitImage}
          onUseOriginal={() => commitImage(pendingCropFile)}
          onCancel={() => {
            setPendingCropFile(null)
            if (inputRef.current) inputRef.current.value = ''
          }}
        />
      )}
      <div className="compact-search-bar">
        <Icon name="search" size={20} />
        <input
          ref={queryInputRef}
          id="combined-fashion-query"
          value={query}
          onChange={(event) => updateQuery(event.target.value)}
          placeholder={refinementRequest && hasImage && !hasText ? t('search.refinePlaceholder') : t('search.placeholder')}
          maxLength={512}
        />
        <button className={`attach-image ${file ? 'has-file' : ''}`} type="button" onClick={() => inputRef.current?.click()}>
          <Icon name="image" size={20} /><span>{file ? t('search.changeImage') : t('search.addImage')}</span>
        </button>
        <button className="compact-submit" type="submit" disabled={!canSearch || loading} aria-busy={loading}>
          <Icon name="arrow" size={20} />
          <span>{submitLabel}</span>
        </button>
      </div>

      {preview && (
        <div className="selected-image">
          <img src={preview} alt={t('search.selectedImage')} />
          <span><strong>{file.name}</strong><small>{(file.size / 1024 / 1024).toFixed(2)} MB</small></span>
          <button type="button" onClick={() => { setFile(null); clearSearchFile(); onClearResult(); if (inputRef.current) inputRef.current.value = '' }}>
            <Icon name="close" size={17} /> {t('search.removeImage')}
          </button>
        </div>
      )}

      <details className="search-preferences">
        <summary className="search-preferences-toggle">
          <span className="search-preferences-heading">
            <strong>{t('search.settingsTitle')}</strong>
            <span>{settingsSummary}</span>
          </span>
          <span className="search-preferences-toggle-action">
            <span className="settings-expand-label">{t('search.settingsExpand')}</span>
            <span className="settings-collapse-label">{t('search.settingsCollapse')}</span>
            <Icon name="chevron" size={17} />
          </span>
        </summary>

        <div className="search-preferences-panel">
          <p className="search-preferences-intro">{t('search.settingsIntro')}</p>

          <div className="search-preferences-layout">
            <div className="search-basic-preferences">
              {hasText && (
                <label>{t('search.queryLanguage')}
                  <select value={language} onChange={(event) => setLanguage(event.target.value)}>
                    <option value="vi">Tiếng Việt</option><option value="en">English</option>
                  </select>
                </label>
              )}
              <label>{t('search.resultCount')}
                <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
                  {RESULT_COUNTS.map((value) => <option key={value} value={value}>{t('search.products', { count: value })}</option>)}
                </select>
              </label>
            </div>

            <fieldset className="audience-preference-fieldset">
              <legend>{t('search.audiencePreference')}</legend>
              <div className="audience-preference-options">
                {['', 'women', 'men', 'kids', 'unisex'].map((value) => (
                  <button
                    className={audience === value ? 'active' : ''}
                    type="button"
                    aria-pressed={audience === value}
                    key={value || 'default'}
                    onClick={() => { setAudience(value); onClearResult() }}
                  >
                    {value ? t(`audience.${value}`) : t('audience.any')}
                  </button>
                ))}
              </div>
            </fieldset>
          </div>

          <div className="audience-manual-notice" role="note">
            <span><Icon name="spark" size={17} /></span>
            <div>
              <strong>{t('search.audienceManualTitle')}</strong>
              <p>{t('search.audienceHint')}</p>
            </div>
          </div>
        </div>
      </details>
    </form>
  )
}
