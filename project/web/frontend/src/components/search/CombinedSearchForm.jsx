import { useEffect, useMemo, useRef, useState } from 'react'
import Icon from '../common/Icon'
import { ACCEPTED_IMAGE_TYPES, MAX_IMAGE_SIZE, RESULT_COUNTS } from '../../data/searchOptions'
import { searchByCombined, searchByImage, searchByText } from '../../services/recommendationApi'
import { clearSearchFile, loadSearchFile, loadSearchFormState, saveSearchFile, saveSearchFormState } from '../../services/searchPersistence'
import { useLanguage } from '../../i18n'
import ImageCropDialog from './ImageCropDialog'

export default function CombinedSearchForm({ loading, onSearchStart, onSearchSuccess, onSearchError, onClearResult, initialQuery = '', restoreRequest = null, querySuggestion = null }) {
  const { t } = useLanguage()
  const [savedForm] = useState(loadSearchFormState)
  const [file, setFile] = useState(null)
  const [pendingCropFile, setPendingCropFile] = useState(null)
  const [query, setQuery] = useState(initialQuery || savedForm.query || '')
  const [language, setLanguage] = useState(savedForm.language === 'en' ? 'en' : 'vi')
  const [topK, setTopK] = useState(RESULT_COUNTS.includes(savedForm.topK) ? savedForm.topK : 5)
  const inputRef = useRef(null)
  const onClearResultRef = useRef(onClearResult)
  const preview = useMemo(() => file ? URL.createObjectURL(file) : '', [file])
  const hasImage = Boolean(file)
  const hasText = Boolean(query.trim())
  const canSearch = hasImage || hasText

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
    saveSearchFormState({ query, language, topK })
  }, [query, language, topK])

  useEffect(() => {
    if (!restoreRequest) return

    const restoredFile = restoreRequest.file || null
    setQuery(restoreRequest.query || '')
    setLanguage(restoreRequest.language === 'en' ? 'en' : 'vi')
    setTopK(RESULT_COUNTS.includes(restoreRequest.topK) ? restoreRequest.topK : 5)
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
    }
    try {
      if (hasImage && hasText) {
        const result = await searchByCombined({ file, query, language, topK })
        onSearchSuccess(result, 'combined', requestContext)
      } else if (hasImage) {
        const result = await searchByImage({ file, topK })
        onSearchSuccess(result, 'image', requestContext)
      } else {
        const result = await searchByText({ query, language, topK })
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
          id="combined-fashion-query"
          value={query}
          onChange={(event) => updateQuery(event.target.value)}
          placeholder={t('search.placeholder')}
          maxLength={512}
        />
        <button className={`attach-image ${file ? 'has-file' : ''}`} type="button" onClick={() => inputRef.current?.click()}>
          <Icon name="image" size={20} /><span>{file ? t('search.changeImage') : t('search.addImage')}</span>
        </button>
        <button className="compact-submit" type="submit" disabled={!canSearch || loading}>
          {loading ? <span className="spinner" /> : <Icon name="arrow" size={20} />}
          <span>{loading ? t('search.loading') : submitLabel}</span>
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

      <details className="search-options">
        <summary><span>{t('search.options')}</span><Icon name="chevron" size={14} /></summary>
        <div className="search-options-content">
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
      </details>
    </form>
  )
}
