import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import Icon from '../common/Icon'
import { useLanguage } from '../../i18n'

const MAX_CANVAS_WIDTH = 760
const MAX_CANVAS_HEIGHT = 520
const MAX_OUTPUT_EDGE = 1800
const MIN_SELECTION_SIZE = 18

function pointerPosition(event, canvas) {
  const bounds = canvas.getBoundingClientRect()
  return {
    x: Math.max(0, Math.min(canvas.width, (event.clientX - bounds.left) * (canvas.width / bounds.width))),
    y: Math.max(0, Math.min(canvas.height, (event.clientY - bounds.top) * (canvas.height / bounds.height))),
  }
}

function normalizedSelection(start, end) {
  return {
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
  }
}

export default function ImageCropDialog({ file, onApply, onCancel, onUseOriginal }) {
  const { t } = useLanguage()
  const canvasRef = useRef(null)
  const imageRef = useRef(null)
  const dragStartRef = useRef(null)
  const [selection, setSelection] = useState(null)
  const [imageReady, setImageReady] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [processing, setProcessing] = useState(false)

  useEffect(() => {
    const imageUrl = URL.createObjectURL(file)
    const image = new Image()
    image.onload = () => {
      const canvas = canvasRef.current
      if (!canvas) return
      const scale = Math.min(
        MAX_CANVAS_WIDTH / image.naturalWidth,
        MAX_CANVAS_HEIGHT / image.naturalHeight,
        1,
      )
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale))
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale))
      imageRef.current = image
      setImageReady(true)
    }
    image.src = imageUrl

    return () => {
      URL.revokeObjectURL(imageUrl)
      imageRef.current = null
    }
  }, [file])

  useEffect(() => {
    const canvas = canvasRef.current
    const image = imageRef.current
    if (!canvas || !image || !imageReady) return

    const context = canvas.getContext('2d')
    context.clearRect(0, 0, canvas.width, canvas.height)
    context.drawImage(image, 0, 0, canvas.width, canvas.height)

    if (!selection) return

    context.fillStyle = 'rgba(12, 12, 12, 0.58)'
    context.fillRect(0, 0, canvas.width, canvas.height)
    context.save()
    context.beginPath()
    context.rect(selection.x, selection.y, selection.width, selection.height)
    context.clip()
    context.drawImage(image, 0, 0, canvas.width, canvas.height)
    context.restore()

    context.strokeStyle = '#ffffff'
    context.lineWidth = 3
    context.strokeRect(selection.x, selection.y, selection.width, selection.height)
    context.strokeStyle = '#b11f2d'
    context.lineWidth = 1
    context.strokeRect(selection.x + 2, selection.y + 2, selection.width - 4, selection.height - 4)
  }, [imageReady, selection])

  useEffect(() => {
    function closeOnEscape(event) {
      if (event.key === 'Escape' && !processing) onCancel()
    }
    document.body.classList.add('crop-dialog-open')
    window.addEventListener('keydown', closeOnEscape)
    return () => {
      document.body.classList.remove('crop-dialog-open')
      window.removeEventListener('keydown', closeOnEscape)
    }
  }, [onCancel, processing])

  function beginSelection(event) {
    if (!imageReady || processing) return
    const canvas = canvasRef.current
    const point = pointerPosition(event, canvas)
    dragStartRef.current = point
    setSelection({ x: point.x, y: point.y, width: 0, height: 0 })
    setDragging(true)
    canvas.setPointerCapture?.(event.pointerId)
  }

  function updateSelection(event) {
    if (!dragging || !dragStartRef.current) return
    setSelection(normalizedSelection(
      dragStartRef.current,
      pointerPosition(event, canvasRef.current),
    ))
  }

  function finishSelection(event) {
    if (!dragging) return
    const canvas = canvasRef.current
    canvas.releasePointerCapture?.(event.pointerId)
    setDragging(false)
    dragStartRef.current = null
    setSelection((current) => (
      current?.width >= MIN_SELECTION_SIZE && current?.height >= MIN_SELECTION_SIZE
        ? current
        : null
    ))
  }

  async function applySelection() {
    const canvas = canvasRef.current
    const image = imageRef.current
    if (!canvas || !image || !selection || processing) return

    setProcessing(true)
    try {
      const scaleX = image.naturalWidth / canvas.width
      const scaleY = image.naturalHeight / canvas.height
      const sourceX = Math.round(selection.x * scaleX)
      const sourceY = Math.round(selection.y * scaleY)
      const sourceWidth = Math.max(1, Math.round(selection.width * scaleX))
      const sourceHeight = Math.max(1, Math.round(selection.height * scaleY))
      const outputScale = Math.min(1, MAX_OUTPUT_EDGE / Math.max(sourceWidth, sourceHeight))
      const outputCanvas = document.createElement('canvas')
      outputCanvas.width = Math.max(1, Math.round(sourceWidth * outputScale))
      outputCanvas.height = Math.max(1, Math.round(sourceHeight * outputScale))
      outputCanvas.getContext('2d').drawImage(
        image,
        sourceX,
        sourceY,
        sourceWidth,
        sourceHeight,
        0,
        0,
        outputCanvas.width,
        outputCanvas.height,
      )

      const outputType = ['image/jpeg', 'image/png', 'image/webp'].includes(file.type)
        ? file.type
        : 'image/jpeg'
      const blob = await new Promise((resolve, reject) => {
        outputCanvas.toBlob(
          (value) => value ? resolve(value) : reject(new Error('Unable to crop image.')),
          outputType,
          0.92,
        )
      })
      const extension = outputType === 'image/png' ? 'png' : outputType === 'image/webp' ? 'webp' : 'jpg'
      const baseName = file.name.replace(/\.[^.]+$/, '')
      onApply(new File([blob], `${baseName}-cropped.${extension}`, {
        type: outputType,
        lastModified: Date.now(),
      }))
    } finally {
      setProcessing(false)
    }
  }

  return createPortal(
    <div className="crop-dialog-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget && !processing) onCancel()
    }}>
      <section className="crop-dialog" role="dialog" aria-modal="true" aria-labelledby="crop-dialog-title">
        <header className="crop-dialog-header">
          <div>
            <span>{t('crop.eyebrow')}</span>
            <h2 id="crop-dialog-title">{t('crop.title')}</h2>
            <p>{t('crop.description')}</p>
          </div>
          <button type="button" onClick={onCancel} aria-label={t('crop.cancel')} disabled={processing}>
            <Icon name="close" size={20} />
          </button>
        </header>

        <div className="crop-canvas-shell">
          <canvas
            ref={canvasRef}
            className={dragging ? 'is-selecting' : ''}
            onPointerDown={beginSelection}
            onPointerMove={updateSelection}
            onPointerUp={finishSelection}
            onPointerCancel={finishSelection}
          />
          {!selection && <div className="crop-canvas-hint"><Icon name="image" size={18} /> {t('crop.instruction')}</div>}
        </div>

        <footer className="crop-dialog-actions">
          <button className="crop-secondary" type="button" onClick={onUseOriginal} disabled={processing}>
            {t('crop.useOriginal')}
          </button>
          <div>
            {selection && (
              <button className="crop-secondary" type="button" onClick={() => setSelection(null)} disabled={processing}>
                {t('crop.redraw')}
              </button>
            )}
            <button className="crop-primary" type="button" onClick={applySelection} disabled={!selection || processing}>
              {processing ? <span className="spinner" /> : <Icon name="check" size={17} />}
              {processing ? t('crop.processing') : t('crop.apply')}
            </button>
          </div>
        </footer>
      </section>
    </div>,
    document.body,
  )
}
