"""Background removal for user-supplied fashion images."""

from io import BytesIO

from PIL import Image


_SESSION = None


def _as_rgba(image):
    if isinstance(image, Image.Image):
        return image.convert("RGBA")
    with Image.open(image) as source:
        return source.convert("RGBA")


def remove_image_background(image, background_color=(255, 255, 255)):
    """Remove the background and composite the garment on a neutral canvas.

    The output keeps the original image dimensions so ResNet preprocessing sees
    stable framing. ``rembg`` is imported lazily to avoid loading its ONNX model
    until an uploaded query actually needs background removal.
    """
    global _SESSION
    try:
        from rembg import new_session, remove
    except ImportError as error:
        raise RuntimeError(
            "Background removal requires rembg. Install model/combine_model/requirements.txt."
        ) from error

    if _SESSION is None:
        _SESSION = new_session("u2netp")
    source = _as_rgba(image)
    output = remove(source, session=_SESSION)
    if isinstance(output, bytes):
        with Image.open(BytesIO(output)) as decoded:
            foreground = decoded.convert("RGBA")
    elif isinstance(output, Image.Image):
        foreground = output.convert("RGBA")
    else:
        raise TypeError(f"Unsupported rembg output type: {type(output).__name__}")

    canvas = Image.new("RGBA", foreground.size, (*background_color, 255))
    return Image.alpha_composite(canvas, foreground).convert("RGB")
