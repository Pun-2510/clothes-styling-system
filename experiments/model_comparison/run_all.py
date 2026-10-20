"""Compatibility entry point for the unified full-gallery comparison.

Keep this module so existing commands do not fail, but delegate all arguments
to the canonical ``compare_full_gallery`` implementation.
"""

from .compare_full_gallery import main


if __name__ == "__main__":
    main()
