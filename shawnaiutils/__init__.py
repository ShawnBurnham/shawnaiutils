try:
    from .callbacks import standard_callbacks
except ImportError:
    from callbacks import standard_callbacks

__all__ = ["standard_callbacks"]
