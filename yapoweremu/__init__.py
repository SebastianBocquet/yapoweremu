try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

__version__ = metadata.version(__name__)
__author__ = "Sebastian Bocquet"
__email__ = "sebastian.bocquet@gmail.com"
__license__ = "MIT"

from .yapoweremu import Emulator
