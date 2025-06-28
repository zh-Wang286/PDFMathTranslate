# Set up version and metadata first
__version__ = "1.9.10"
__author__ = "Byaidu"

# Set up logging
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Import functionality
from pdf2zh.high_level import translate, translate_stream
from pdf2zh.pdf2zh import translate_file

__all__ = ["translate", "translate_stream", "translate_file"]
