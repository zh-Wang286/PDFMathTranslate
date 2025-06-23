"""Utility functions for PDFMathTranslate."""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

TOKENIZER_PATH = "qwen3-tokenizer.json"
_tokenizer_instance = None
_tokenizer_failed = False


@lru_cache(maxsize=1)
def get_tokenizer():
    """
    Loads the tokenizer from the specified path.

    The path is hardcoded to a specific Hugging Face cache location as per requirements.
    This function will cache the tokenizer instance for performance.

    Returns:
        An instance of tokenizers.Tokenizer, or None if the tokenizer file
        cannot be found or loaded.
    """
    global _tokenizer_instance, _tokenizer_failed
    if _tokenizer_instance:
        return _tokenizer_instance
    if _tokenizer_failed:
        return None

    try:
        from tokenizers import Tokenizer
    except ImportError:
        logger.error("The 'tokenizers' library is not installed. Please install it with 'pip install tokenizers'")
        _tokenizer_failed = True
        return None

    try:
        # The user specified that the file at TOKENIZER_PATH is a tokenizer.json.
        # The from_file method can handle loading it.
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        logger.debug(f"Successfully loaded tokenizer from {TOKENIZER_PATH}")
        _tokenizer_instance = tokenizer
        return _tokenizer_instance
    except Exception:
        logger.warning(
            f"Failed to load tokenizer from the specified path: {TOKENIZER_PATH}. "
            "Token counting will fall back to a simple character count."
        )
        _tokenizer_failed = True
        return None


def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in a given text using a pre-defined Qwen tokenizer.

    If the tokenizer cannot be loaded, it falls back to a rough estimation
    (length of text) and logs a warning. This is a simple fallback and may not be accurate.

    Args:
        text: The input string.

    Returns:
        The number of tokens.
    """
    tokenizer = get_tokenizer()
    if tokenizer:
        return len(tokenizer.encode(text).ids)
    else:
        # Fallback if tokenizer is not available
        return len(text)


if __name__ == "__main__":
    # Basic logging setup to see messages
    logging.basicConfig(level=logging.INFO)

    test_text = "你好"
    num_tokens = count_tokens(test_text)
    print(f"Text: '{test_text}'")
    print(f"Number of tokens: {num_tokens}")

    # Example with a longer text
    long_text = "This is a longer sentence to test the tokenizer."
    num_tokens_long = count_tokens(long_text)
    print(f"Text: '{long_text}'")
    print(f"Number of tokens: {num_tokens_long}")
