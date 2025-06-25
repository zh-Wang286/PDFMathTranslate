"""Utility functions for PDFMathTranslate."""

import logging
import re
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


def contains_zh_or_en(text: str) -> bool:
    """
    检查字符串是否包含中文或英文字母。

    Args:
        text: 要检查的字符串

    Returns:
        bool: 如果字符串包含中文字符或英文字母返回True，否则返回False
    """
    return bool(re.search(r'[\u4e00-\u9fff]|[a-zA-Z]', text))


if __name__ == "__main__":
    # Basic logging setup to see messages
    logging.basicConfig(level=logging.INFO)

    test_text = """
<think>                                                                                                                 converter.py:673
                             Alright, I need to translate this medical text from Chinese to English. First, I'll read through the entire source text to understand the context and main
                             points. It's about the increased risk of deep vein thrombosis (DVT) in cancer patients compared to non-cancer patients, with specific statistics for different
                             types of cancers.

                             I should make sure to maintain the terminology accurately. Terms like "deep vein thrombosis" and "venous thrombosis" are standard, so I'll keep those as is. The
                             abbreviations like OR (odds ratio) and CI (confidence interval) are commonly used in medical literature, so they don't need translation.

                             Looking at the statistics, I need to ensure that the numbers and their formats are correctly translated. For example, "odds ratio (OR): 22.2; 95% conﬁdence
                             interval (CI): 3.6–136.1" should remain consistent in English. I'll also check the punctuation, making sure that commas and hyphens are appropriately placed.

                             The sentence structure in Chinese might be a bit different from English, so I'll adjust it for clarity and flow. For instance, the original mentions "In a
                             recent case–control study," which I'll translate to maintain the same meaning but ensure it's concise.

                             I should also pay attention to the specific types of cancers mentioned, like "lung cancer," "hematological cancer," and "gastrointestinal cancer." These terms
                             are straightforward, but I'll double-check their English equivalents to ensure accuracy.

                             The part about "large cohort studies" needing to estimate VT incidence in low-incidence cancers is important. I'll make sure this is clearly conveyed in
                             English, perhaps rephrasing for better readability without losing the original meaning.

                             Finally, the conclusion about the importance of knowing VT incidence for clinical decisions on prophylactic treatment should be translated to emphasize its
                             practical relevance. I'll ensure that the translation maintains the professional tone required for medical texts.

                             Overall, I'll go through each sentence, translate it accurately, and then review the entire text to ensure consistency, clarity, and adherence to medical
                             terminology standards.
                             </think>

    """

    num_tokens = count_tokens(test_text)
    print(f"Text: '{test_text}'")
    print(f"Number of tokens: {num_tokens}")

    # # Example with a longer text
    # long_text = "This is a longer sentence to test the tokenizer."
    # num_tokens_long = count_tokens(long_text)
    # print(f"Text: '{long_text}'")
    # print(f"Number of tokens: {num_tokens_long}")

    # 测试contains_zh_or_en函数
    test_cases = [
        "Hello World",  # 纯英文
        "你好世界",    # 纯中文
        "Hello你好",   # 中英混合
        "123456",     # 纯数字
        "!@#$%^",    # 纯符号
        "123Hello",   # 数字和英文
        "123你好",    # 数字和中文
        "",          # 空字符串
        " ",         # 空格
        "Hello世界123!@#" # 混合内容
    ]

    print("\n=== Testing contains_zh_or_en function ===")
    for test in test_cases:
        result = contains_zh_or_en(test)
        print(f"Text: '{test}'")
        print(f"Contains Chinese or English: {result}\n")
