import re


def to_original_expression(original_text: str, phrase: str) -> str:
    """
    Finds and returns the closest match of a phrase within the original text.

    This function searches for a case-insensitive match of the given phrase in the
    original text, considering spaces between words. If a match is found, it returns
    the matched text from the original text. Otherwise, it returns the input phrase.

    Args:
        original_text (str): The text to search within.
        phrase (str): The phrase to search for in the original text.

    Returns:
        str: The matched text from the original text, or the input phrase if no match
             is found.
    """
    pattern = re.compile(
        r"\s*" + r"\s*".join([re.escape(word) for word in phrase.split()]) + r"\s*",
        re.IGNORECASE,
    )
    match = pattern.search(original_text)
    if match:
        return match.group(0).strip()
    else:
        return phrase
