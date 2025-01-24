from typing import Literal

def normalize(
    text: str,
    repeat: int = 1,
    tilde: Literal["normalize", "normalize_zenkaku", "ignore", "remove"] = "normalize",
) -> str: ...
