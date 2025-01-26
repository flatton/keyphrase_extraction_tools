class Tree:
    def label(self) -> str: ...
    def leaves(self) -> list[tuple[str, str]]: ...
    def subtrees(self) -> list["Tree"]: ...

class RegexpParser:
    def __init__(self, grammar: str) -> None: ...
    def parse(self, chunk_struct: list[tuple[str, str]], trace: int = 0) -> Tree: ...
