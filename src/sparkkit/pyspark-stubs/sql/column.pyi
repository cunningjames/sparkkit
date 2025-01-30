from typing import Any, Tuple

class Column:
    def __rshift__(self, other: Any) -> Tuple["Column", "Column"]: ...
