
from typing import Callable

from sparkkit.dplyr import ChainableDF  # Ensure this import matches your project structure

class DataFrame:
    def __rshift__(self, operation: Callable[[ChainableDF], ChainableDF]) -> ChainableDF: ...