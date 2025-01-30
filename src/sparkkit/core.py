from __future__ import annotations

import re
from typing import Any, Callable

from pyspark.sql import DataFrame
from pyspark.sql.column import Column


class ChainableDF:
    """Provides chainable methods for performing operations on a PySpark DataFrame."""

    def __init__(self, df: DataFrame, group_cols: list[str | Column] | None = None):
        """
        Initializes the ChainableDF with a PySpark DataFrame and optional grouping columns.

        Args:
            df (DataFrame): The PySpark DataFrame to wrap.
            group_cols (list[str | Column], optional): Columns to group by. Defaults to None.
        """
        self.df = df
        self.group_cols = [
            col._jc.toString() if isinstance(col, Column) else col  # type: ignore
            for col in (group_cols or [])
        ]

    def __rshift__(
        self, operation: Callable[[ChainableDF], ChainableDF]
    ) -> ChainableDF:
        """
        Applies a chainable operation to the DataFrame using the >> operator.

        Args:
            operation (Callable[[ChainableDF], ChainableDF]): The operation to apply.

        Returns:
            ChainableDF: The resulting ChainableDF after applying the operation.
        """
        return operation(self)


class Selector:
    """Base class for column selectors used in DataFrame operations."""

    def __call__(self, df: DataFrame) -> list[str | Column]:
        """
        Select columns from the DataFrame.
        
        Args:
            df (DataFrame): The DataFrame to select columns from.
        
        Returns:
            list[str | Column]: A list of selected column names or Column objects.
        
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError


class StartsWith(Selector):
    """Selector that selects columns starting with a specified prefix."""

    def __init__(self, prefix: str):
        """
        Initialize the StartsWith selector with a prefix.
        
        Args:
            prefix (str): The prefix to match column names.
        """
        self.prefix = prefix

    def __call__(self, df: DataFrame) -> list[str | Column]:
        """
        Select columns that start with the given prefix.
        
        Args:
            df (DataFrame): The DataFrame to select columns from.
        
        Returns:
            list[str | Column]: A list of column names that start with the prefix.
        """
        return [col for col in df.columns if col.startswith(self.prefix)]


class EndsWith(Selector):
    """Selector that selects columns ending with a specified suffix."""

    def __init__(self, suffix: str):
        """
        Initialize the EndsWith selector with a suffix.
        
        Args:
            suffix (str): The suffix to match column names.
        """
        self.suffix = suffix

    def __call__(self, df: DataFrame) -> list[str | Column]:
        """
        Select columns that end with the given suffix.
        
        Args:
            df (DataFrame): The DataFrame to select columns from.
        
        Returns:
            list[str | Column]: A list of column names that end with the suffix.
        """
        return [col for col in df.columns if col.endswith(self.suffix)]


class Contains(Selector):
    """Selector that selects columns containing a specified substring."""

    def __init__(self, string: str):
        """
        Initialize the Contains selector with a substring.
        
        Args:
            string (str): The substring to look for in column names.
        """
        self.string = string

    def __call__(self, df: DataFrame) -> list[str | Column]:
        """
        Select columns that contain the specified substring.
        
        Args:
            df (DataFrame): The DataFrame to select columns from.
        
        Returns:
            list[str | Column]: A list of column names that contain the substring.
        """
        return [col for col in df.columns if self.string in col]


class Matches(Selector):
    """Selector that selects columns matching a specified regular expression pattern."""

    def __init__(self, pattern: str):
        """
        Initialize the Matches selector with a regex pattern.
        
        Args:
            pattern (str): The regular expression pattern to match column names.
        """
        self.pattern = re.compile(pattern)

    def __call__(self, df: DataFrame) -> list[str | Column]:
        """
        Select columns that match the given regex pattern.
        
        Args:
            df (DataFrame): The DataFrame to select columns from.
        
        Returns:
            list[str | Column]: A list of column names that match the pattern.
        """
        return [col for col in df.columns if self.pattern.search(col)]


class NumRange(Selector):
    """Selector that selects columns within a numerical range following a prefix."""

    def __init__(self, prefix: str, range: range):
        """
        Initialize the NumRange selector with a prefix and numerical range.
        
        Args:
            prefix (str): The prefix of the column names.
            range (range): The numerical range to match.
        """
        self.prefix = prefix
        self.range = range

    def __call__(self, df: DataFrame) -> list[str | Column]:
        """
        Select columns that fall within the specified numerical range.
        
        Args:
            df (DataFrame): The DataFrame to select columns from.
        
        Returns:
            list[str | Column]: A list of column names within the range.
        """
        return [
            f"{self.prefix}{i}" for i in self.range if f"{self.prefix}{i}" in df.columns
        ]


class Everything(Selector):
    """Selector that selects all columns in the DataFrame."""

    def __call__(self, df: DataFrame) -> list[str | Column]:
        """
        Select all columns from the DataFrame.
        
        Args:
            df (DataFrame): The DataFrame to select columns from.
        
        Returns:
            list[str | Column]: A list of all column names in the DataFrame.
        """
        return df.columns  # type: ignore


def starts_with(prefix: str) -> Selector:
    """
    Create a Selector that selects columns starting with the given prefix.

    Args:
        prefix (str): The prefix to match column names.

    Returns:
        Selector: An instance of StartsWith selector.

    Examples:
        >>> from sparkkit.core import ChainableDF, select, starts_with
        >>> chain = ChainableDF(df)
        >>> result = chain >> select(starts_with("prefix"))
    """
    return StartsWith(prefix)


def ends_with(suffix: str) -> Selector:
    """
    Create a Selector that selects columns ending with the given suffix.

    Args:
        suffix (str): The suffix to match column names.

    Returns:
        Selector: An instance of EndsWith selector.

    Examples:
        >>> from sparkkit.core import ChainableDF, select, ends_with
        >>> chain = ChainableDF(df)
        >>> result = chain >> select(ends_with("suffix"))
    """
    return EndsWith(suffix)


def contains(string: str) -> Selector:
    """
    Create a Selector that selects columns containing the given substring.

    Args:
        string (str): The substring to look for in column names.

    Returns:
        Selector: An instance of Contains selector.

    Examples:
        >>> from sparkkit.core import ChainableDF, select, contains
        >>> chain = ChainableDF(df)
        >>> result = chain >> select(contains("substring"))
    """
    return Contains(string)


def matches(pattern: str) -> Selector:
    """
    Create a Selector that selects columns matching the given regex pattern.

    Args:
        pattern (str): The regular expression pattern to match column names.

    Returns:
        Selector: An instance of Matches selector.

    Examples:
        >>> from sparkkit.core import ChainableDF, select, matches
        >>> chain = ChainableDF(df)
        >>> result = chain >> select(matches(r"regex_pattern"))
    """
    return Matches(pattern)


def num_range(prefix: str, start: int, end: int) -> Selector:
    """
    Create a Selector that selects columns within a numerical range following a prefix.

    Args:
        prefix (str): The prefix of the column names.
        start (int): The starting number of the range.
        end (int): The ending number of the range.

    Returns:
        Selector: An instance of NumRange selector.

    Examples:
        >>> from sparkkit.core import ChainableDF, select, num_range
        >>> chain = ChainableDF(df)
        >>> result = chain >> select(num_range("col_", 1, 5))
    """
    return NumRange(prefix, range(start, end + 1))


def everything() -> Selector:
    """
    Create a Selector that selects all columns in the DataFrame.

    Returns:
        Selector: An instance of Everything selector.

    Examples:
        >>> from sparkkit.core import ChainableDF, select, everything
        >>> chain = ChainableDF(df)
        >>> result = chain >> select(everything())
    """
    return Everything()
