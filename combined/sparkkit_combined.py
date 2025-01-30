from __future__ import annotations
import re
from typing import Any, Callable
from pyspark.sql import DataFrame
from pyspark.sql.column import Column
import pyspark.sql.functions as f
from pyspark.sql import DataFrame, SparkSession
from typing import Callable, cast
from pyspark.sql.functions import col
from pyspark.sql.functions import count as spark_count
from pyspark.sql.functions import expr, lit, when
from pyspark.sql.window import Window
from enum import Enum
from pyspark.sql.functions import col, expr
from typing import Callable

# Contents of core.py




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

# Contents of dplyr.py





def _rshift_for_case_when(self: Column, other: Any) -> tuple[Column, Column]:
    if not isinstance(other, Column):
        other = f.lit(other)
    return (self, other)


Column.__rshift__ = _rshift_for_case_when  # type: ignore


def display() -> Callable:
    def _display(chain: ChainableDF) -> None:
        if hasattr(chain.df, "display"):
            chain.df.display()  # type: ignore
        else:
            chain.df.show(n=20)
        return None

    return _display


def dataframe_rshift(
    self, operation: Callable[[ChainableDF], ChainableDF]
) -> ChainableDF:
    return operation(ChainableDF(self))


DataFrame.__rshift__ = dataframe_rshift  # type: ignore


def table(table_name: str) -> ChainableDF:
    spark = SparkSession.active()
    return ChainableDF(spark.table(table_name))

# Contents of functions.py





def with_columns(*cols: Column, batch: bool = False, **kwargs: Column) -> Callable:
    """
    Add one or more columns to the DataFrame.

    Args:
        *cols (Column): Columns to add.
        batch (bool, optional): Whether to add columns in batch. Defaults to False.
        **kwargs (Column): Additional columns to add as keyword arguments.

    Returns:
        Callable: A function that takes a ChainableDF and returns a modified ChainableDF.

    Examples:
        >>> from sparkkit.core import ChainableDF, with_columns
        >>> from pyspark.sql.functions import lit
        >>> chain = ChainableDF(df)
        >>> result = chain >> with_columns(new_col=lit(100))
    """

    def get_alias(col: Column) -> str | None:
        return (
            col._jc.toString().split(" AS ")[-1]  # type: ignore
            if " AS " in col._jc.toString()  # type: ignore
            else None
        )

    def _with_columns(chain: ChainableDF) -> ChainableDF:
        new_df = chain.df
        columns_dict = {}

        for col in cols:
            alias = get_alias(col)
            if not alias:
                raise ValueError(
                    "Column objects passed without aliases must have an alias."
                )
            columns_dict[alias] = col

        columns_dict.update(kwargs)

        if batch:
            new_df = new_df.withColumns(columns_dict)
        else:
            for name, col in columns_dict.items():
                new_df = new_df.withColumn(name, col)

        return ChainableDF(new_df, group_cols=chain.group_cols)

    return _with_columns


def select(*cols: str | Column | Selector, **renames: str) -> Callable:
    """
    Select specific columns from the DataFrame, with optional renaming.

    Args:
        *cols (str | Column | Selector): Columns to select. Can include Selectors for dynamic selection.
        **renames (str): Keyword arguments for renaming columns (new_name=old_name).

    Returns:
        Callable: A function that takes a ChainableDF and returns a modified ChainableDF.

    Examples:
        >>> from sparkkit.core import ChainableDF, select, starts_with
        >>> chain = ChainableDF(df)
        >>> result = chain >> select("id", starts_with("name"), new_name="value_alias")
    """

    def _select(chain: ChainableDF) -> ChainableDF:
        selected_cols: list[str | Column] = []
        for col in cols:
            if isinstance(col, Selector):
                selected_cols.extend(col(chain.df))
            else:
                selected_cols.append(col)

        for new_name, old_name in renames.items():
            selected_cols.append(f.col(old_name).alias(new_name))

        return ChainableDF(chain.df.select(*selected_cols), group_cols=chain.group_cols)

    return _select


def where(condition: str | Column) -> Callable:
    """
    Filter rows in the DataFrame based on a condition.

    Args:
        condition (str | Column): The condition to filter rows. Can be a string expression or a Column.

    Returns:
        Callable: A function that takes a ChainableDF and returns a filtered ChainableDF.

    Examples:
        >>> from sparkkit.core import ChainableDF, where
        >>> from pyspark.sql.functions import col
        >>> chain = ChainableDF(df)
        >>> result = chain >> where(col("age") > 30)

        # Using a string condition
        >>> result = chain >> where("age > 30")
    """

    def _where(chain: ChainableDF) -> ChainableDF:
        if isinstance(condition, Column):
            filtered_df = chain.df.filter(condition)
        else:
            filtered_df = chain.df.filter(condition)
        return ChainableDF(filtered_df, group_cols=chain.group_cols)

    return _where


def group_by(*columns: str | Column) -> Callable:
    """
    Group the DataFrame by one or more columns.

    Args:
        *columns (str | Column): Columns to group by.

    Returns:
        Callable: A function that takes a ChainableDF and returns a grouped ChainableDF.

    Examples:
        >>> from sparkkit.core import ChainableDF, group_by
        >>> chain = ChainableDF(df)
        >>> result = chain >> group_by("department", "role")
    """

    def _group_by(chain: ChainableDF) -> ChainableDF:
        group_cols = [c if isinstance(c, Column) else c for c in columns]
        return ChainableDF(chain.df, group_cols=group_cols)

    return _group_by


def summarize(**aggs: Column) -> Callable:
    """
    Aggregate the DataFrame using specified aggregation functions.

    Args:
        **aggs (Column): Aggregation expressions as keyword arguments where key is the alias.

    Returns:
        Callable: A function that takes a ChainableDF and returns an aggregated ChainableDF.

    Examples:
        >>> from sparkkit.core import ChainableDF, summarize
        >>> from pyspark.sql.functions import sum, avg
        >>> chain = ChainableDF(df)
        >>> result = chain >> summarize(total_sales=sum("sales"), average_age=avg("age"))
    """

    def _summarize(chain: ChainableDF) -> ChainableDF:
        if chain.group_cols:
            grouped_df = chain.df.groupBy(*chain.group_cols)
            new_df = grouped_df.agg(
                *[expr.alias(alias) for alias, expr in aggs.items()]
            )
        else:
            new_df = chain.df.agg(*[expr.alias(alias) for alias, expr in aggs.items()])
        return ChainableDF(new_df)

    return _summarize


def mutate(*cols: Column, batch: bool = False, **kwargs: Column) -> Callable:
    """
    Add or modify one or more columns in the DataFrame.

    Args:
        *cols (Column): Columns to add or modify.
        batch (bool, optional): Whether to add or modify columns in batch. Defaults to False.
        **kwargs (Column): Additional columns to add or modify as keyword arguments.

    Returns:
        Callable: A function that takes a ChainableDF and returns a modified ChainableDF.

    Examples:
        >>> from sparkkit.core import ChainableDF, mutate
        >>> from pyspark.sql.functions import lit, col
        >>> chain = ChainableDF(df)
        >>> result = chain >> mutate(new_col=lit(100), modified_col=col("existing_col") + 1)
    """

    def get_alias(col: Column) -> str:
        alias = col._jc.toString().split(" AS ")[-1] if " AS " in col._jc.toString() else None  # type: ignore
        if not alias:
            raise ValueError(
                "Column objects passed without aliases must have an alias."
            )
        return cast(str, alias)

    def _mutate(chain: ChainableDF) -> ChainableDF:
        new_df = chain.df
        columns_dict = {get_alias(col): col for col in cols}
        columns_dict.update(kwargs)

        if batch:
            new_df = new_df.withColumns(columns_dict)
        else:
            for name, col in columns_dict.items():
                new_df = new_df.withColumn(name, col)

        return ChainableDF(new_df, group_cols=chain.group_cols)

    return _mutate


def distinct(*cols: str | Column) -> Callable:
    """
    Remove duplicate rows from the DataFrame based on specified columns.

    Args:
        *cols (str | Column): Columns to consider for determining duplicates. If not provided, all columns are considered.

    Returns:
        Callable: A function that takes a ChainableDF and returns a distinct ChainableDF.

    Examples:
        >>> from sparkkit.core import ChainableDF, distinct
        >>> chain = ChainableDF(df)
        >>> result = chain >> distinct("id", "name")

        # Remove duplicates based on all columns
        >>> result = chain >> distinct()
    """

    def _distinct(chain: ChainableDF) -> ChainableDF:
        if cols:
            new_df = chain.df.select(*cols).distinct()
        else:
            new_df = chain.df.distinct()
        return ChainableDF(new_df, group_cols=chain.group_cols)

    return _distinct


def union(other: DataFrame | ChainableDF) -> Callable:
    """
    Perform a union of the current DataFrame with another DataFrame, removing duplicate rows.

    Args:
        other (DataFrame | ChainableDF): The DataFrame to union with.

    Returns:
        Callable: A function that takes a ChainableDF and returns a unioned ChainableDF.

    Examples:
        >>> from sparkkit.core import ChainableDF, union
        >>> chain = ChainableDF(df1)
        >>> other_chain = ChainableDF(df2)
        >>> result = chain >> union(other_chain)

        # Union with a regular DataFrame
        >>> result = chain >> union(df2)
    """
    return _union(other, distinct=True)


def union_all(other: DataFrame | ChainableDF) -> Callable:
    """
    Perform a union of the current DataFrame with another DataFrame, including duplicate rows.

    Args:
        other (DataFrame | ChainableDF): The DataFrame to union with.

    Returns:
        Callable: A function that takes a ChainableDF and returns a unioned ChainableDF.

    Examples:
        >>> from sparkkit.core import ChainableDF, union_all
        >>> chain = ChainableDF(df1)
        >>> other_chain = ChainableDF(df2)
        >>> result = chain >> union_all(other_chain)

        # Union all with a regular DataFrame
        >>> result = chain >> union_all(df2)
    """
    return _union(other, distinct=False)


def _union(other: DataFrame | ChainableDF, distinct: bool = False) -> Callable:
    def _union_op(chain: ChainableDF) -> ChainableDF:
        other_df = other.df if isinstance(other, ChainableDF) else other
        new_df = chain.df.unionByName(other_df)

        if distinct:
            new_df = new_df.distinct()

        other_groups = other.group_cols if isinstance(other, ChainableDF) else []
        combined_groups = list(set(chain.group_cols + other_groups))
        return ChainableDF(new_df, group_cols=combined_groups)

    return _union_op


def read(path: str, format: str, **options) -> ChainableDF:
    """
    Read data from a specified path in the given format.

    Args:
        path (str): The file path to read from.
        format (str): The format of the file (e.g., 'csv', 'parquet', 'delta').
        **options: Additional options for reading the data.

    Returns:
        ChainableDF: A ChainableDF object containing the loaded DataFrame.

    Examples:
        >>> from sparkkit.core import ChainableDF, read
        >>> chain = read("/path/to/data.csv", "csv", header=True, inferSchema=True)
    """
    spark = SparkSession.active()
    reader = spark.read.format(format)
    if options:
        reader = reader.options(**options)
    return ChainableDF(reader.load(path))


def read_csv(
    path: str, header: bool = True, infer_schema: bool = True, sep: str = ",", **options
) -> ChainableDF:
    """
    Read a CSV file into a ChainableDF.

    Args:
        path (str): The file path to read from.
        header (bool, optional): Whether the CSV file has a header row. Defaults to True.
        infer_schema (bool, optional): Whether to infer the schema of the CSV file. Defaults to True.
        sep (str, optional): The delimiter used in the CSV file. Defaults to ",".
        **options: Additional options for reading the CSV data.

    Returns:
        ChainableDF: A ChainableDF object containing the loaded DataFrame.

    Examples:
        >>> from sparkkit.core import ChainableDF, read_csv
        >>> chain = read_csv("/path/to/data.csv", header=True, infer_schema=True, sep=",")
    """
    options = {
        "header": str(header).lower(),
        "inferSchema": str(infer_schema).lower(),
        "sep": sep,
        **options,
    }
    return read(path, "csv", **options)


def read_parquet(path: str, **options) -> ChainableDF:
    """
    Read a Parquet file into a ChainableDF.

    Args:
        path (str): The file path to read from.
        **options: Additional options for reading the Parquet data.

    Returns:
        ChainableDF: A ChainableDF object containing the loaded DataFrame.

    Examples:
        >>> from sparkkit.core import ChainableDF, read_parquet
        >>> chain = read_parquet("/path/to/data.parquet")
    """
    return read(path, "parquet", **options)


def read_delta(path: str, **options) -> ChainableDF:
    """
    Read a Delta table into a ChainableDF.

    Args:
        path (str): The file path to read from.
        **options: Additional options for reading the Delta data.

    Returns:
        ChainableDF: A ChainableDF object containing the loaded DataFrame.

    Examples:
        >>> from sparkkit.core import ChainableDF, read_delta
        >>> chain = read_delta("/path/to/delta_table")
    """
    return read(path, "delta", **options)


def sample(
    n: int | None = None,
    fraction: float | None = None,
    replace: bool = False,
    seed: int | None = None,
) -> Callable:
    """
    Sample a subset of rows from the DataFrame.

    Args:
        n (int | None, optional): Number of rows to sample. If provided, `fraction` is ignored. Defaults to None.
        fraction (float | None, optional): Fraction of rows to sample. Defaults to None.
        replace (bool, optional): Whether to sample with replacement. Defaults to False.
        seed (int | None, optional): Seed for random sampling. Defaults to None.

    Returns:
        Callable: A function that takes a ChainableDF and returns a sampled ChainableDF.

    Examples:
        >>> from sparkkit.core import ChainableDF, sample
        >>> chain = ChainableDF(df)
        >>> result = chain >> sample(n=100, seed=42)

        >>> # Sample 10% of the data without replacement
        >>> result = chain >> sample(fraction=0.1)
    """

    def _sample(chain: ChainableDF) -> ChainableDF:
        total_count = chain.df.count()
        frac = fraction or (n / total_count if n is not None else 0.1)
        sampled_df = chain.df.sample(withReplacement=replace, fraction=frac, seed=seed)
        return ChainableDF(sampled_df, group_cols=chain.group_cols)

    return _sample


def sample_by(
    col: str | Column | list[str | Column], fractions: dict, seed: int | None = None
) -> Callable:
    """
    Sample rows from the DataFrame based on specified fractions per group.

    Args:
        col (str | Column | list[str | Column]): Column(s) to group by for sampling.
        fractions (dict): Dictionary specifying the fraction for each group.
        seed (int | None, optional): Seed for random sampling. Defaults to None.

    Returns:
        Callable: A function that takes a ChainableDF and returns a sampled ChainableDF.

    Examples:
        >>> from sparkkit.core import ChainableDF, sample_by
        >>> chain = ChainableDF(df)
        >>> fractions = {"group1": 0.1, "group2": 0.2}
        >>> result = chain >> sample_by("group_col", fractions, seed=42)

        >>> # Sampling based on multiple columns
        >>> result = chain >> sample_by(["group_col1", "group_col2"], fractions, seed=42)
    """

    def _sample_by(chain: ChainableDF) -> ChainableDF:
        if isinstance(col, list):
            exprs = [c if isinstance(c, Column) else f.col(c) for c in col]
            combined_col = exprs[0]
            for c in exprs[1:]:
                combined_col = combined_col + "|" + c
            mapped_fractions = {}
            for k, v in fractions.items():
                if isinstance(k, tuple):
                    mapped_fractions["|".join(map(str, k))] = v
                else:
                    mapped_fractions[str(k)] = v
            temp_df = chain.df.withColumn("_combined", combined_col)
            sampled_df = temp_df.sampleBy(
                "_combined", mapped_fractions, seed=seed
            ).drop("_combined")
        else:
            col_name = col if isinstance(col, str) else col._jc.toString()  # type: ignore
            sampled_df = chain.df.sampleBy(col_name, fractions, seed=seed)

        return ChainableDF(sampled_df, group_cols=chain.group_cols)

    return _sample_by


def count(*columns: str | Column) -> Callable:
    """
    Count the number of rows or group by columns and count.

    Args:
        *columns (str | Column): Columns to group by before counting. If not provided, counts all rows.

    Returns:
        Callable: A function that takes a ChainableDF and returns a ChainableDF with counts.

    Examples:
        >>> from sparkkit.core import ChainableDF, count
        >>> chain = ChainableDF(df)
        >>> # Count all rows
        >>> total = chain >> count()

        >>> # Count rows per group
        >>> result = chain >> count("group_col")
    """

    def _count(chain: ChainableDF):
        if not columns:
            return chain.df.count()
        else:
            group_cols = [c if isinstance(c, Column) else col(c) for c in columns]
            counted_df = chain.df.groupBy(*group_cols).count()
            return ChainableDF(counted_df, group_cols=chain.group_cols)

    return _count


def n() -> Column:
    """
    Return a column expression for counting all rows.

    Returns:
        Column: A PySpark Column expression representing a count of all rows.

    Examples:
        >>> from sparkkit.core import ChainableDF, summarize, n
        >>> from pyspark.sql.functions import col
        >>> chain = ChainableDF(df)
        >>> result = chain >> summarize(total=n())
    """
    return spark_count("*")


# Define the 'otherwise' sentinel
class Otherwise:
    def __rshift__(
        self, value: Column | str | int | float | bool
    ) -> tuple[Otherwise, Column | str | int | float | bool]:
        """
        Implement the right shift operator (>>) to support the case_when syntax.

        Args:
            value (Column | str | int | float | bool): The value to return when no 
                other conditions match.

        Returns:
            tuple[Otherwise, Column | str | int | float | bool]: A tuple containing
                this Otherwise instance and the value.
        """
        return (self, value)


otherwise = Otherwise()


def case_when(*pairs: tuple[Column, Column]) -> Column:
    """
    Create a case when expression based on given condition-value pairs.

    Args:
        *pairs (tuple[Column, Column]): Pairs of conditions and their corresponding values.

    Returns:
        Column: A PySpark Column expression representing the case when logic.

    Examples:
        >>> from sparkkit.core import ChainableDF, mutate, case_when, otherwise
        >>> from pyspark.sql.functions import when, col
        >>> chain = ChainableDF(df)
        >>> result = chain >> mutate(
        ...     status=case_when(
        ...         (col("age") > 18) >> "adult",
        ...         (col("age") <= 18) >> "minor",
        ...         otherwise >> "unknown"
        ...     )
        ... )
    """
    if not pairs:
        raise ValueError("case_when requires at least one (condition >> value) pair.")

    expr = f.when(pairs[0][0], pairs[0][1])
    for i, (condition, value) in enumerate(pairs[1:], start=1):
        if isinstance(condition, Otherwise):
            expr = expr.otherwise(value)
        else:
            expr = expr.when(condition, value)

    return expr


def pivot_longer(
    cols: str | Column | Selector | list[str | Column | Selector],
    names_to: str = "name",
    values_to: str = "value",
    names_prefix: str | None = None,
) -> Callable:
    """
    Pivot the DataFrame from wide to long format.

    Args:
        cols (str | Column | Selector | list[str | Column | Selector]): Columns to pivot.
        names_to (str, optional): Name of the new column for variable names. Defaults to "name".
        values_to (str, optional): Name of the new column for values. Defaults to "value".
        names_prefix (str | None, optional): Prefix to remove from original column names. Defaults to None.

    Returns:
        Callable: A function that takes a ChainableDF and returns a pivoted ChainableDF.

    Examples:
        >>> from sparkkit.core import ChainableDF, pivot_longer
        >>> chain = ChainableDF(df)
        >>> result = chain >> pivot_longer(cols=["Q1", "Q2", "Q3", "Q4"], names_to="quarter", values_to="sales")

        >>> # Using a prefix
        >>> result = chain >> pivot_longer(cols=starts_with("sales_"), names_to="region", values_to="amount", names_prefix="sales_")
    """

    def _pivot_longer(chain: ChainableDF) -> ChainableDF:
        if isinstance(cols, (str, Column)):
            selected_cols = [cols]
        elif isinstance(cols, Selector):
            selected_cols = cols(chain.df)
        else:
            selected_cols = []
            for col in cols:
                if isinstance(col, (str, Column)):
                    selected_cols.append(col)
                elif isinstance(col, Selector):
                    selected_cols.extend(col(chain.df))

        name_mapping = {}
        if names_prefix:
            for col in selected_cols:
                col_name = col if isinstance(col, str) else col._jc.toString()  # type: ignore
                if col_name.startswith(names_prefix):
                    name_mapping[col_name] = col_name[len(names_prefix) :]

        id_cols = [c for c in chain.df.columns if c not in selected_cols]

        exprs = [
            f.expr(
                f"stack({len(selected_cols)}, "
                + ", ".join(
                    [
                        f"'{name_mapping.get(col, col)}', `{col}`"
                        for col in selected_cols
                    ]
                )
                + f") as ({names_to}, {values_to})"
            )
        ]

        new_df = chain.df.select(*id_cols, *exprs)
        return ChainableDF(new_df, group_cols=chain.group_cols)

    return _pivot_longer


def pivot_wider(names_from: str, values_from: str, names_prefix: str = "") -> Callable:
    """
    Pivot the DataFrame from long to wide format.

    Args:
        names_from (str): Column to use for new column names.
        values_from (str): Column to use for new column values.
        names_prefix (str, optional): Prefix to add to new column names. Defaults to "".

    Returns:
        Callable: A function that takes a ChainableDF and returns a pivoted ChainableDF.

    Examples:
        >>> from sparkkit.core import ChainableDF, pivot_wider
        >>> chain = ChainableDF(df)
        >>> result = chain >> pivot_wider(names_from="quarter", values_from="sales", names_prefix="Q_")
    """

    def _pivot_wider(chain: ChainableDF) -> ChainableDF:
        id_cols = [c for c in chain.df.columns if c not in (names_from, values_from)]

        pivoted = chain.df.groupBy(id_cols).pivot(names_from).agg(f.first(values_from))

        if names_prefix:
            for col in pivoted.columns:
                if col not in id_cols:
                    pivoted = pivoted.withColumnRenamed(col, f"{names_prefix}{col}")

        return ChainableDF(pivoted, group_cols=chain.group_cols)

    return _pivot_wider


def back(chain: ChainableDF) -> DataFrame:
    """
    Retrieve the underlying PySpark DataFrame from a ChainableDF.

    Args:
        chain (ChainableDF): The ChainableDF instance.

    Returns:
        DataFrame: The underlying PySpark DataFrame.

    Examples:
        >>> from sparkkit.core import ChainableDF, back
        >>> chain = ChainableDF(df)
        >>> df_original = chain >> back()
    """
    return chain.df

# Contents of joins.py





class JoinType(Enum):
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    FULL = "full"
    SEMI = "semi"
    ANTI = "anti"
    CROSS = "cross"


class JoinDirection(Enum):
    LEFT = "left"
    RIGHT = "right"


def join(
    other: DataFrame | ChainableDF,
    by: str | list[str] | None = None,
    how: str | JoinType = JoinType.INNER,
    suffix: tuple[str, str] = ("_x", "_y"),
    dir: JoinDirection | str | None = None,
) -> Callable:
    def _join(chain: ChainableDF) -> ChainableDF:
        join_type = how.value if isinstance(how, JoinType) else how
        other_df = other.df if isinstance(other, ChainableDF) else other

        if join_type in ("semi", "anti") and dir:
            direction = dir.value if isinstance(dir, JoinDirection) else dir
            join_type = f"{direction}_{join_type}"

        if join_type == "cross":
            new_df = chain.df.crossJoin(other_df)
        else:
            new_df = chain.df.join(other_df, on=by, how=join_type)
            if by is not None and join_type not in ("semi", "anti"):
                join_cols = [by] if isinstance(by, str) else by
                overlap = set(chain.df.columns) & set(other_df.columns) - set(join_cols)
                for col in overlap:
                    new_df = new_df.withColumnRenamed(f"{col}", f"{col}{suffix[0]}")

        other_groups = other.group_cols if isinstance(other, ChainableDF) else []
        combined_groups = list(set(chain.group_cols + other_groups))
        return ChainableDF(new_df, group_cols=combined_groups)

    return _join


def inner_join(
    other: DataFrame | ChainableDF,
    by: str | list[str] | None = None,
    suffix: tuple[str, str] = ("_x", "_y"),
) -> Callable:
    return join(other, by=by, how=JoinType.INNER, suffix=suffix)


def left_join(
    other: DataFrame | ChainableDF,
    by: str | list[str] | None = None,
    suffix: tuple[str, str] = ("_x", "_y"),
) -> Callable:
    return join(other, by=by, how=JoinType.LEFT, suffix=suffix)


def right_join(
    other: DataFrame | ChainableDF,
    by: str | list[str] | None = None,
    suffix: tuple[str, str] = ("_x", "_y"),
) -> Callable:
    return join(other, by=by, how=JoinType.RIGHT, suffix=suffix)


def full_join(
    other: DataFrame | ChainableDF,
    by: str | list[str] | None = None,
    suffix: tuple[str, str] = ("_x", "_y"),
) -> Callable:
    return join(other, by=by, how=JoinType.FULL, suffix=suffix)


def semi_join(
    other: DataFrame | ChainableDF,
    by: str | list[str] | None = None,
    dir: JoinDirection | str = JoinDirection.LEFT,
) -> Callable:
    return join(other, by=by, how=JoinType.SEMI, dir=dir)


def anti_join(
    other: DataFrame | ChainableDF,
    by: str | list[str] | None = None,
    dir: JoinDirection | str = JoinDirection.LEFT,
) -> Callable:
    return join(other, by=by, how=JoinType.ANTI, dir=dir)


def cross_join(other: DataFrame | ChainableDF) -> Callable:
    return join(other, how=JoinType.CROSS)


def as_of_join(
    other: DataFrame | ChainableDF,
    by: str | list[str],
    time_col_left: str,
    time_col_right: str | None = None,
    suffix: tuple[str, str] = ("_x", "_y"),
    inclusive: bool = True,
) -> Callable:
    def _as_of_join(chain: ChainableDF) -> ChainableDF:
        right_df = other.df if isinstance(other, ChainableDF) else other
        left_df = chain.df

        effective_time_col_right = time_col_right or time_col_left

        partition_cols = [by] if isinstance(by, str) else by

        left_alias = "l"
        right_alias = "r"

        joined = left_df.alias(left_alias).join(
            right_df.alias(right_alias),
            on=partition_cols,
            how="left",
        )

        op = "<=" if inclusive else "<"
        condition = expr(
            f"{right_alias}.{effective_time_col_right} {op} {left_alias}.{time_col_left}"
        )
        filtered = joined.filter(condition)

        window_spec = Window.partitionBy(
            [col(f"{left_alias}.{c}") for c in partition_cols]
        ).orderBy(col(f"{right_alias}.{effective_time_col_right}").desc())

        filtered_with_rn = filtered.withColumn(
            "rn_asof", f.row_number().over(window_spec)
        )
        picked = filtered_with_rn.filter("rn_asof = 1").drop("rn_asof")

        overlap = set(left_df.columns).intersection(set(right_df.columns))
        overlap -= set(partition_cols)

        overlap -= {time_col_left}
        if time_col_right and time_col_right in overlap:
            overlap -= {time_col_right}

        new_df = picked
        for col_name in overlap:
            new_df = new_df.withColumnRenamed(
                f"{right_alias}.{col_name}", f"{col_name}{suffix[1]}"
            )

        if time_col_right and time_col_right != time_col_left:
            new_df = new_df.withColumnRenamed(
                f"{right_alias}.{time_col_right}", f"{time_col_right}{suffix[1]}"
            )

        for c in left_df.columns:
            old_name = f"{left_alias}.{c}"
            if old_name in new_df.columns:
                new_df = new_df.withColumnRenamed(old_name, c)

        for c in right_df.columns:
            old_name = f"{right_alias}.{c}"
            if old_name in new_df.columns:
                new_name = f"{c}{suffix[1]}" if c not in partition_cols else c
                new_df = new_df.withColumnRenamed(old_name, new_name)

        other_groups = other.group_cols if isinstance(other, ChainableDF) else []
        combined_groups = list(set(chain.group_cols + other_groups))
        return ChainableDF(new_df, group_cols=combined_groups)

    return _as_of_join
# Contents of time_series.py





def resample(date_col: str | Column, freq: str) -> Callable:
    """
    Resample the DataFrame based on a date column and frequency.

    Args:
        date_col (str | Column): The date column to resample on.
        freq (str): The frequency to resample by (e.g., '1d' for daily, '1h' for hourly).

    Returns:
        Callable: A function that takes a ChainableDF and returns a resampled ChainableDF.

    Examples:
        >>> from sparkkit.core import ChainableDF, resample
        >>> chain = ChainableDF(df)
        >>> result = chain >> resample("event_date", "1d")
        
        >>> # Resample hourly
        >>> result = chain >> resample("timestamp", "1h")
    """
    def _resample(chain: ChainableDF) -> ChainableDF:
        window_duration = parse_freq(freq)

        col_expr = date_col if isinstance(date_col, Column) else f.col(date_col)
        timestamp_col = f.to_timestamp(col_expr)

        window_col = f.window(timestamp_col, window_duration)

        new_df = chain.df.withColumn("_window", window_col)

        return ChainableDF(new_df, group_cols=["_window"])

    return _resample


def as_of_join(
    other: DataFrame | ChainableDF,
    by: str | list[str],
    time_col_left: str,
    time_col_right: str | None = None,
    suffix: tuple[str, str] = ("_x", "_y"),
    inclusive: bool = True,
) -> Callable:
    """
    Perform an as-of join between two DataFrames based on timing conditions.

    This function merges the calling DataFrame with the `other` DataFrame using a left 
    join on the specified key columns (`by`). After joining, it filters the resulting 
    DataFrame to include only the rows where the `other` DataFrame's time column 
    (`time_col_right`) is less than or equal to (if `inclusive` is True) or strictly 
    less than (if `inclusive` is False) the calling DataFrame's time column 
    (`time_col_left`). For each group defined by the `by` columns, only the most recent
    matching row from the `other` DataFrame is retained.

    Args:
        other (DataFrame | ChainableDF): The right DataFrame to join with.
        by (str | list[str]): Column(s) to join on. These columns must exist in both 
            DataFrames.
        time_col_left (str): Time column from the left (calling) DataFrame.
        time_col_right (str | None, optional): Time column from the right (`other`) 
            DataFrame. If not provided, `time_col_left` is used. Defaults to None.
        suffix (tuple[str, str], optional): Suffixes to append to overlapping column 
            names from the left and right DataFrames, respectively. Defaults to ("_x", 
            "_y").
        inclusive (bool, optional): Determines the join condition. If True, includes 
            rows where `other.time_col_right` <= `left.time_col_left`. If False, only 
            includes rows where `other.time_col_right` < `left.time_col_left`. Defaults 
            to True.

    Returns:
        Callable: A function that takes a `ChainableDF` and returns a joined 
            `ChainableDF` after performing the as-of join.

    Examples:
        >>> from sparkkit.core import ChainableDF, as_of_join
        >>> chain_left = ChainableDF(df_left)
        >>> chain_right = ChainableDF(df_right)
        >>> result = chain_left >> as_of_join(
        ...     chain_right, 
        ...     by="id", 
        ...     time_col_left="timestamp", 
        ...     inclusive=True
        ... )

        >>> # Using different time columns and custom suffix
        >>> result = chain_left >> as_of_join(
        ...     chain_right,
        ...     by=["id", "category"],
        ...     time_col_left="start_time",
        ...     time_col_right="end_time",
        ...     suffix=("_left", "_right"),
        ...     inclusive=False
        ... )
    """
    def _as_of_join(chain: ChainableDF) -> ChainableDF:
        right_df = other.df if isinstance(other, ChainableDF) else other
        left_df = chain.df

        effective_time_col_right = time_col_right or time_col_left

        partition_cols = [by] if isinstance(by, str) else by

        left_alias = "l"
        right_alias = "r"

        joined = left_df.alias(left_alias).join(
            right_df.alias(right_alias),
            on=partition_cols,
            how="left",
        )

        op = "<=" if inclusive else "<"
        condition = expr(
            f"{right_alias}.{effective_time_col_right} {op} {left_alias}.{time_col_left}"
        )
        filtered = joined.filter(condition)

        window_spec = Window.partitionBy(
            [col(f"{left_alias}.{c}") for c in partition_cols]
        ).orderBy(col(f"{right_alias}.{effective_time_col_right}").desc())

        filtered_with_rn = filtered.withColumn(
            "rn_asof", f.row_number().over(window_spec)
        )
        picked = filtered_with_rn.filter("rn_asof = 1").drop("rn_asof")

        overlap = set(left_df.columns).intersection(set(right_df.columns))
        overlap -= set(partition_cols)

        overlap -= {time_col_left}
        if time_col_right and time_col_right in overlap:
            overlap -= {time_col_right}

        new_df = picked
        for col_name in overlap:
            new_df = new_df.withColumnRenamed(
                f"{right_alias}.{col_name}", f"{col_name}{suffix[1]}"
            )

        if time_col_right and time_col_right != time_col_left:
            new_df = new_df.withColumnRenamed(
                f"{right_alias}.{time_col_right}", f"{time_col_right}{suffix[1]}"
            )

        for c in left_df.columns:
            old_name = f"{left_alias}.{c}"
            if old_name in new_df.columns:
                new_df = new_df.withColumnRenamed(old_name, c)

        for c in right_df.columns:
            old_name = f"{right_alias}.{c}"
            if old_name in new_df.columns:
                new_name = f"{c}{suffix[1]}" if c not in partition_cols else c
                new_df = new_df.withColumnRenamed(old_name, new_name)

        other_groups = other.group_cols if isinstance(other, ChainableDF) else []
        combined_groups = list(set(chain.group_cols + other_groups))
        return ChainableDF(new_df, group_cols=combined_groups)

    return _as_of_join


def parse_freq(freq: str) -> str:
    """
    Parse a frequency string into a window duration string.

    Args:
        freq (str): The frequency string (e.g., '1d', '2h', '30min').

    Returns:
        str: A window duration string compatible with PySpark's window functions.

    Raises:
        ValueError: If the frequency format is invalid or the unit is unsupported.
    """
    unit_map = {
        "s": "seconds",
        "min": "minutes",
        "h": "hours",
        "d": "days",
        "w": "weeks",
        "M": "months",
        "y": "years",
    }

    match = re.match(r"(\d+)([a-zA-Z]+)", freq)
    if not match:
        raise ValueError(f"Invalid frequency format: {freq}")

    number, unit = match.groups()
    if unit not in unit_map:
        raise ValueError(f"Unsupported frequency unit: {unit}")

    return f"{number} {unit_map[unit]}"

