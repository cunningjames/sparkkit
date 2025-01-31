from __future__ import annotations

import re
from typing import Callable, cast

import pyspark.sql.functions as f
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.column import Column
from pyspark.sql.functions import col
from pyspark.sql.functions import count as spark_count
from pyspark.sql.functions import expr, lit, when
from pyspark.sql.window import Window

from .core import ChainableDF, Selector


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
