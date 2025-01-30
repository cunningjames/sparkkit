from __future__ import annotations

import re
from typing import Callable

import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from pyspark.sql.column import Column
from pyspark.sql.functions import col, expr
from pyspark.sql.window import Window

from .core import ChainableDF


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
