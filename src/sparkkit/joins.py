from __future__ import annotations

from enum import Enum
from typing import Any, Callable

import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from pyspark.sql.column import Column
from pyspark.sql.functions import col, expr
from pyspark.sql.window import Window

from .core import ChainableDF


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