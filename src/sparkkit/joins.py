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
    if by is not None and not isinstance(by, list):
        by = [by]

    def _join(chain: ChainableDF) -> ChainableDF:
        join_type = how.value if isinstance(how, JoinType) else how
        other_df = other.df if isinstance(other, ChainableDF) else other

        if join_type in ("semi", "anti") and dir:
            direction = dir.value if isinstance(dir, JoinDirection) else dir
            join_type = f"{direction}_{join_type}"

        if join_type == "cross":
            new_df = chain.df.crossJoin(other_df)
        else:
            if by is not None and join_type not in ("semi", "anti"):
                overlap = set(chain.df.columns) & set(other_df.columns) - set(by)
                df = chain.df
                for col in overlap:
                    other_df = other_df.withColumnRenamed(f"{col}", f"{col}{suffix[1]}")
                    df = df.withColumnRenamed(f"{col}", f"{col}{suffix[0]}")
                chain = ChainableDF(df)

            new_df = chain.df.join(other_df, on=by, how=join_type)

            # if by is not None and join_type not in ("semi", "anti"):
            #     join_cols = [by] if isinstance(by, str) else by
            #     overlap = set(chain.df.columns) & set(other_df.columns) - set(join_cols)
            #     for col in overlap:
            #         new_df = new_df.withColumnRenamed(f"{col}", f"{col}{suffix[0]}")

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
