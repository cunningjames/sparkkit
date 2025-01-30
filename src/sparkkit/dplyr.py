from __future__ import annotations

from typing import Any, Callable

import pyspark.sql.functions as f
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.column import Column

from .core import ChainableDF


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
