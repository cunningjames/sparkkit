import tempfile
from pathlib import Path

import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, functions as f
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

from sparkkit import inner_join, otherwise

from sparkkit.core import (
    ChainableDF,
    contains,
    starts_with,
)

from sparkkit.functions import (
    case_when,
    distinct,
    group_by,
    read,
    read_csv,
    read_delta,
    read_parquet,
    sample,
    sample_by,
    select,
    summarize,
    where,
    with_columns,
)

import os
import sys


if os.name == 'nt':
    os.environ["PYSPARK_PYTHON"] = sys.executable


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing"""
    return (
        SparkSession.builder
        .master("local[1]")
        .appName("sparkkit-tests")
        .getOrCreate()
    )


@pytest.fixture
def sample_df(spark):
    """Create a sample DataFrame for testing"""
    data = [
        (1, "A", 10, 100),
        (2, "B", 20, 200),
        (3, "A", 30, 300),
        (4, "B", 40, 400),
    ]
    return spark.createDataFrame(data, ["id", "group", "value", "extra"])


@pytest.fixture
def chainable_df(sample_df):
    """Create a ChainableDF instance for testing"""
    return ChainableDF(sample_df)


def test_chainable_df_init(sample_df):
    """Test ChainableDF initialization"""
    chain = ChainableDF(sample_df)
    assert isinstance(chain.df, type(sample_df))
    assert chain.group_cols == []


def test_chainable_df_rshift(chainable_df):
    """Test the >> operator functionality"""
    result = chainable_df >> select("id", "value")
    assert isinstance(result, ChainableDF)
    assert result.df.columns == ["id", "value"]


def test_select_basic(chainable_df):
    """Test basic select operation with column names"""
    result = chainable_df >> select("id", "value")
    assert result.df.columns == ["id", "value"]


def test_select_with_selector(chainable_df):
    """Test select with column selectors"""
    # Test starts_with selector
    result = chainable_df >> select(starts_with("val"))
    assert result.df.columns == ["value"]

    # Test contains selector
    result = chainable_df >> select(contains("u"))
    assert result.df.columns == ["group", "value"]


def test_where_filter(chainable_df):
    """Test where filtering operation"""
    result = chainable_df >> where("value > 20")
    rows = result.df.collect()
    assert len(rows) == 2
    assert all(row.value > 20 for row in rows)


def test_group_by_summarize(chainable_df):
    """Test group by and summarize operations"""
    result = (
        chainable_df
        >> group_by("group")
        >> summarize(
            count=f.count("*"),
            avg_value=f.avg("value"),
        )
    )
    
    rows = result.df.collect()
    assert len(rows) == 2  # Two groups: A and B
    
    # Verify aggregations
    for row in rows:
        if row.group == "A":
            assert row["count"] == 2
            assert row.avg_value == 20.0  # (10 + 30) / 2
        elif row.group == "B":
            assert row["count"] == 2
            assert row.avg_value == 30.0  # (20 + 40) / 2


def test_with_columns(chainable_df):
    """Test adding new columns"""
    result = chainable_df >> with_columns(
        doubled=f.col("value") * 2,
        constant=f.lit(42)
    )
    
    # Check new columns exist
    assert "doubled" in result.df.columns
    assert "constant" in result.df.columns
    
    # Verify calculations
    rows = result.df.collect()
    for row in rows:
        assert row.doubled == row.value * 2
        assert row.constant == 42


def test_inner_join(chainable_df, spark):
    """Test inner join operation"""
    # Create a second DataFrame for joining
    other_data = [
        (1, "X"),
        (3, "Y"),
    ]
    other_df = spark.createDataFrame(other_data, ["id", "letter"])
    
    result = chainable_df >> inner_join(other_df, by="id")
    
    rows = result.df.collect()
    assert len(rows) == 2  # Only matching IDs (1 and 3)
    assert all(row.id in [1, 3] for row in rows)
    assert all(hasattr(row, "letter") for row in rows)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_read_csv(spark, temp_dir):
    """Test reading CSV files"""
    # Create test CSV file
    csv_path = temp_dir / "test.csv"
    data = "id,name,value\n1,A,10\n2,B,20\n"
    csv_path.write_text(data)
    
    # Read using read_csv with infer_schema=False to keep values as strings
    df = read_csv(str(csv_path), infer_schema=False)
    rows = df.df.collect()
    
    assert len(rows) == 2
    assert df.df.columns == ["id", "name", "value"]
    assert rows[0].name == "A"
    assert rows[1].value == "20"  # String because inferSchema=True is not set


def test_read_csv_with_options(spark, temp_dir):
    """Test reading CSV files with options"""
    # Create test CSV file
    csv_path = temp_dir / "test.csv"
    data = "id|name|value\n1|A|10\n2|B|20\n"
    csv_path.write_text(data)
    
    # Read using read_csv with custom separator
    df = read_csv(str(csv_path), sep="|", infer_schema=True)
    rows = df.df.collect()
    
    assert len(rows) == 2
    assert rows[0].value == 10  # Integer because inferSchema=True
    assert rows[1].value == 20


def test_read_parquet(spark, temp_dir):
    """Test reading Parquet files"""
    # Create test DataFrame
    schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("value", IntegerType(), False)
    ])
    data = [(1, "A", 10), (2, "B", 20)]
    test_df = spark.createDataFrame(data, schema)
    
    # Write to parquet
    parquet_path = str(temp_dir / "test.parquet")
    test_df.write.parquet(parquet_path)
    
    # Read using read_parquet
    df = read_parquet(parquet_path)
    rows = df.df.collect()
    
    assert len(rows) == 2
    assert df.df.columns == ["id", "name", "value"]
    assert rows[0].value == 10
    assert rows[1].name == "B"


@pytest.mark.skip(reason="Delta Lake not configured")
def test_read_delta(spark, temp_dir):
    """Test reading Delta tables - skipped until Delta Lake is properly configured"""
    pass


def test_read_generic(spark, temp_dir):
    """Test generic read function"""
    # Create test CSV file
    csv_path = temp_dir / "test.csv"
    data = "id,name,value\n1,A,10\n2,B,20\n"
    csv_path.write_text(data)
    
    # Read using generic read
    df = read(str(csv_path), "csv", header="true")
    rows = df.df.collect()
    
    assert len(rows) == 2
    assert df.df.columns == ["id", "name", "value"]


def test_case_when(chainable_df):
    """Test case_when functionality"""
    result = chainable_df >> with_columns(
        category=case_when(
            (f.col("value") > 30) >> "high",
            (f.col("value") > 20) >> "medium",
            otherwise >> "low",
        )
    )
    
    rows = result.df.collect()
    assert len(rows) == 4
    
    # Verify categorization
    categories = {row.value: row.category for row in rows}
    assert categories[10] == "low"
    assert categories[20] == "low"
    assert categories[30] == "medium"
    assert categories[40] == "high"


def test_sample(chainable_df):
    """Test sampling with a fraction"""
    original_count = chainable_df.df.count()
    sampled_df = chainable_df >> sample(fraction=0.5, seed=42)
    sampled_count = sampled_df.df.count()
    assert sampled_count <= original_count
    # Simple check: sampling is nondeterministic, but should be fewer or equal.

def test_sample_by(chainable_df):
    """Test sampling by a column with different fractions"""
    fractions = {"A": 0.5, "B": 1.0}
    sampled_df = chainable_df >> sample_by("group", fractions=fractions, seed=42)
    # Get counts per group
    counts_df = sampled_df.df.groupBy("group").count()
    counts = {row.group: row["count"] for row in counts_df.collect()}
    # B group should be complete
    assert counts.get("B", 0) == 2
    # A group should be reduced
    assert counts.get("A", 0) <= 1
