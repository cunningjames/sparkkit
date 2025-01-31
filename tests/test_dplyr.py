import os
import sys
import tempfile
from pathlib import Path

import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from sparkkit import (
    anti_join,
    cross_join,
    full_join,
    inner_join,
    left_join,
    otherwise,
    right_join,
    semi_join,
)
from sparkkit.core import ChainableDF, contains, starts_with
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
from sparkkit.time_series import as_of_join, parse_freq, resample

if os.name == "nt":
    os.environ["PYSPARK_PYTHON"] = sys.executable


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing"""
    return (
        SparkSession.builder.master("local[1]").appName("sparkkit-tests").getOrCreate()
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
        doubled=f.col("value") * 2, constant=f.lit(42)
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
    schema = StructType(
        [
            StructField("id", IntegerType(), False),
            StructField("name", StringType(), False),
            StructField("value", IntegerType(), False),
        ]
    )
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


@pytest.fixture
def time_series_data(spark):
    """Create time series test data"""
    left_data = [
        (1, "2023-01-01 10:00:00", 100),
        (1, "2023-01-01 11:00:00", 200),
        (2, "2023-01-01 10:30:00", 300),
    ]
    right_data = [
        (1, "2023-01-01 09:55:00", "A"),
        (1, "2023-01-01 10:55:00", "B"),
        (2, "2023-01-01 10:25:00", "C"),
    ]
    left_df = spark.createDataFrame(left_data, ["id", "timestamp", "value"])
    right_df = spark.createDataFrame(right_data, ["id", "timestamp", "label"])
    return left_df, right_df

@pytest.fixture
def time_series_resample_data(spark):
    """Create time series data for resampling tests"""
    data = [
        (1, "2023-01-01 10:00:00", 100),
        (1, "2023-01-01 10:15:00", 150),
        (1, "2023-01-01 10:30:00", 200),
        (2, "2023-01-01 10:05:00", 300),
        (2, "2023-01-01 10:35:00", 400),
    ]
    return spark.createDataFrame(data, ["id", "timestamp", "value"])


def test_as_of_join_basic(time_series_data):
    """Test basic as-of join functionality"""
    left_df, right_df = time_series_data
    chain = ChainableDF(left_df)

    result = chain >> as_of_join(
        right_df, by="id", time_col_left="timestamp", inclusive=True
    )

    rows = result.df.collect()
    assert len(rows) == 3

    # Verify joined data
    expected = [
        (1, "2023-01-01 10:00:00", 100, "A"),
        (1, "2023-01-01 11:00:00", 200, "B"),
        (2, "2023-01-01 10:30:00", 300, "C"),
    ]

    for row, (id, ts, val, label) in zip(rows, expected):
        assert row.id == id
        assert str(row.timestamp_x) == ts
        assert row.value == val
        assert row.label == label


def test_resample_basic(time_series_resample_data):
    """Test basic resampling functionality"""
    chain = ChainableDF(time_series_resample_data)
    
    result = chain >> resample("timestamp", "15min")
    
    # Verify window column was added
    assert "_window" in result.df.columns
    
    # Verify group_cols is set correctly
    assert result.group_cols == ["_window"]
    
    # Verify window start times
    windows = result.df.select("_window.start").distinct().collect()
    expected_starts = [
        "2023-01-01 10:00:00",
        "2023-01-01 10:15:00",
        "2023-01-01 10:30:00"
    ]
    assert sorted([str(w.start) for w in windows]) == sorted(expected_starts)

def test_resample_column_input(time_series_resample_data):
    """Test resample with Column input for date_col"""
    chain = ChainableDF(time_series_resample_data)
    
    result = chain >> resample(f.col("timestamp"), "15min")
    
    # Verify window column was added
    assert "_window" in result.df.columns
    assert result.group_cols == ["_window"]

def test_resample_invalid_frequency(time_series_resample_data):
    """Test resample with invalid frequency string"""
    chain = ChainableDF(time_series_resample_data)
    
    with pytest.raises(ValueError):
        chain >> resample("timestamp", "15 bananas")

def test_as_of_join_different_time_columns(time_series_data):
    """Test as-of join with different time columns"""
    left_df, right_df = time_series_data
    chain = ChainableDF(left_df)

    # Rename right timestamp column
    right_df = right_df.withColumnRenamed("timestamp", "event_time")

    result = chain >> as_of_join(
        right_df,
        by="id",
        time_col_left="timestamp",
        time_col_right="event_time",
        inclusive=True,
    )

    rows = result.df.collect()
    assert len(rows) == 3

    # Verify joined data
    expected = [
        (1, "2023-01-01 10:00:00", 100, "A"),
        (1, "2023-01-01 11:00:00", 200, "B"),
        (2, "2023-01-01 10:30:00", 300, "C"),
    ]

    for row, (id, ts, val, label) in zip(rows, expected):
        assert row.id == id
        assert str(row.timestamp) == ts
        assert row.value == val
        assert row.label == label


def test_parse_freq_invalid_unit():
    """Test parse_freq raises ValueError on invalid unit"""
    with pytest.raises(ValueError):
        parse_freq("10x")

def test_left_join(chainable_df, spark):
    """Test left join operation"""
    other_data = [
        (1, "X"),
        (3, "Y"),
    ]
    other_df = spark.createDataFrame(other_data, ["id", "letter"])

    result = chainable_df >> left_join(other_df, by="id")

    rows = result.df.collect()
    assert len(rows) == 4  # All left DataFrame rows
    assert all(hasattr(row, "letter") for row in rows)

def test_right_join(chainable_df, spark):
    """Test right join operation"""
    other_data = [
        (1, "X"),
        (3, "Y"),
    ]
    other_df = spark.createDataFrame(other_data, ["id", "letter"])

    result = chainable_df >> right_join(other_df, by="id")

    rows = result.df.collect()
    assert len(rows) == 2  # Only matching right DataFrame rows
    assert all(hasattr(row, "group") for row in rows)

def test_full_join(chainable_df, spark):
    """Test full join operation"""
    other_data = [
        (1, "X"),
        (3, "Y"),
    ]
    other_df = spark.createDataFrame(other_data, ["id", "letter"])

    result = chainable_df >> full_join(other_df, by="id")

    rows = result.df.collect()
    assert len(rows) == 4  # Union of both DataFrames
    # Additional assertions can be added as needed

def test_semi_join(chainable_df, spark):
    """Test semi join operation"""
    other_data = [
        (1, "X"),
        (3, "Y"),
    ]
    other_df = spark.createDataFrame(other_data, ["id", "letter"])

    result = chainable_df >> semi_join(other_df, by="id")

    rows = result.df.collect()
    assert len(rows) == 2  # Only matching left DataFrame rows

def test_anti_join(chainable_df, spark):
    """Test anti join operation"""
    other_data = [
        (1, "X"),
        (3, "Y"),
    ]
    other_df = spark.createDataFrame(other_data, ["id", "letter"])

    result = chainable_df >> anti_join(other_df, by="id")

    rows = result.df.collect()
    assert len(rows) == 2  # Non-matching left DataFrame rows

def test_cross_join(chainable_df, spark):
    """Test cross join operation"""
    other_data = [
        (100,),
        (200,),
    ]
    other_df = spark.createDataFrame(other_data, ["extra"])

    result = chainable_df >> cross_join(other_df)

    rows = result.df.collect()
    assert len(rows) == 8  # Cartesian product of both DataFrames

def test_join_overlapping_columns(chainable_df, spark):
    """Test join operation with overlapping non-key columns"""
    # Create left DataFrame with overlapping column 'group'
    left_data = [
        (1, "A_left", 10),
        (2, "B_left", 20),
    ]
    left_df = spark.createDataFrame(left_data, ["id", "group", "value"])

    # Create right DataFrame with overlapping column 'group'
    right_data = [
        (1, "A_right", 100),
        (3, "C_right", 300),
    ]
    right_df = spark.createDataFrame(right_data, ["id", "group", "extra"])

    # Create ChainableDF instance for left DataFrame
    chain = ChainableDF(left_df)

    # Perform inner join on 'id'
    result = chain >> inner_join(right_df, by="id")

    # Collect results
    rows = result.df.collect()

    # Verify the number of joined rows
    assert len(rows) == 1  # Only matching 'id' == 1

    # Verify that overlapping 'group' column from right_df is renamed
    assert "group_x" in result.df.columns
    assert "group_y" in result.df.columns

    # Verify the values of the renamed columns
    joined_row = rows[0]
    assert joined_row.group_x == "A_left"
    assert joined_row.group_y == "A_right"
    assert joined_row.extra == 100

