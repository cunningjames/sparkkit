# SparkKit

A dplyr-inspired interface for PySpark with chainable DataFrame operations.

## Installation

```bash
pip install sparkkit
```

For development:

```bash
pip install -e ".[dev,test]"
```

## Usage

Basic operations:

```python
from sparkkit import ChainableDF
import pyspark.sql.functions as f

df = ChainableDF(spark_df)

# Resample time series
df >> resample(date_col="timestamp", freq="1d")

# Group and aggregate
df >> group_by("category") >> summarize(total=f.sum("amount"))
```

## Development

```bash
pip install -e ".[dev,test]"
pytest
```

## License

MIT License
