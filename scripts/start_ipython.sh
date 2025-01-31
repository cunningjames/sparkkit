#!/usr/bin/env zsh

TMPFILE=$(mktemp "/tmp/ipython_startup.$RANDOM.$(date +%s).XXXXXX.py")

cat << 'EOF' > $TMPFILE
from datetime import date, datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from sparkkit import *
import pyspark.sql.functions as f

spark = SparkSession.builder.getOrCreate()

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
left = spark.createDataFrame(left_data, ["id", "timestamp", "value"])
right = spark.createDataFrame(right_data, ["id", "timestamp", "label"])

def display_(self, n: int = 20) -> None:
    return self.show(n=n)

DataFrame.display = display_
EOF

# Ensure the temporary file is deleted once the script exits, 
# whether IPython is exited normally or the script is interrupted.
trap 'rm -f "$TMPFILE"' EXIT

ipython -i $TMPFILE
