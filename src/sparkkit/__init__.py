"""
SparkKit: A dplyr-inspired interface for PySpark
"""

import sys
import warnings

if sys.version_info >= (3, 12):
    warnings.warn(
        "Python 3.12 is not officially supported by SparkKit as PySpark does not "
        "support this version yet. Some functionalities may not work as expected.",
        DeprecationWarning,
    )

from .core import (
    # Core DataFrame operations
    ChainableDF,
    
    # Column selectors
    starts_with,
    ends_with,
    contains,
    matches,
    num_range,
    everything,
)

from .dplyr import (
    # Display helper
    display,
    table,
)

from .joins import (
    # Join operations
    join,
    inner_join,
    left_join,
    right_join,
    full_join,
    semi_join,
    anti_join,
    cross_join,
)

from .functions import (
    # Core DataFrame operations
    with_columns,
    select,
    where,
    group_by,
    summarize,
    mutate,
    distinct,
    union,
    union_all,
    
    # Column operations
    case_when,
    
    # Pivot operations
    pivot_longer,
    pivot_wider,
    
    # Sampling
    sample,
    sample_by,
    
    # Aggregation helpers
    count,
    n,
    
    # I/O operations
    read,
    read_csv,
    read_parquet,
    read_delta,
    
    otherwise,
)

from .time_series import (
    # Time series operations
    resample,
    as_of_join,
)

__version__ = "0.1.0"
