# Reproducibility

Paper reproduction verifies frozen hashes before loading data. It validates
condition/trial coverage, derives RQ1 and RQ2 winners from validation results,
and derives RQ3 from raw fixed-refit predictions. No HPO or training runs.

The historical RQ3 implementation did not explicitly enforce the lower
held-out test-period boundary when constructing the common forecast-origin
surface. Archived derived products therefore include 30 pre-test origins from
June 2015. The manuscript applies the declared held-out interval beginning
July 1, 2015. This implementation enforces exactly 2,926 origins and reproduces
all manuscript RQ3 values from frozen row-level predictions.

The dependency lock describes the current tested reproduction environment. It
does not claim to reconstruct the historical CUDA, driver, or package stack,
which was not completely archived.

