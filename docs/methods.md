# Scientific methods and contracts

The target is the causal 30-day trailing mean of Lake Superior water level,
`avg_wl_lake_30`. Inputs never include values after forecast origin.

Three input views are evaluated: daily smoothed target with raw exogenous
variables, daily smoothed target with smoothed exogenous variables, and a
30-day-spaced smoothed view. Contexts are 180, 360, 540, and 720 days; sparse
views therefore contain 6, 12, 18, and 24 observations.

Outputs comprise six independently trained point forecasts at 30, 60, 90,
120, 150, and 180 days and one six-output model at the same leads. Conditions
span four NeuralForecast architectures, giving 336 immutable conditions.

The canonical chronological split is 70/10/20 by target date. Model selection
uses only the minimum validation RMSE. Selected configurations are refit from
scratch on train plus validation before test evaluation.

RQ3 thresholds are horizon-specific terciles of observed target changes fit
only on combined train and validation. Manuscript evaluation uses forecast
origins 2015-07-01 through 2023-07-04. Neural RMSE is computed within each seed
and regime, then summarized as the mean and sample standard deviation across
ten seeds (`ddof=1`).

