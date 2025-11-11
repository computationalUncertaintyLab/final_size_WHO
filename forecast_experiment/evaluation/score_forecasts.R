# Forecast Evaluation using scoringutils
# Author: mcandrew
# Description: Score quantile forecasts using the scoringutils R package

# ===== SETUP =====

# Install scoringutils if not already installed
if (!require("scoringutils", quietly = TRUE)) {
 remotes::install_github("epiforecasts/scoringutils", dependencies = TRUE)# install.packages("scoringutils")
}

# Load required libraries
library(scoringutils)
library(data.table)

cat("Loading forecast data...\n")

# ===== LOAD DATA =====

# Read the prepared forecast data
forecast_data <- fread("./forecast_experiment/evaluation/forecasts_scoringutils_format.csv")

cat(sprintf("Loaded %d rows of forecast data\n", nrow(forecast_data)))
cat(sprintf("Columns: %s\n", paste(names(forecast_data), collapse=", ")))

# ===== DATA PREPROCESSING =====

# Convert date columns to Date type
forecast_data[, forecast_date := as.Date(forecast_date)]
forecast_data[, target_end_date := as.Date(target_end_date)]

# Ensure quantile_level is numeric
forecast_data[, quantile_level := as.numeric(quantile_level)]

# Ensure other numeric columns are properly typed
forecast_data[, horizon := as.numeric(horizon)]
forecast_data[, predicted := as.numeric(predicted)]
forecast_data[, observed := as.numeric(observed)]

# Display summary of data
cat("\n===== DATA SUMMARY =====\n")
cat(sprintf("Unique locations: %d\n", length(unique(forecast_data$location))))
cat(sprintf("Unique seasons: %d\n", length(unique(forecast_data$season))))
cat(sprintf("Unique forecast dates: %d\n", length(unique(forecast_data$forecast_date))))
cat(sprintf("Unique target end dates: %d\n", length(unique(forecast_data$target_end_date))))
cat(sprintf("Horizons: %s\n", paste(sort(unique(forecast_data$horizon)), collapse=", ")))
cat(sprintf("Quantile levels: %s\n", paste(sort(unique(forecast_data$quantile_level)), collapse=", ")))
cat(sprintf("Date range: %s to %s\n", 
            min(forecast_data$forecast_date), 
            max(forecast_data$forecast_date)))

# ===== CONVERT TO FORECAST OBJECT =====

cat("\n===== CREATING FORECAST OBJECT =====\n")

# Define the forecast unit - all columns that uniquely identify a single forecast
# Each unique combination of these should have multiple quantile_level rows
forecast_unit <- c("location", "forecast_date", "target_end_date", "horizon", "model", "season")

cat(sprintf("Forecast unit: %s\n", paste(forecast_unit, collapse=", ")))

# Convert to scoringutils forecast object
forecast_obj <- as_forecast_quantile(
  data = forecast_data,
  forecast_unit = forecast_unit,
  observed = "observed",
  predicted = "predicted",
  quantile_level = "quantile_level"
)

cat("Forecast object created successfully\n")
cat(sprintf("Forecast type: %s\n", class(forecast_obj)[1]))

# ===== SCORE FORECASTS =====

cat("\n===== SCORING FORECASTS =====\n")
cat("This may take a while for large datasets...\n")

# Score all forecasts
# This computes WIS, coverage, bias, and other metrics for each unique forecast
scores <- score(forecast_obj)

cat(sprintf("Scoring complete! Generated %d scored forecasts\n", nrow(scores)))

# Display available metrics
metric_cols <- names(scores)[!names(scores) %in% c(forecast_unit, "target", "latest_MMWR")]
cat(sprintf("\nAvailable metrics: %s\n", paste(metric_cols, collapse=", ")))

# Display summary statistics for key metrics
# cat("\n===== METRIC SUMMARIES =====\n")

# if ("wis" %in% names(scores)) {
#   cat(sprintf("WIS - Mean: %.2f, Median: %.2f, SD: %.2f\n", 
#               mean(scores$wis, na.rm=TRUE),
#               median(scores$wis, na.rm=TRUE),
#               sd(scores$wis, na.rm=TRUE)))
# }

# if ("interval_coverage_90" %in% names(scores)) {
#   cat(sprintf("90%% Interval Coverage - Mean: %.3f\n", 
#               mean(scores$interval_coverage_90, na.rm=TRUE)))
# }

# if ("interval_coverage_50" %in% names(scores)) {
#   cat(sprintf("50%% Interval Coverage - Mean: %.3f\n", 
#               mean(scores$interval_coverage_50, na.rm=TRUE)))
# }

# if ("bias" %in% names(scores)) {
#   cat(sprintf("Bias - Mean: %.3f, Median: %.3f\n", 
#               mean(scores$bias, na.rm=TRUE),
#               median(scores$bias, na.rm=TRUE)))
# }

# ===== EXPORT RESULTS =====

cat("\n===== EXPORTING RESULTS =====\n")

# Convert to data.table if not already
scores_dt <- as.data.table(scores)

# Export single comprehensive CSV with all scores and temporal information
output_file <- "./forecast_experiment/evaluation/forecast_scores.csv"
fwrite(scores_dt, output_file)

cat(sprintf("Scores saved to: %s\n", output_file))
cat(sprintf("Output contains %d rows and %d columns\n", nrow(scores_dt), ncol(scores_dt)))

# Display sample of results
# cat("\n===== SAMPLE OF SCORED FORECASTS =====\n")
# print(head(scores_dt, 10))

# # Summary by season
# cat("\n===== SUMMARY BY SEASON =====\n")
# if ("season" %in% names(scores_dt) && "wis" %in% names(scores_dt)) {
#   season_summary <- scores_dt[, .(
#     n_forecasts = .N,
#     mean_wis = mean(wis, na.rm=TRUE),
#     median_wis = median(wis, na.rm=TRUE),
#     mean_coverage_50 = mean(interval_coverage_50, na.rm=TRUE),
#     mean_coverage_90 = mean(interval_coverage_90, na.rm=TRUE)
#   ), by = season]
#   print(season_summary)
# }

# # Summary by location
# cat("\n===== SUMMARY BY LOCATION (Top 10) =====\n")
# if ("location" %in% names(scores_dt) && "wis" %in% names(scores_dt)) {
#   location_summary <- scores_dt[, .(
#     n_forecasts = .N,
#     mean_wis = mean(wis, na.rm=TRUE),
#     median_wis = median(wis, na.rm=TRUE)
#   ), by = location][order(mean_wis)][1:10]
#   print(location_summary)
# }

# # Summary by horizon
# cat("\n===== SUMMARY BY HORIZON =====\n")
# if ("horizon" %in% names(scores_dt) && "wis" %in% names(scores_dt)) {
#   horizon_summary <- scores_dt[, .(
#     n_forecasts = .N,
#     mean_wis = mean(wis, na.rm=TRUE),
#     median_wis = median(wis, na.rm=TRUE),
#     mean_coverage_50 = mean(interval_coverage_50, na.rm=TRUE),
#     mean_coverage_90 = mean(interval_coverage_90, na.rm=TRUE)
#   ), by = horizon][order(horizon)]
#   print(horizon_summary)
# }

# cat("\n===== EVALUATION COMPLETE =====\n")
# cat("All scores have been saved with complete temporal information.\n")
# cat("You can now analyze the results by grouping on any combination of:\n")
# cat("  - season\n")
# cat("  - location\n")
# cat("  - forecast_date\n")
# cat("  - target_end_date\n")
# cat("  - horizon\n")
# cat("  - latest_MMWR\n")

