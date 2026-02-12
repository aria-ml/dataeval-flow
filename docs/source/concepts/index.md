# Explanation

Understanding the concepts and design decisions behind DataEval App.

## Architecture

DataEval App wraps the [DataEval](https://github.com/aria-ml/dataeval/) library into
a configurable application with YAML-driven workflows and containerized deployment.
It bridges dataset loading, preprocessing, evaluation, and result reporting into a
single pipeline.

## Key Concepts

- **Data Cleaning** — Automated detection and flagging of dataset quality issues
  including outliers, duplicates, and label anomalies. See the
  [DataEval documentation](https://github.com/aria-ml/dataeval/) for algorithm details.

- **Outlier Detection** — Statistical methods (modified z-score, z-score, IQR) applied
  across image dimensions, pixel statistics, and visual features to identify anomalous
  samples.

- **Preprocessing** — Configurable image transform pipelines built on torchvision,
  specified as ordered steps in the workflow configuration.

- **Configuration** — YAML-based workflow configuration supporting data cleaning
  parameters, model specifications, and preprocessing steps.

- **MAITE Compatibility** — Datasets are converted to the MAITE protocol format,
  enabling interoperability with the broader JATIC ecosystem.
