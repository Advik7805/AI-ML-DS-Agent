# Data Privacy — Northwind Global group policy

Owner: Group Privacy Office · Applies to all business units

## Personal data
Customer usage records are personal data. They may be used for churn scoring and service
assurance, and for nothing else without a fresh lawful basis.

## Retention
Usage telemetry is retained for **24 months**, then aggregated to monthly summaries. Raw
per-call detail is deleted.

## Model artifacts
Trained models must not embed identifiable records. SHAP explanations are computed at request
time and are not persisted alongside customer identifiers.

## Access
Analyst access to raw usage data requires a time-boxed grant, reviewed quarterly.
