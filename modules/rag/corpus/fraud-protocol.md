# Fraud Protocol — Northwind Pay

Owner: Payment Risk, Northwind Pay · Applies to every card authorisation

## Scoring
Every transaction is scored before settlement. Above the operating threshold the transaction
is held, not declined — a false positive that declines a legitimate purchase costs more in
churn than the fraud it prevents.

## Review queue
Held transactions enter a manual review queue with a **15-minute** SLA during business hours.
Outside those hours, holds over $1,000 page the on-call analyst.

## Velocity signals
The strongest signals are velocity-based: transaction count in the trailing hour, distinct
merchants in the trailing day, and geographic distance from the customer's usual radius.

## Feedback
Analyst dispositions are written back to the training set nightly. A model retrained without
that feedback loop degrades within weeks.
