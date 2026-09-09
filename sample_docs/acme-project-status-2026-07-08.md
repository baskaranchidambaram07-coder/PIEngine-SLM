# Acme Corp Digital Platform — Weekly Status Report

**Week ending:** 8 July 2026
**Programme:** Acme Digital Platform Phase 2
**Delivery Manager:** Rajagopal H
**Overall status:** AMBER

## Executive summary

Phase 2 build is 68% complete. Development velocity remains on plan, but the programme is Amber due to two factors: Acme UAT resourcing for August is still unconfirmed (action A-45 overdue), and search performance in the pilot environment does not yet meet the 2-second SLA.

## Milestone status

| Milestone | Baseline | Forecast | Status |
|-----------|----------|----------|--------|
| M5 — Feature complete | 15 Aug 2026 | 15 Aug 2026 | GREEN |
| M6 — UAT start | 18 Aug 2026 | 25 Aug 2026 (at risk) | AMBER |
| M7 — Go-live | 30 Sep 2026 | 30 Sep 2026 | AMBER |

## Progress this week

- Completed invoice ingestion API and reconciliation dashboard widgets.
- SSO design document submitted to Acme InfoSec on 7 July; review meeting expected week of 14 July.
- Search profiling identified the vector index rebuild job as the main contributor to peak-hour latency; fix in design.

## Budget

Consumed 61% of Phase 2 budget against 65% planned effort — tracking slightly under. CR-118 (340 hours) still awaiting Acme procurement sign-off; work not started.

## Key asks of the customer

1. Confirm August UAT resource plan (10 business users, 50% allocation) — blocking M6.
2. Expedite procurement sign-off on CR-118 to protect the September go-live.
3. Nominate InfoSec reviewer availability for the SSO review meeting.
