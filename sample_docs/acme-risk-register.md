# Risk Register — Acme Digital Platform Phase 2

**Last reviewed:** 8 July 2026

## Open risks

### R-07 — UAT resourcing not confirmed (HIGH)
Acme has not yet confirmed the 10 business users needed for UAT starting 18 August. If confirmation slips past 18 July, UAT start moves at least one week and the 30 September go-live is directly threatened.
**Owner:** Daniel Weber (Acme) / Rajagopal H tracking. **Mitigation:** escalated to Priya Sharma at the 11 July steering committee; fallback plan drafted to run UAT in two staggered waves.

### R-09 — Search performance below SLA (HIGH)
Peak-hour search latency is 4.1s against a 2s contractual SLA. Root cause identified: vector index rebuild job competing for resources during business hours.
**Owner:** Tom Okafor. **Mitigation:** move index rebuilds to a nightly window and add read replicas; fix scheduled for sprint 14, verification in the pilot environment by 25 July.

### R-11 — InfoSec review may delay SSO build (MEDIUM)
Anita Krishnan's InfoSec reviews historically take 2–3 weeks. SSO build has a 4-week window in the plan; a late or negative review compresses build and test time.
**Owner:** Meera Iyer. **Mitigation:** design document submitted 7 July with a pre-review walkthrough offered; SSO build resources can be temporarily redirected to CR-118 if approval slips.

### R-12 — CR-118 procurement sign-off pending (MEDIUM)
Invoice OCR change request (340 hours) agreed by steering committee but purchase order not yet issued. Work cannot start without PO per Acme procurement rules; each week of delay pushes OCR delivery past go-live, meaning a post-go-live release.
**Owner:** Daniel Weber. **Mitigation:** Rajagopal to raise with Priya if PO not received by 15 July.

## Recently closed

### R-05 — Data residency compliance (CLOSED 26 June)
Analytics pipeline confirmed to keep all PII within ap-south-1 (Mumbai). Verified by Tom Okafor and accepted by Acme at the 26 June steering committee.
