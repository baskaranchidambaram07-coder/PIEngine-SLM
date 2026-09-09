# Architecture Decision Log — Acme Digital Platform

## ADR-014 — Azure AD SSO via OIDC (Approved 26 June 2026, pending InfoSec review)

**Context:** Acme requires single sign-on for the customer portal before Phase 2 go-live. Acme's identity provider is Azure AD (Entra ID).
**Decision:** Integrate using OpenID Connect authorization-code flow with PKCE. Portal sessions limited to 8 hours with silent token refresh. Group-to-role mapping maintained in the platform, synced from three Azure AD security groups (portal-users, portal-approvers, portal-admins).
**Consequences:** No passwords stored in the platform; account lifecycle owned by Acme IT. InfoSec review required before build (see risk R-11).

## ADR-015 — All production data in ap-south-1 (Approved 26 June 2026)

**Context:** Acme CFO mandate that production data, including analytics and backups, must remain in the Mumbai region.
**Decision:** Primary and DR both within ap-south-1 using separate availability zones. Cross-region replication disabled; DR strategy is zonal, accepting a documented residual risk for full-region outage, which Acme signed off.
**Consequences:** Region-wide outage RTO is 24h via backup restore. Any future use of managed AI services must verify Mumbai-region availability first.

## ADR-016 — Search on OpenSearch with nightly index rebuilds (Amended 8 July 2026)

**Context:** Peak-hour search latency of 4.1s breaches the 2s SLA; profiling showed index rebuild jobs competing with query traffic.
**Decision:** Move full index rebuilds to a 01:00–04:00 IST window, switch intraday updates to incremental delta indexing, and add two read replicas for query traffic.
**Consequences:** Search results may lag content changes by up to 15 minutes intraday, which the business accepted. Infra cost increases ~USD 410/month.
