# TODOS

## Critical — Failure Mode Gaps

### TODO-001: TimescaleDB disk full handling
**Phase:** 2 (Cloud Core)
**Priority:** Critical
**What:** MQTT consumer should monitor available disk space and alert before TimescaleDB fills up. Without this, batch inserts silently fail and metrics are lost.
**Why:** Raw ingestion is 3.4 GB/day (compressed). A 500 GB disk fills in ~150 days. If the retention policy fails or compression lags, it fills sooner. Silent data loss is the worst kind of failure — operators think the system is working when it's not recording.
**How:** Add a periodic disk check in the MQTT consumer (every 60s). If available space < 10%, log a warning and publish an alert via Alertmanager. If < 5%, stop batch inserts and alert critical.
**Depends on:** Phase 2 cloud implementation.
**Added:** 2026-03-30 (eng review)

### TODO-002: ByteTrack track ID overflow
**Phase:** 1 (Edge Agent)
**Priority:** Critical
**What:** ByteTrack uses incrementing integer IDs for person tracks. At ~3000 new tracks/second across 30 cameras, INT32 max (2.1B) is reached in ~8 days. QueueAnalytics uses track_id as dict keys — overflow causes collisions and corrupted analytics.
**Why:** This is a time bomb. The system works perfectly for a week, then analytics silently degrade. Extremely hard to debug in production.
**How:** Verify ByteTrack's `_next_id` implementation. Either (a) reset to 0 when no active tracks exist, or (b) use modular arithmetic with a large prime, or (c) switch to INT64. QueueAnalytics must also handle track_id reuse gracefully (clear stale entries for reused IDs).
**Depends on:** Phase 1 ByteTrack integration.
**Added:** 2026-03-30 (eng review)
