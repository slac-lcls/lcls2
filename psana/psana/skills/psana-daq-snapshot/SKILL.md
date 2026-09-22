---
name: psana-daq-snapshot
description: Compose a read-only DAQ health report for a live session, an identified past session, or a hutch-wide historical window spanning multiple launches. Reuse supplied scope and evidence; investigate available state, logs, metrics and relevant configuration, then rank supported hypotheses with coverage limits.
---

# LCLS-II DAQ Snapshot

Produce one readable report for the requested hutch and session/window. Use
[psana-daq-control](../psana-daq-control/SKILL.md)'s **READ-ONLY POSTURE —
NON-NEGOTIABLE**: gather evidence and recommend; humans execute remediation.

## Establish scope once

Reuse supplied hutch, live/historical mode, window (including time zone),
launch/run identities, configuration and evidence. Ask only for missing details
that affect the result; continue independent investigation when possible. A
hutch-wide report includes all relevant launches and is organized by findings
and chronology, not platform/partition. Preserve platform as correlation
metadata. If launch selection is needed, use
[psana-daq-logs](../psana-daq-logs/SKILL.md)'s **Session selection and historical
bounds**. A launch/session is not synonymous with a DAQ run.

Pass the same scope, known release identities and evidence references to each
loaded skill. Check availability before using an angle. Explicitly mark each
leg as checked, partially checked, unavailable (with reason), or not applicable.
A service failure must not prevent useful work from retained evidence. Do not
query today's state or configuration as a substitute for historical evidence.

## Sweep available evidence

Load each relevant leaf skill once, when needed. Reuse collected evidence and
cached query results for follow-up questions; repeat only checks affected by
new evidence or changed scope. Give short progress updates for substantive legs.

- **State:** for live investigations follow the router's **Branch-on-state
  workflow for vague reports**, using the control skill's bare status call and
  `showPlatform`. For historical work reconstruct transitions from retained
  control logs/run metadata; label them reconstructed. Live `showPlatform` and
  today's `activedet.json` cannot establish past component membership.
- **Logs:** use the logs skill's **Session selection and historical bounds**,
  **Compressed/rotated logs and error counts**, and **Interpreting `<C>`/`<E>`
  messages**. Include all relevant launches and retain uncertain overlaps.
- **Metrics:** if available, load
  [psana-daq-monitor](../psana-daq-monitor/SKILL.md). Start with applicable
  groups A (event rate), B (deadtime), C (damage), D (DRP errors), I (MEB buffers)
  and G (event-builder fixups/timeouts); narrow follow-ups to implicated signals.
  Follow **Historical windows and coverage**. Use the supplied historical
  window, or corroborated run bounds; if only file activity can be estimated,
  label the query interval approximate and its basis. Prometheus queries use
  `startTime`/`endTime`; panel rendering uses `timeRange`. Apply the chosen
  interval to discovery, queries, panels and links. Preserve per-series identity
  to distinguish overlapping launches and endpoint reuse after restarts.
- **Configuration:** load [psana-configdb](../psana-configdb/SKILL.md) when
  evidence or the user's question implicates a detector, timing, trigger or
  system configuration. Otherwise mark this leg not applicable. Follow
  **Correlating history with a specific run**; distinguish history candidates,
  recorded Configure content and a corroborated applied key. If evidence is
  unavailable, leave the historical identity unknown.

## Readable report

Retain the existing headings and `Evidence:` field for readers of earlier
reports. This is a readable template, not a new machine-readable schema;
application-specific storage/chat/note contracts belong to their consumer.
No parser or history integration is supplied by this tree.

```text
## Most Likely Cause
<supported hypothesis, confirmed cause with corroboration, or "Unknown">
Confidence: <reasoned assessment of the conclusion, including missing evidence>
Evidence: <source references, timestamps/line numbers, and evidence origin>

## Also Possible
<ranked alternatives and discriminating checks, or none supported>

## Raw Findings
- Scope: <hutch, window/time zone, launches/runs, release identities>
- State: <status/observation; historical reconstruction labelled>
- showPlatform: <live result, unavailable, or not applicable to historical work>
- Logs: <status, observations and citations; error lines vs occurrences>
- Metrics: <status, query window/selectors, observations and retention/gaps>
- ConfigDB: <status, candidate/applied identity and corroboration, or unknown>
- Remedies: <proposed, tried, or verified; supporting before/after evidence>
- Deferred issues: <unresolved items and next evidence needed>
```

Evidence-origin labels (`verified-live`, `verified-against-real-logs`,
`inferred-from-code-only`) describe how an observation was obtained; they do
not measure certainty that it caused the failure. Label supplied/synthetic
fixtures as such, never as production verification. Keep observations,
hypotheses, confirmed causes, and proposed/tried/verified remedies distinct.
A remedy is verified only with outcome evidence tied to that attempt and
scope; improvement after an intervention alone may leave causality uncertain.
Do not force a cause ranking when the cause is unknown.

## History boundary

This skill neither searches nor writes GitHub issues. `psana-daq-history` is
not included in this tree. See the suite [README](../README.md) for the missing
integration and requirements for evaluating any later history package.
