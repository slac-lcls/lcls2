# LCLS-II DAQ Diagnostic Skills

Opencode agent skills for diagnosing the LCLS-II DAQ. These are used
primarily **offline, outside AMI** — via the CLI or any opencode session —
which is the DAQ team's stated preference. They also work from inside an
AMI-spawned agent (the "Agent" button in the AMI flowchart GUI) for live
debugging, which is a real but secondary capability.

This directory is scanned for `skills/*/SKILL.md` files by any consumer
that wants to discover and load them on demand via the `skill` tool
(AMI's `mcp_server.py` is one such consumer, copying discovered skills
into a spawned agent's `.opencode/skills/`, but it is not the only way
these skills are used). None of these skills are always-loaded into every
session — each is discoverable by its `description:` frontmatter alone,
and loads only when a session's question actually matches it.

## Skill map

| Skill | Role | Purpose |
|---|---|---|
| `psana-daq` | Router + composed sweep | Entry point for any DAQ issue. Dispatches to a single leaf skill when the report names a specific angle; runs an autonomous sweep across all leaf skills for a vague report and returns one ranked report. Establishes `hutch` and session scope (live/past) once, up front. **Read-only, everywhere** — searches GitHub + the local diagnostic store before sweeping, never writes anything itself. |
| `psana-daq-control` | Leaf | Run-control state machine, failed transitions, rollcall, `activedet.json`. Needs no Grafana, MCP, or ConfigDB — works when everything else is down. |
| `psana-daq-monitor` | Leaf | Grafana/Prometheus DAQ metrics — event rate, deadtime, damage, buffers, event builder/MEB health. |
| `psana-daq-logs` | Leaf | Raw DAQ log files on disk, for the current session or a specific identifiable past session. |
| `psana-configdb` | Leaf | Read-only ConfigDB lookups — detector/system configuration, config history. |
| `psana-daq-history` | Composed, opt-in | **The only skill in this suite that writes anything.** Persists a `psana-daq` sweep report, triages its findings into local case files, and files/comments on GitHub issues on `slac-lcls/lcls2`. Persisting and triage need no setup; only GitHub filing needs a token, with a no-MCP hand-over-command fallback. Loaded only after a sweep's report is shown, on explicit hand-off — never mid-sweep, never after targeted dispatch. |

Each skill's `description:` frontmatter is the authoritative statement of when
to load it — the purposes above are a short paraphrase, not a substitute for
reading the frontmatter or the skill itself.

## How they compose

`psana-daq` establishes `hutch` (the target instrument/hutch) and session
scope (live, or a past date/time range) once, and hands both off explicitly
to whichever leaf skill it routes to, rather than having each leaf skill
re-derive or re-ask for either.

`psana-daq`'s autonomous sweep (for vague reports) is not a fifth diagnostic
technique — it is entirely composed of the four leaf skills' existing
methods, called in sequence and cited by section rather than restated. It
was originally a separate skill (`psana-daq-snapshot`) and was merged into
the router because it had no inbound citations of its own, needed the
router's identity/scope/preflight machinery to run, and its "vague report"
trigger condition duplicated the router's own dispatch table.

**All writes in this suite live in one skill.** `psana-daq` searches a
small diagnostic store — local `reports/` (immutable per-sweep snapshots)
and local `cases/` (mutable, **staging only** — draft GitHub issue
bodies) — against the unauthenticated GitHub REST API, no setup required,
and renders findings condensed when a match is found. It never writes to
either tree and never calls a GitHub write endpoint. After a sweep's
report is shown (never after targeted dispatch — a live "what's the
deadtime right now?" query has nothing worth persisting), it offers a
single hand-off to `psana-daq-history`, which owns everything downstream:
persisting the report, the mechanical + one-prompt triage that decides
which findings become case files, staging those files, and — on
request — filing/commenting on GitHub issues (MCP setup, label
suggestion by querying the repo's real labels rather than a maintained
enumerated vocabulary, and the no-MCP hand-over-command fallback). See
`psana-daq-history/SKILL.md` for the full model.

**Condensing survives even if `psana-daq-history` is never loaded.** The
router's own search finds prior issues and renders them as one-line
pointers in the report (`## Known Issues`) rather than full findings —
this is what makes filing an issue pay off for *every future sweep*, not
just ones that also load the history skill.

Discussion on a filed issue may continue in a Slack thread — GitHub's
native Slack app creates one automatically per issue, with no bot token or
agent-side Slack access involved. `psana-daq-history` documents the
convention (post conclusions back to the issue) at the point filing
happens; the issue, not the thread, is what a later sweep's search
actually finds.

## Design principles

Every skill in this suite follows these. New skills should too:

- **Read-only with respect to the DAQ.** The agent gathers evidence and
  recommends; the human executes every remediation against the DAQ itself.
  This is non-negotiable — see `psana-daq-control/SKILL.md`'s "READ-ONLY
  POSTURE — NON-NEGOTIABLE" section for the canonical statement.
  `psana-daq` takes this further and is read-only everywhere, not just
  the DAQ — it never writes a local file or calls a GitHub write endpoint.
  **Every write in the suite is `psana-daq-history`'s job**: local files
  on confirmation at hand-off; GitHub issues/comments on per-action human
  confirmation (via MCP) or human-executed (via a hand-over command when
  no MCP/token is configured) — never automatically, and never to any DAQ
  system.
- **Confidence and provenance are two separate axes — do not conflate
  them.** `Confidence: high|medium|low` describes how sure a diagnostic
  *conclusion* is. A separate, four-value provenance tag describes how a
  piece of *evidence* was obtained: `verified-live` (executed against the
  real service or filesystem), `verified-against-real-logs` (grepped from
  real production logs), `inferred-from-code-only` (read from source,
  never operationally confirmed), or `documented-in-issue` (a GitHub issue
  documents this as a known cause, rather than this sweep having observed
  it directly). A finding's confidence may be raised by a matched issue,
  but the report must name that as the reason — never a silent upgrade.
- **Cite, don't restate.** Sibling skills reference each other's sections by
  heading name rather than copying command tables, PromQL, or grep patterns.
  Duplicated logic has drifted out of sync more than once in this suite's
  history — citing the owning skill is how that's avoided going forward.

