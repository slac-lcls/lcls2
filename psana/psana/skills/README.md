# LCLS-II DAQ Diagnostic Skills

Opencode agent skills for diagnosing the LCLS-II DAQ. The primary consumer is
the AMI-spawned agent (the "Agent" button in the AMI flowchart GUI), running
alongside a live, fully-running DAQ.

This directory is scanned by AMI's `mcp_server.py` (`skills/*/SKILL.md`
glob) and copied into a spawned agent's `.opencode/skills/`, where each skill
becomes loadable on-demand via the `skill` tool. None of these skills are
always-loaded into every agent session today — a config change to make
`psana-daq` always-loaded is tracked separately and not yet applied.

## Skill map

| Skill | Role | Purpose |
|---|---|---|
| `psana-daq` | Router + composed sweep | Entry point for any DAQ issue. Dispatches to a single leaf skill when the report names a specific angle; runs an autonomous sweep across all leaf skills for a vague report and returns one ranked report. Establishes `hutch` and session scope (live/past) once, up front. |
| `psana-daq-control` | Leaf | Run-control state machine, failed transitions, rollcall, `activedet.json`. Needs no Grafana, MCP, or ConfigDB — works when everything else is down. |
| `psana-daq-monitor` | Leaf | Grafana/Prometheus DAQ metrics — event rate, deadtime, damage, buffers, event builder/MEB health. |
| `psana-daq-logs` | Leaf | Raw DAQ log files on disk, for the current session or a specific identifiable past session. |
| `psana-configdb` | Leaf | Read-only ConfigDB lookups — detector/system configuration, config history. |
| `psana-daq-history` | Stateful, composed | GitHub-issue-backed triage history — has `psana-daq`'s autonomous sweep search for prior occurrences of a failure mode before investigating, and record the outcome afterward, on human approval. The only stateful skill in this suite; every other skill here is stateless (reads the world, reports, remembers nothing). |

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

`psana-daq-history` retrofits `psana-daq`'s sweep with a search-first step
and a record-write step; it does not duplicate the sweep logic, and the
sweep carries no history/search/checkpointing logic of its own.

## Design principles

Every skill in this suite follows these. New skills should too:

- **Read-only with respect to the DAQ.** The agent gathers evidence and
  recommends; the human executes every remediation against the DAQ itself.
  This is non-negotiable — see `psana-daq-control/SKILL.md`'s "READ-ONLY
  POSTURE — NON-NEGOTIABLE" section for the canonical statement. This does
  not prohibit writes to systems other than the DAQ: `psana-daq-history`
  writes diagnostic records to GitHub, but only on explicit human approval,
  and never to any DAQ system.
- **Confidence-labelled claims.** Every claim is tagged with how it was
  established: `verified-live` (executed against the real service or
  filesystem), `verified-against-real-logs` (grepped from real production
  logs), or `inferred-from-code-only` (read from source, never operationally
  confirmed). `psana-daq-history` adds a fourth tier,
  `verified-in-production-incident`, for claims observed in a diagnosed live
  failure.
- **Cite, don't restate.** Sibling skills reference each other's sections by
  heading name rather than copying command tables, PromQL, or grep patterns.
  Duplicated logic has drifted out of sync more than once in this suite's
  history — citing the owning skill is how that's avoided going forward.

