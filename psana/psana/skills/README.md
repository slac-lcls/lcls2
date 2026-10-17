# LCLS-II DAQ Diagnostic Skills

Agent skills, in the Agent Skills format (a `SKILL.md` with YAML
frontmatter, plus optional bundled `reference/`/`scripts/`), for
diagnosing the LCLS-II DAQ. Written to be harness-agnostic: known working
harnesses today are opencode, Claude Code, and Codex, used primarily
**offline** via the CLI — the DAQ team's stated preference — but nothing
here assumes a specific one. Any harness that can discover and load
`SKILL.md` files, and run shell commands plus (for `psana-daq-monitor`/
`psana-daq-history`) the relevant MCP tools, should work.

This directory is scanned for `skills/*/SKILL.md` files by any consumer
that wants to discover and load them on demand. None of these skills are
always-loaded — each is discoverable by its `description:` frontmatter
alone, and loads only when a session's question matches it.

## Skill map

| Skill | Role | Purpose |
|---|---|---|
| `psana-daq` | Router + composed sweep | Entry point for any DAQ issue. Dispatches to a single leaf when the report names a specific angle (handing off to a second leaf only when the first's result requires it); runs an autonomous sweep across all leaves for a vague report and returns one ranked report. Establishes `hutch`, `platform`, and a `[start, end]` UTC window once, up front — there is no separate live/past mode, just a window that may or may not end at "now." **Read-only, everywhere** — searches local reports before the sweep's legs run (to decide coverage), then GitHub + local issue drafts after the legs (once there's something concrete to search for) — never writes anything itself. |
| `psana-daq-control` | Leaf | Run-control state machine and transitions, reconstructed from `control.log` — needs no Grafana, MCP, ConfigDB, or reachable DAQ. |
| `psana-daq-monitor` | Leaf | Grafana/Prometheus DAQ metrics — event rate, deadtime, damage, buffers, event builder/MEB. Always a range query over the handed-off window. |
| `psana-daq-logs` | Leaf | Raw DAQ log files on disk for a session in a given window — live or a specific past date/range, same mechanism either way. |
| `psana-configdb` | Leaf | Read-only ConfigDB lookups — detector/system configuration, config history (read as candidate database edits, not current state). |
| `psana-daq-history` | Composed, opt-in | **The only skill in this suite that writes anything.** Persists a `psana-daq` sweep report, triages its findings into local issue drafts, and files/comments on GitHub issues on `slac-lcls/lcls2`. Persisting and triage need no setup; GitHub filing uses a GitHub MCP server when one is configured, or falls back to a ready-to-paste draft. Loaded only after a sweep's report, on explicit hand-off — never mid-sweep, never after targeted dispatch. |

Each skill's `description:` frontmatter is the authoritative statement of
when to load it — the purposes above are a paraphrase, not a substitute.

## How they compose

`psana-daq` establishes `hutch`, `platform`, and a `[start, end]` UTC
window once, and hands them off explicitly to whichever leaf it routes
to. It also establishes the active ConfigDB configuration alias when a
ConfigDB handoff is possible — a separate fact from `platform`, not
implied by it. `platform` lets `psana-daq-monitor` filter DAQ-process
metrics by the Prometheus `partition` label, but that label means
something different on XPM metrics than on DAQ-process metrics — see
`psana-daq-monitor/SKILL.md` for the detail. **"Live" is simply a window
ending at "now," scoped to the newest launch — not a separate code path
any leaf needs to branch on.** This collapsed what used to be duplicated
live-vs-past logic across every leaf into one model, stated once.

`psana-daq`'s autonomous sweep (for vague reports) is not a fifth
diagnostic technique — it's entirely composed of the four leaves' existing
methods, called in sequence and cited by section, not restated.

**All writes in this suite live in one skill.** `psana-daq` searches local
`reports/` before any leg runs (to decide coverage — a session is covered
only if its last log file's mtime predates the report that covers it),
then, once the legs have actually run and named something concrete,
GitHub plus local `issue-drafts/`. A match condenses to a one-line pointer
under `## Known Issues` — presentation only; a newly-observed occurrence
or session delta is still stated in `## Raw Findings`. `psana-daq` never
writes to either tree and never calls a GitHub write endpoint. After a
sweep's report (never after targeted dispatch), it offers a single
hand-off to `psana-daq-history`, which owns everything downstream:
persisting the report, triaging findings into issue drafts, staging them,
and — on request — filing/commenting on GitHub (MCP setup, label
suggestion against the repo's real labels, manual-paste fallback).

**Condensing survives even if `psana-daq-history` is never loaded** — the
router's own search finds prior issues/drafts and renders them condensed
regardless, which is what makes filing an issue pay off for every future
sweep, not just ones that also load the history skill.

Discussion on a filed issue may continue in a Slack thread —
`#lcls-daq-diagnostics` is subscribed to the repo's issue activity via
GitHub's native Slack app, no bot token or agent-side Slack access
involved. `psana-daq-history` documents the convention (post conclusions
back to the issue) at filing time; the issue, not the thread, is what a
later sweep's search finds.

## Design principles

Every skill follows these; new skills should too:

- **One window, not two modes.** Every question is about a hutch and a
  time window; "live" is a window ending now. No skill should need a
  separate live/past branch — if one seems necessary, the window model is
  being violated somewhere, not a legitimate exception.
- **Read-only with respect to the DAQ.** The agent gathers evidence and
  recommends; the human executes every remediation. `psana-daq` goes
  further and is read-only *everywhere* — no local file, no GitHub write.
  **Every write in the suite is `psana-daq-history`'s job.**
- **Confidence and provenance are two separate axes.** `Confidence:
  high|medium|low` describes how sure a diagnostic *conclusion* is. A
  separate four-value tag describes how a piece of *evidence* was
  obtained: `verified-live`, `verified-against-real-logs`,
  `inferred-from-code-only`, `documented-in-issue`. A finding's confidence
  may be raised by a matched issue, but the report must name that as the
  reason — never a silent upgrade.
- **A candidate deduction is not proof.** A state-machine inference, a
  phase-1/phase-2 split, a config-history nearest-match — all are
  candidates until the actual corroborating log line, recorded content,
  or outcome evidence confirms them. `Unknown` is a complete, valid
  `Most Likely Cause` when nothing isolates one.
- **Absence of a match is "no evidence found," not a negative result.** A
  zero rate, an empty metrics series, a missing rollcall reply — each
  needs its own corroboration before being reported as "not running,"
  "healthy," or "deselected."
- **mtime is approximate file-activity evidence** — never a run boundary,
  never proof of which config/session is live.
- **No corpus-derived counts or frequency tables in shipped text.** They
  decay the moment the corpus they were derived from keeps growing. State
  the interpretation; cite a re-derivation recipe in `reference/` if one
  is useful; never a snapshot number in `SKILL.md` itself.
- **Cite, don't restate.** Sibling skills reference each other's sections
  by heading name rather than copying command tables, PromQL, or grep
  patterns.
