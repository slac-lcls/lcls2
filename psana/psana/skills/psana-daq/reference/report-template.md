# Sweep Report Schema — Field-by-Field Rules

The full report template and every field-level rule for `psana-daq`'s
autonomous sweep. `SKILL.md`'s "Autonomous sweep" step 8 points here for
the complete schema; it keeps only the mandate that this report is the
sweep's one deliverable, not the schema detail itself.

---

## The template

Fill in the placeholders; do not change the headings or field names:

```
## New Crashes
<process deaths (Python exception or segfault — see psana-daq-logs/
 reference/crash-signatures.md) with NO existing Known Issues match found
 in step 5 — a filed GitHub issue or a staged issue draft both count as
 existing; write "none" if none found>
1. <component> — <one-line description of the crash>
   Confidence: <high | medium | low>
   Evidence:
     - <finding> — <tier>

## Most Likely Cause
<one-line statement — "Unknown" is a complete, valid answer when the
 evidence doesn't isolate a cause; do not force a guess to fill this field>
Confidence: <high | medium | low>
Evidence:
  - <finding> — <verified-live | verified-against-real-logs | inferred-from-code-only | documented-in-issue>
  - <finding> — <tier>
Remedies: <proposed | tried | verified — see "Remedies" below; omit this
 line entirely if no remedy has been proposed>

## Also Possible
1. <alternative explanation for the SAME symptom> — Confidence: <high | medium | low>
   Evidence:
     - <finding> — <tier>
2. ...

## Independent Issues
<separate faults found during the sweep that do NOT explain the primary
 symptom — write "none" if every finding relates to the one fault above.
 Rank: crashes first, then data loss/damage, then transition failures,
 then everything else>
1. <issue> — Confidence: <high | medium | low>
   Evidence:
     - <finding> — <tier>

## Known Issues
<matched GitHub issues or issue drafts for findings above — write "none"
 if step 5's search found no matches. Condensed: one line + an optional
 one-line delta, not a re-derivation of the finding's evidence. Real
 Markdown links for GitHub issues, not bare #N.>
- [#<N>](https://github.com/slac-lcls/lcls2/issues/<N>) (open) — <title> — matches: Independent Issue 1
- [#<N>](https://github.com/slac-lcls/lcls2/issues/<N>) (CLOSED <date>) — <title> — ⚠ RECURRENCE after believed fix
- issue-drafts/<name>.md (staged, unfiled) — matches: New Crashes 1

## Raw Findings
- Session: <prefix(es) examined — comma-separated if more than one, or
  "none — no sessions in window">
- Control: <finding, or "skipped — <reason>">
- Logs: <finding, or "skipped — <reason>">
- Metrics: <finding, or "skipped — <reason>", or "no samples returned for the requested window">
- ConfigDB: <finding, or "no detector implicated — skipped">
```

---

## Two independent label systems — do not conflate them

- **`Confidence:`** (verdict-level, one per cause/alternative/issue) — how
  confident the sweep is that this *conclusion* is right, not how the
  supporting evidence was obtained:
  - **high** — direct evidence names the failing component and the
    mechanism is established
  - **medium** — evidence is consistent with this cause but doesn't
    isolate it from other explanations
  - **low** — a plausible inference; competing explanations have not been
    ruled out
  - **Unknown is a valid, complete `Most Likely Cause`** when nothing
    isolates one — do not force a low-confidence guess into this field
    just to fill it.

  A match in `## Known Issues` may justify raising this rating, but the
  finding must name `## Known Issues` as the reason — no silent upgrades.
  A single state-machine or protocol-phase deduction (e.g. "stuck in
  connected targeting running ⇒ died on configure") is a candidate
  prediction, not proof — require the actual corroborating log line before
  stating a specific failed transition as fact.

- **Evidence tiers** (per evidence item, one per line under `Evidence:`) —
  how that specific piece of evidence was obtained. Exactly these four —
  don't invent a fifth:
  - `verified-live`
  - `verified-against-real-logs`
  - `inferred-from-code-only`
  - `documented-in-issue` — a filed GitHub issue documents this as a known
    cause, rather than this sweep having observed it directly.

  **An unfiled issue draft (`github_issue: null`) is not
  `documented-in-issue`** — that tier means a real GitHub issue. Tag such a
  finding with whatever tier its underlying evidence actually is — almost
  always `verified-against-real-logs`, since that's what the draft's own
  content was grepped from at staging time. The "staged, unfiled" fact is
  already signaled by `## Known Issues`' own line — tracking-status and
  evidence-provenance are separate axes.

## Remedies

A remedy is **verified** only with outcome evidence tied to that specific
attempt and scope — an improvement observed after an intervention, on its
own, still leaves causality uncertain (a concurrent change, a naturally
recovering condition, or a different fault clearing at the same time are
all alternative explanations that outcome evidence alone doesn't rule out).
Use `proposed` for an untried suggestion, `tried` for an attempted-but-not-
yet-confirmed fix, `verified` only when the outcome was actually checked
against the specific thing that was changed. Omit the `Remedies:` line
entirely rather than writing `Remedies: none` when nothing has been
proposed.

## Also Possible vs. Independent Issues — the discrimination rule

Does this finding offer an *alternative explanation for the same symptom*
the user reported (→ **Also Possible**), or is it a *separate fault* that
would still exist even if the primary cause were fixed (→ **Independent
Issues**)? A dying HSD card that presents as configure-timeout and,
separately, as elevated deadtime is one fault surfacing twice — report it
once, not as two Independent Issues.

## Ranking within `## Independent Issues`

Crashes first, then data loss/damage, then transition failures, then
everything else. A known recurrence (already in `## Known Issues`) ranks
below an otherwise-equal new finding — it's already tracked.

## `## New Crashes` vs. `## Known Issues` vs. `## Independent Issues` — gating and cross-reference rules

- A crash **with no existing Known Issues match** goes in `## New Crashes`.
- A crash **with an existing match — a filed GitHub issue or an issue
  draft alike, "existing" is not GitHub-only** — goes in `## Known Issues`
  instead, as a pointer. Do not also list it in `## New Crashes`.
- **New evidence about an already-tracked crash** (a new session it
  recurs in, a second occurrence, a different diagnostic string) belongs
  in `## Known Issues`' own one-line delta (e.g. "⚠ recurrence extended to
  `<session>`") — not a new `## New Crashes` entry.
- A crash that is **also the primary cause** appears in `## New Crashes`;
  `## Most Likely Cause` cross-references it ("see New Crashes #1") rather
  than duplicating the evidence block.

## Report-level anti-re-derivation principle

If a finding matches a `## Known Issues` entry, **every** section
referencing it — `## Also Possible`, `## Independent Issues`, `##
Most Likely Cause` — points at it rather than restating its content.

## `## Known Issues` link form

Real Markdown links for GitHub issues (`[#134](https://github.com/slac-lcls/lcls2/issues/134)`),
not a bare `#134`. Keep the closed-issue recurrence marker (`⚠ RECURRENCE
after believed fix`) and the staged-draft line form (`issue-drafts/
<name>.md (staged, unfiled)`) as shown in the template above.
