# Crash Definition — Corpus Derivation

`SKILL.md`'s "Crash signatures" section states the operational definition
briefly; `psana-daq/SKILL.md`'s "Crash definition" cites it in turn rather
than restating it. This file is the full derivation: the corpus it was
checked against, the two rounds of correction, the per-signature file
counts, and the signatures that were explicitly checked and rejected.

**No per-component frequency counts** (e.g. "`epix100_N` crashed in 55 of
5,327 files") are shipped here, by design — those decay the moment the
corpus they were derived from keeps growing, and nothing re-derives them.
If that breakdown is ever needed, re-run it against the current corpus
rather than trusting a number from this file.

---

## Corpus

Checked directly against the real xpp DAQ log corpus (5,327 files,
`/sdf/home/x/xppopr/daq/logs/2026/09/`) — two rounds of correction from an
initial broader proposal; the definition below is what survived.

## Signatures that count as a crash

- **Python exception:** `Traceback (most recent call last)` (primary
  signal — 184 files in the corpus) · `<C> **** python error` (this DAQ's
  own wrapper marker — 147 files, always co-occurs with `Traceback`) · a
  terminating `<Type>Error:` line (158 files) · `rogue.GeneralError` noted
  as the dominant concrete type (54 files — mostly epix100/hsd firmware
  register-transaction timeouts). Report the exception type and the
  failing call.
- **Segfault:** `Segmentation fault` (9 files) · `(core dumped)`
  (33 files, appears alongside both `Aborted` and `Segmentation fault`).

## Explicitly excluded — checked and rejected as signatures

- Bare `Exception:` (1,251 files, but 4,880 of those lines are `"No
  credentials were supplied"` Kerberos noise — not a crash signal).
- `task N: Terminated` (verified, by reading surrounding context, to be
  preceded in every sampled case by `JOB ... CANCELLED ... DUE to SIGNAL
  Terminated` — deliberate operator shutdown, not a crash).
- `task N: Killed` and `terminate called` (1 and 3 files respectively —
  too rare in this corpus to justify as independent signatures).

## Secondary note, not a gate

84 of 231 files containing bare `task N: Aborted` carry neither a
traceback nor a segfault in that same file — the proximate cause is
elsewhere. Treat this as guidance ("check sibling logs from the same
session for the actual cause") rather than a crash-detection trigger,
since gating on bare `Aborted` would flag deaths with no diagnostic
content in `## New Crashes`.

## Why two rounds of correction

The initial broader proposal for this definition included more
signatures than survived. The exclusions above are not arbitrary
omissions — each was checked against the corpus and found to either (a)
be dominated by non-crash noise (bare `Exception:`), (b) actually
represent deliberate operator action (`task N: Terminated`), or (c) be
too rare to justify as an independent, reusable signature (`Killed`,
`terminate called`). This history is preserved here so a future
maintainer re-deriving or extending the definition doesn't re-propose and
re-reject the same signatures without this context.
