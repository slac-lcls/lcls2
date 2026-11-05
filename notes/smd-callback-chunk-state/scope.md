# SMD callback chunk-state regression

Branch: `tests/smd-callback-chunk-state`
Base: `71378658a54dcdd941c9137355a985fa4b637595`

Implement step 1 of the agreed plan only: demonstrate that local `istep` and
`ievt` enumeration in a Jungfrau-style SMD callback changes timestamp selection
when physical scan steps cross SMD chunks. Do not change runtime behavior,
Jungfrau processing, or MPI scheduling. Do not push or publish.

Acceptance:

- Generate a small local XTC run with multiple physical steps.
- Build an independent timestamp oracle using direct-file reading, without
  the SMD callback path.
- Exercise all-step event windows and a specific nonzero step, with an initial
  event skip and upper event-index limit matching the current Jungfrau filter.
- Compare exact timestamps in serial and single-EB MPI, varying SMD chunk size
  and output batch size. Include a whole-run chunk as a passing control.
- Mark only known selection mismatches as strict expected failures. Setup,
  subprocess, MPI, and unrelated assertion failures must remain failures.
- Run the tests normally and with `--runxfail` to demonstrate the underlying
  failures. No production-core changes are planned, so the focused matrix is
  the required runtime check; the full core suites belong to the later fix.

Codex implements; Claude independently reviews the committed changes in a
detached worktree using the restricted skill helper. Maximum three review
rounds, each with a 15-minute timeout. Preserve review evidence and responses.
