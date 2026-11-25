Implementation df42d1cd665a455eb8ec522c865d07f5146cea1a:
5 cleanup harness rejection tests PASS after exact-cause fixes.
30 native adapter tests and 20 campaign controller tests PASS before that commit;
their production implementations were unchanged. bash -n qualify.sbatch PASS;
git diff --check PASS. No code changes since the reviewed implementation.

Coordinator compared all five frozen-01 Python source files byte-for-byte with
git show df42d1cd6:path and verified SHA256SUMS before submission. Job 61502056
uses the frozen source; batch also verifies hashes before execution.
