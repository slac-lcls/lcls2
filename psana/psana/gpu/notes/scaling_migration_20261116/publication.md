# Upstream publication — November 16, 2026

Published `results/psana2-gpu-scaling` to `slac-lcls/lcls2` using a normal,
non-force Git push. The branch did not exist in the pre-publication remote check.

- Initial published tip: `6df8e1ea1df0833292aa9314dc46f1b0ae63388f`.
- Reviewed implementation: `9c4efbde2109ccccb56fe13edbb47ffbabc3db5d`;
  Claude round 2 passed with no findings.
- `git ls-remote` confirmed the initial published tip exactly.
- GitHub connector readback returned the nine-campaign index from the branch
  and the passing `review-02.json` from the initial published commit.

This follow-up commit updates publication status and navigation text only,
including the adapted overview's destination hash in the relocation ledger.
It leaves reviewed code, numerical results and frozen artifacts unchanged.
Historical review records retain their original pre-publication context.

Post-update checks passed: all 1631 relocation paths and source blobs, 1612
frozen files, nine catalog campaigns, 2000 artifact hashes, 240 arithmetic
cases, and Git whitespace validation.

[Campaign index](../../docs/performance/campaigns/README.md) ·
[Current implementation plan](../performance_campaign_organization_20261116/IMPLEMENTATION.md).

Archive tagging and retirement of the three source branches are the next step;
source heads and open-PR dependencies must be rechecked first. No source branches
were deleted during publication. Dataset qualification and the maintained
regression template remain later work. No benchmark jobs were submitted.
