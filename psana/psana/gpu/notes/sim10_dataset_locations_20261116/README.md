# Published Sim10 dataset locations: implementation and review

Documentation-only PR for `results/psana2-gpu-scaling`.
The [dataset guide](../../docs/performance/datasets.md) records published SDF
`prjdatpsana` runs 3 and 4 and the prerequisites for any later source cleanup.

## Review outcome

- Independent restricted read-only Claude review, coordinated by Codex.
- Round 1: pass with two non-blocking wording suggestions; both accepted/fixed.
- Round 2: pass with no findings on `be12fc6abecd5520fd9b355240e9e4e0b6ece353`.
- This evidence-only follow-up does not change the reviewed implementation.
- [Scope](scope.md), [round-1 review](review-01.json), [response](response-01.md),
  [round-2 review](review-02.json), and [final response](response-02.md).

## Validation and evidence limits

[Validation](validation-02.json) records 10 catalog campaigns, 2,082 archive
hashes, 275 arithmetic cases, frozen bookkeeping verification, 20 regression
controller tests, relative links, and whitespace checks, all passing on the
reviewed implementation. No runtime or profile files changed.

[Publication reconciliation](source-verification.json) and
[400k source records](source-400k-verification.json) summarize read-only checks
against existing SDF receipts. These are not fresh full-file hashes, an event
scan, a live database query, a DataSource/GPU run, or PM transfer qualification.
The reviewer used the supplied evidence for locations outside its restricted
checkout; see the review's limitations. Logs and expanded reviewer contexts
remain in task storage, referenced with hashes in the validation and
[artifact manifest](review-artifacts.json); raw provider output and connection
settings are not published here.

No dataset cleanup is part of this PR. Preserve frozen historical paths and
hashes, generation records, calibration evidence, and the private PM JF-only
subset. Confirm active consumers separately before any source removal.
