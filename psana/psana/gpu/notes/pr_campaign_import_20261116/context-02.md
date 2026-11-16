# Independent import review, round 2
Exact reviewed SHA: 4d462252b80370fde35b12af9d490ecef5c28fbc
Task base: 12366915ee2e7991176c19b03d4a24134e76c7f3
Prior reviewed commit: caffb52bb862d17044da690328cebeaa0d430b3e
Branch: results/psana2-gpu-scaling

Scope unchanged: selectively import PR187/208 and PR299 documentary packages,
preserve frozen campaign bytes/PR299 paths, reconcile navigation and isolate
historical code from collection/rewrite/install. No runtime change, scaling
migration, runner implementation, hardware jobs, publication or PR creation.
Project guidance: distinguish measured and current behavior; preserve original
scientific limits; no hardware performance assertions in routine pytest.

Prior review passed with two low non-blocking suggestions; both now addressed.
Read notes/pr_campaign_import_20261116/{review-01.json,response-01.md,README.md,
imports.json,check_pr_import.py,validation-02.json,import-check-02.log} under
psana/psana/gpu/. Original validation.json and five logs remain unchanged.
Check both findings and any introduced regression. Rest of round-1 import
has no changes: reports, original scripts/evidence, navigation, conftest,
.pre-commit-config.yaml, pyproject.toml and psana/meson.build.
The checker moved from scripts/performance/analysis to the import notes so it
is source-only and not installed. Semicolon statements split. New ledger fields
capture actual source-byte equality and pre-import/source delta-base identities.
Policies remain 698 byte-preserved, five reconciled-navigation; actual bytes are
700 identical / three different. read_staging/user_kernels were reconciled but
needed only the original source patch: pre-import/source delta-base hashes match.
The checker confirms those hashes under --source-git. No frozen source changes.

Original source pins unchanged:
PR187/208 66072235d3cd18c002a13c581af82e111a418758 (66 paths)
PR299 e8d69a7e6604e1d459ad04c633de50a56b46ac9a (637 paths).
Required checks: source ledger/delta/hash and 109 links; PR299 verifier plus five
negative/positive tests; PR187 47 hashes and six tests; isolation fixtures.
All original tests passed; review fixes changed only import records/checker.
Affected checker rerun passed and authored git diff --check passed. No full
native build/wheel or actual GPU tests. Ruff/pre-commit binaries unavailable;
configuration coverage only. Reviewer must not claim to run tests.

Updated validation:
{
  "previous_reviewed_commit": "caffb52bb862d17044da690328cebeaa0d430b3e",
  "recheck": {
    "imported_paths": 703,
    "policies": {
      "reconciled-navigation": 5,
      "byte-preserved": 698
    },
    "source_identical_files": 700,
    "source_different_files": 3,
    "source_git_blobs": 703,
    "pr187_snapshots": 47,
    "report_links": 109,
    "limitations": [
      "Integrity/navigation check, not a new performance result or original raw-data replay.",
      "Historical absolute scratch links and remote URLs are not availability-checked."
    ]
  },
  "unchanged_checks": "Original PR299 archive/verifier/tests, PR187 snapshots/tests, isolation checks and packaging/configuration unchanged; their passed validation.json results carry forward.",
  "diff_scope": "Only import record and relocation/clarification of its checker; no campaign archive, navigation, runtime, examples or package-config changes.",
  "source_equality": "700 exact source files including two navigation-policy pages; three navigation files differ from source."
}


Original validation for unchanged archives/configuration:
{
  "base": "12366915ee2e7991176c19b03d4a24134e76c7f3",
  "date": "2026-11-16",
  "state": "precommit validation; exact commit binding follows in review context",
  "checks": {
    "import_integrity": {
      "imported_paths": 703,
      "policies": {
        "reconciled-navigation": 5,
        "byte-preserved": 698
      },
      "source_git_blobs": 703,
      "pr187_snapshots": 47,
      "report_links": 109,
      "limitations": [
        "Integrity/navigation check, not a new performance result or original raw-data replay.",
        "Historical absolute scratch links and remote URLs are not availability-checked."
      ]
    },
    "pr299_verifier": {
      "package_files": 637,
      "byte_preserved_files": 624,
      "original_checksum_entries": 1171,
      "preflights": 12,
      "timed_samples": 48,
      "pairs": 24,
      "measured_head": "acfd029a8d27898b5f165e02facd3c3bbc9c4ad2",
      "manifest_sha256": "1433a2107fd8dd6ec589ce4fc8863b78df39e45d1e933e2fbca8814f6ce1eabd",
      "limitations": [
        "Recorded event digests, not discarded event-array replay.",
        "Storage rollup checked against derived summaries; raw storage logs unavailable.",
        "Sizing binding uses original-tree hash index, not omitted raw files."
      ]
    },
    "pr299_unittest": "5 passed",
    "pr187_unittest": "6 passed",
    "isolation": {
      "pytest_fixture": "1 maintained test passed; all frozen sentinel imports excluded",
      "actual_gpu_discovered_test_files": [
        "psana/psana/gpu/scripts/feespec_bulk_benchmark/test_cache_repair.py",
        "psana/psana/gpu/scripts/feespec_bulk_benchmark/test_harness.py",
        "psana/psana/gpu/scripts/jf_scaling/test_cache_preflight.py",
        "psana/psana/gpu/scripts/jf_scaling/test_contract.py",
        "psana/psana/gpu/scripts/jf_scaling/test_feespec.py",
        "psana/psana/gpu/scripts/stage1_regression/test_analyze_profiles.py",
        "psana/psana/gpu/scripts/stage1_regression/test_matrix.py"
      ],
      "actual_discovery_note": "pytest traversal with import-free probe collectors; no maintained GPU tests executed",
      "meson_fixture_installed": [
        "gpu/docs/design.md",
        "gpu/examples/input_only.py",
        "gpu/gpu_events.py",
        "gpu/scripts/jf_scaling/bench.py"
      ],
      "hatch_selection": "source and container-staged install routes passed",
      "rewrite_lint_configuration": "Ruff TOML and five pre-commit exclusions checked; tools not installed here",
      "limitations": [
        "No full native build or production wheel was built.",
        "Ruff/pre-commit binaries unavailable; configuration checks only."
      ]
    }
  },
  "logs": {
    "import-check": {
      "file": "import-check.log",
      "sha256": "39365502fd60e8925021bc7fc2b3711178e78461f31061d077d16b3770074c9b"
    },
    "pr299-verify": {
      "file": "pr299-verify.log",
      "sha256": "e8cd85120254c6f8f3b1324a33bca47c9cb9a83b5513bcb0739204a7b8d0cc34"
    },
    "pr299-tests": {
      "file": "pr299-tests.log",
      "sha256": "b6acfea150f6cd0a99d3fea1fb1477eadd641840346b1909d4da5e7eec7bf7c2"
    },
    "pr187-tests": {
      "file": "pr187-tests.log",
      "sha256": "3ef4de7d704f916249e3b35903258b0d09627d6b3a3ea85a476ecf6a77d8c2a6"
    },
    "isolation": {
      "file": "isolation.log",
      "sha256": "a70d536c0cc1a0e3ec4f7b00de149af3d2b2343589ffc4a51d2b334d1d9f8c2d"
    }
  },
  "resolved_implementation_issues": [
    "Old GPU README patch conflicted with current navigation; reconciled manually, preserving current API links.",
    "Initial checker compared tar POSIX write permissions with umask-adjusted worktree permissions; corrected to Git executable-bit semantics. No source bytes changed."
  ],
  "limitations": [
    "No full native build or production wheel.",
    "Ruff/pre-commit binaries unavailable; configuration checks only.",
    "No GPU work, data retrieval, or replay of original raw measurements."
  ]
}


For the prior limitation about unseen pre-import pages, coordinator-generated
exact task-base-to-reviewed-commit diff of the three existing navigation pages:
```diff
diff --git a/psana/psana/gpu/README.md b/psana/psana/gpu/README.md
index 7ac7f8362..747122790 100644
--- a/psana/psana/gpu/README.md
+++ b/psana/psana/gpu/README.md
@@ -85,6 +85,9 @@ for the worked examples.
 | [GPU XTC parser](docs/gpu_xtc_parser.md) | Configure tables, device locators and segment-preserving field access |
 | [Alignment callpaths and lifetime](docs/issue159_callpath_and_lifetime_walkthrough.md) | Worked buffer/lease examples and historical alignment implementation |
 | [Limits and open work](docs/limitations.md) | Supported scope, configuration restrictions and unresolved issues |
+| [Performance overview](docs/performance/README.md) | Dated PR studies and links to workload-specific reports |
+| [Benchmark hardware](docs/performance/hardware.md) | Dated NVMe/PCIe inventory and cache semantics |
+| [Performance history](docs/performance/history.md) | Major changes, PRs and measured revisions |
 | [Read/staging performance](docs/performance/read_staging.md) | Latest full JF and partial JF+feespec cold/warm, bulk off/on matrices |
 | [User-kernel performance](docs/performance/user_kernels.md) | Matched scheduling comparison and calibration/integration scaling |
 | [CPU/GPU complexity](docs/complexity.md) | Measured code size and responsibility comparison |
@@ -109,6 +112,6 @@ Current design and API pages describe the source, not development stages. The
 performance pages identify their measured revisions and workloads; those rates
 are not universal guarantees. Completed stage plans, handoffs and older reports
 live in [Git history](https://github.com/slac-lcls/lcls2/tree/fa40ec52a/psana/psana/gpu/docs).
-New measurements replace the relevant current report while preserving compact
-provenance. [AMI integration](docs/proposals/ami_integration.md) remains a separate
+New studies update the performance overview and history while retaining dated
+reports, measured revisions and compact provenance. [AMI integration](docs/proposals/ami_integration.md) remains a separate
 proposal, not an implemented interface.
diff --git a/psana/psana/gpu/docs/performance/read_staging.md b/psana/psana/gpu/docs/performance/read_staging.md
index 62e335130..799851248 100644
--- a/psana/psana/gpu/docs/performance/read_staging.md
+++ b/psana/psana/gpu/docs/performance/read_staging.md
@@ -1,6 +1,10 @@
 # JF and JF+feespec read/staging performance
 
-Latest revision comparisons: October 16–17, 2026 (summarized below).
+For the October 27–28 PR187/208 comparisons, see the [revision report](pr187_pr208.md).
+Use the [overview](README.md) and [hardware inventory](hardware.md) to select
+comparable measurements. The scaling tables below retain their historical scope.
+
+Revision comparisons in this report: October 16–17, 2026 (summarized below).
 The topology scaling campaigns are from September 28, runtime `10df4c6e3`.
 These measure JF read, GPU XTC parsing and dense raw gathering **without a
 GpuTask or calibration**. A benchmark-only input adapter requests the gather.
diff --git a/psana/psana/gpu/docs/performance/user_kernels.md b/psana/psana/gpu/docs/performance/user_kernels.md
index 85f0d5ee2..b2b5771e7 100644
--- a/psana/psana/gpu/docs/performance/user_kernels.md
+++ b/psana/psana/gpu/docs/performance/user_kernels.md
@@ -1,5 +1,9 @@
 # Batched user-kernel performance
 
+For newer calibration-only PR187/208 measurements, see the [revision report](pr187_pr208.md).
+The calibration-plus-integration results below remain a separate workload.
+See the [overview](README.md) and [history](history.md) for navigation.
+
 Latest accepted campaigns: September 28–29, 2026, runtime/algorithms
 `6ba5fa586`. Both scheduling variants run the same external Jungfrau calibration
 and radial-integration kernels and produce independent float64 `(3, 64)` host

```

Review verdict and stable findings under the required schema; reuse IDs.
Do not edit, delegate, run commands, or launch another loop. All necessary
files are in this detached checkout; this prompt embeds external validation.
