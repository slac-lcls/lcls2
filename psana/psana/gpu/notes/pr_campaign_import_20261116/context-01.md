# Independent import review, round 1
Reviewed SHA: caffb52bb862d17044da690328cebeaa0d430b3e
Base before import: 12366915ee2e7991176c19b03d4a24134e76c7f3
Branch: results/psana2-gpu-scaling, a clean feature-derived local results branch.

User requests next step of campaign consolidation, with Claude review loop.
Scope: ONLY selectively import pinned PR187/208 and PR299 documentary packages,
preserve historical script/evidence bytes and PR299 paths, reconcile navigation,
and isolate frozen code from ordinary test collection/lint/format/install.
No runtime implementation, scaling-catalog migration, dataset retrieval, new
runner, GPU jobs, publication, PR creation or branch deletion.
The maintained regression template is a later/final step, not these snapshots.

Review all authored integration/checker/config changes and the scientific claims
in navigation against source reports. Spot-check ledger/evidence relationships;
coordinator supplied exhaustive cryptographic checks (not a requirement to read
182k imported lines). Source reports/harnesses are immutable historical evidence,
not newly maintained code: do not demand fixes to known old campaign code.
Do flag incorrect import, broken navigation, changed scientific conclusions,
missing required evidence, unsafe collection/packaging or runtime impact.

Inspect:
- psana/psana/gpu/notes/pr_campaign_import_20261116/{scope.md,README.md,imports.json,validation.json,check_isolation.py} and the five adjacent logs.
- psana/psana/gpu/scripts/performance/analysis/check_pr_import.py
- conftest.py, .pre-commit-config.yaml, pyproject.toml, psana/meson.build
- GPU README and docs/performance/{README.md,history.md,read_staging.md,user_kernels.md}
- PR187/208 original report docs/performance/pr187_pr208.md and archive scripts/pr187_pr208/manifest.json
- PR299 original notes/pr299_perlmutter_20261115.md, notes/evidence/pr299_perlmutter_20261115/ and scripts/pr299_perlmutter/ (including verifier/tests).
- Prior accepted organization and plan: notes/performance_campaign_organization_20261116/IMPLEMENTATION.md (still says step 1 is next; coordinator will update completion metadata after review).
All GPU-relative paths above are under psana/psana/gpu/.

Exact source pins:
PR187/208 66072235d3cd18c002a13c581af82e111a418758 (delta base 8cf05e86a7511f84a24f82b9c9720bc2308c471d): 66 paths, 61 exact and five reconciled navigation pages.
PR299 e8d69a7e6604e1d459ad04c633de50a56b46ac9a (delta base 1995752058bb9aaa3f16ccd86c22e84153257538): all 637 paths exact.
703 paths total, 700 added + 3 modified; 698 exact, 5 navigation reconciliations.
719 total staged paths includes import ledger/checks/logs and isolation config.
Coordinator checked changed-path allowlist: no runtime or examples altered.
Authored/navigation git diff --check passed; frozen originals intentionally retain original whitespace (CSV CRLF, logs/provenance trailing blank/space lines).
The runtime/existing docs/examples/harnesses otherwise remain unchanged.

Project guidance: label measured vs implemented; preserve original limits; no
hardware thresholds in pytest; don't infer GDS/cache correctness from throughput.
PR299 head measured acfd029... had unfixed C1, Run77 deferred, uncontrolled cache,
CPU controls are unequal work to GPU, no universal no-regression claim.
PR187/208 has remaining-loop equivalence uncertainty and historical analysis
qualifications. Report imports do not assert datasets remain retrievable.
Isolation checks are real offline pytest traversal/fixture, Meson minimal install,
and Hatch source/staged selection; not a full native build or wheel or actual
GPU test execution. Ruff/pre-commit binaries unavailable; config checks only.
Old README patch conflict and initial checker umask mismatch were corrected;
see validation record, do not hide these initial failures.

Validation evidence (coordinator-run, all passing; frozen input hashes bind to reviewed commit):
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


No prior findings. Return schema verdict, exact reviewed SHA and stable finding IDs.
Use available file-reading tools only; do not run commands or delegate.
