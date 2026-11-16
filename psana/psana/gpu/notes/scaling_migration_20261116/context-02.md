# Independent scaling migration review, round 2
Exact reviewed SHA: 9c4efbde2109ccccb56fe13edbb47ffbabc3db5d
Prior reviewed SHA: c817d22586052a41beb2400739e194d9a0105bf7
Scope unchanged: migration of five scaling records, frozen source dependency
closure, unified nine-entry catalog and source-only offline tools. No runtime,
GPU jobs, publication, dataset qualification or regression runner implementation.

Round 1 passed with one low non-blocking suggestion MIG-R1-01. It is accepted
and fixed. Review the fix and any introduced regression. The entire round-1
scope and evidence are committed in notes/scaling_migration_20261116/context-01.md
under psana/psana/gpu/. Read review-01.json, response-01.md, validation-02.json,
migration-02.log, catalog-02.log and tests-02.log alongside it. Review changed
scripts/performance/analysis/check_catalog.py and test_archive_checks.py.
The original 1631-file relocation has 1612 byte-exact files and all 359 original
addition paths plus 1272 dependencies. Only adapted checker/test destination
hashes changed in relocation.json; source identities and frozen files did not.

The new check selects cohort 45028778 full rows and requires unique
(media, gpus, repeat) tuples before controls are built. The negative test gives
two one-GPU Weka controls the same repeat while preserving distinct case IDs
and measurements, and requires the explicit duplicate-repeat error. No rates,
report tables or scientific states changed. Re-run: 12 focused tests passed;
all 240 rows, 2000 catalog artifact checks and 1631 source blobs verified.
Original archive/replay/isolation checks carry forward unchanged. A scoped diff
allowlist confirms only the checker, its tests, ledger and review records changed.

No commands/delegation/edits. Read/Glob/Grep only; don't claim to run tests.
Scientific qualifications from round 1 remain: five OSTs for count-1 files,
SDF four BDs total vs Perlmutter four per GPU, normalized SDF replay with missing
original inputs, cube pending, PR187 loop equivalence unresolved, PR299 cache
uncontrolled/C1 unfixed/Run77 deferred. Known limitations unchanged: no full
native build/wheel or GPU tests; Ruff/pre-commit binaries absent, configuration
checked; upstream URLs are pending publication, local index works.

Follow-up validation:
{
  "previous_reviewed_commit": "c817d22586052a41beb2400739e194d9a0105bf7",
  "migration": {
    "status": "pass",
    "paths": 1631,
    "frozen_files": 1612,
    "original_additions": 359,
    "inherited_dependencies": 1272,
    "source_git_blobs": 1631,
    "limitations": [
      "Literal-reference closure includes full source roots and provenance references; external runtime/data paths are not reconstructed.",
      "Historical SDF missing-input receipt is preserved; no new cross-site availability claim."
    ]
  },
  "catalog": {
    "status": "pass",
    "campaigns": 9,
    "file_hashes_checked": 2000,
    "arithmetic_cases": 240,
    "scope": "Archive hashes, authored links/tables, taxonomy, per-profile populations and arithmetic; no native raw-data re-audit"
  },
  "focused_tests": "12 passed",
  "fix": "MIG-R1-01: reject duplicate SDF (media, gpus, repeat) keys within cohort 45028778 full rows before paired controls are built.",
  "unchanged": "Frozen archives, report text/manifests, SDF replay code, package/discovery configuration, PR study sources, runtime and examples; their validation-01 results carry forward.",
  "checks_exit_codes": {
    "migration": 0,
    "catalog": 0,
    "tests": 0
  }
}


Coordinator-generated exact code diff:
```diff
diff --git a/psana/psana/gpu/scripts/performance/analysis/check_catalog.py b/psana/psana/gpu/scripts/performance/analysis/check_catalog.py
index 4bba74d37..04a315279 100644
--- a/psana/psana/gpu/scripts/performance/analysis/check_catalog.py
+++ b/psana/psana/gpu/scripts/performance/analysis/check_catalog.py
@@ -86,6 +86,9 @@ def check_sdf(record, report):
     sdf_rows = rows(sdf / "runs.csv")
     require(Counter(r["phase"] for r in sdf_rows) == {"full": 102, "pilot": 48, "anchor": 6}, "SDF case inventory")
     require(len({(r["cohort"], r["case"]) for r in sdf_rows}) == 156, "SDF duplicate case")
+    storage_rows = [r for r in sdf_rows if r['cohort'] == '45028778' and r['phase'] == 'full']
+    repeat_keys = [(r['media'], r['gpus'], r['repeat']) for r in storage_rows]
+    require(len(set(repeat_keys)) == len(repeat_keys), 'SDF storage/GPU duplicate repeat')
     fields = ("cohort", "media", "gpus", "bds", "workers", "streams", "batch", "task_mib", "bounce_mib")
     groups = defaultdict(list)
     for row in sdf_rows:
diff --git a/psana/psana/gpu/scripts/performance/analysis/test_archive_checks.py b/psana/psana/gpu/scripts/performance/analysis/test_archive_checks.py
index b5a9420f5..2fad99ce0 100644
--- a/psana/psana/gpu/scripts/performance/analysis/test_archive_checks.py
+++ b/psana/psana/gpu/scripts/performance/analysis/test_archive_checks.py
@@ -55,6 +55,22 @@ class ArchiveChecks(unittest.TestCase):
             with self.assertRaisesRegex(ValueError, 'Scaling duplicate case'):
                 catalog.check_scaling({}, None)
 
+    def test_sdf_repeated_control_repeat_rejected(self):
+        original = catalog.rows
+        def changed(path):
+            values = original(path)
+            if path.name == 'runs.csv':
+                controls = [r for r in values if r['cohort'] == '45028778' and r['phase'] == 'full'
+                            and r['media'] == 'weka' and r['gpus'] == '1']
+                self.assertGreaterEqual(len(controls), 2)
+                controls[1]['repeat'] = controls[0]['repeat']
+                # Keep distinct case IDs and values; only repeat identity is corrupt.
+                self.assertNotEqual(controls[0]['case'], controls[1]['case'])
+            return values
+        with patch.object(catalog, 'rows', side_effect=changed):
+            with self.assertRaisesRegex(ValueError, 'SDF storage/GPU duplicate repeat'):
+                catalog.check_sdf({}, None)
+
     def test_manifest_taxonomy(self):
         path = catalog.ROOT / "campaigns/perlmutter/sim10-staging/date/manifest.json"
         catalog.validate_location(dict(site="perlmutter", workload="sim10-staging"), path)

```
Return schema verdict with exact SHA; close/reuse MIG-R1-01 as warranted.
