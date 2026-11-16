"""Regression checks for missing summary cells and misleading archive metadata."""
import copy
import csv
import importlib.util
import re
import unittest
from unittest.mock import patch

import check_catalog as catalog
import replay_sdf_report as replay


class ArchiveChecks(unittest.TestCase):
    def test_missing_or_duplicate_summary_cells(self):
        sources = [
            ("notes/sim10_striping_20261110/reader/sweep/audit.json", "cells", "C++"),
            ("notes/psana_striping_pr298_20261111/screen-20261115/evidence/screen01/summary.json", None, "Stripe"),
            ("notes/psana_gpu_scaling_20261116/evidence/screen01-results/scaling-summary.json", "rows", "Scaling"),
        ]
        original = catalog.read_json
        for relative, key, label in sources:
            target = catalog.LEGACY / relative
            for mutation in ("drop", "duplicate"):
                changed = copy.deepcopy(original(target))
                cells = changed[key] if key else changed
                if mutation == "drop":
                    cells.pop()
                else:
                    cells[-1] = copy.deepcopy(cells[0])
                def injected(path):
                    return changed if path == target else original(path)
                with self.subTest(source=relative, mutation=mutation):
                    with patch.object(catalog, "read_json", side_effect=injected):
                        with self.assertRaisesRegex(ValueError, re.escape(label + " cell population")):
                            catalog.main()

    def test_duplicate_trial_cases(self):
        for function, filename, label in [(catalog.check_cpp, 'trials.csv', 'C++'), (catalog.check_stripe, 'trials.csv', 'Stripe')]:
            original = catalog.rows
            def changed(path):
                values = original(path)
                if path.name == filename:
                    values[1] = copy.deepcopy(values[0])
                return values
            with self.subTest(label=label), patch.object(catalog, 'rows', side_effect=changed):
                with self.assertRaisesRegex(ValueError, re.escape(label + ' duplicate case')):
                    function({}, None)
        original = catalog.read_json
        def changed_audit(path):
            value = original(path)
            if path.name == 'audit.json':
                value['cases'][1] = copy.deepcopy(value['cases'][0])
            return value
        with patch.object(catalog, 'read_json', side_effect=changed_audit):
            with self.assertRaisesRegex(ValueError, 'Scaling duplicate case'):
                catalog.check_scaling({}, None)

    def test_sdf_repeated_control_repeat_rejected(self):
        original = catalog.rows
        def changed(path):
            values = original(path)
            if path.name == 'runs.csv':
                controls = [r for r in values if r['cohort'] == '45028778' and r['phase'] == 'full'
                            and r['media'] == 'weka' and r['gpus'] == '1']
                self.assertGreaterEqual(len(controls), 2)
                controls[1]['repeat'] = controls[0]['repeat']
                # Keep distinct case IDs and values; only repeat identity is corrupt.
                self.assertNotEqual(controls[0]['case'], controls[1]['case'])
            return values
        with patch.object(catalog, 'rows', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'SDF storage/GPU duplicate repeat'):
                catalog.check_sdf({}, None)

    def test_manifest_taxonomy(self):
        path = catalog.ROOT / "campaigns/perlmutter/sim10-staging/date/manifest.json"
        catalog.validate_location(dict(site="perlmutter", workload="sim10-staging"), path)
        for record in (dict(site="sdf", workload="sim10-staging"), dict(site="perlmutter", workload="staging")):
            with self.subTest(record=record):
                with self.assertRaisesRegex(ValueError, "taxonomy mismatch"):
                    catalog.validate_location(record, path)

    def test_bad_authored_table_value(self):
        text = "| GPUs | Rate |\n|---:|---:|\n| 1 | 50.75 |\n"
        with self.assertRaisesRegex(ValueError, "README table differs"):
            catalog.check_readme_table(text, [["1", "5.075"]], "Scaling")

    def test_links_in_sources_and_template_readmes(self):
        for relative in ("FORMAT.md", "templates/README.md"):
            target = catalog.ROOT / relative
            original = type(target).read_text
            def injected(path, *args, **kwargs):
                return original(path, *args, **kwargs) + ("\n[bad](not-a-real-catalog-file)\n" if path == target else "")
            with self.subTest(path=relative):
                with patch.object(type(target), "read_text", injected):
                    with self.assertRaisesRegex(ValueError, "Broken link"):
                        catalog.main()

    def test_replay_rejects_wrong_cohort_population(self):
        spec = importlib.util.spec_from_file_location("original_sdf_report", replay.SOURCE / "build_report.py")
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        with (replay.SOURCE / "artifacts/runs.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        replay.validate_population(rows, original.COHORTS)
        changed = copy.deepcopy(rows)
        row = next(r for r in changed if r["cohort"] == "44232284" and r["phase"] == "pilot")
        row["cohort"] = "44309691"
        row["case"] = "moved-pilot-for-regression-check"
        # Global phase counts still match; the cohort-local guard must catch this.
        with self.assertRaisesRegex(ValueError, "per-cohort population"):
            replay.validate_population(changed, original.COHORTS)
        duplicate = copy.deepcopy(rows)
        duplicate[1] = copy.deepcopy(duplicate[0])
        with self.assertRaisesRegex(ValueError, "Duplicate source case"):
            replay.validate_population(duplicate, original.COHORTS)


class CatalogContracts(unittest.TestCase):
    def test_unknown_schema_kind_analysis_and_duplicate_ids(self):
        original = catalog.read_json(catalog.ROOT / 'catalog.json')
        for mutation, message in [('schema','Catalog schema'), ('kind','Campaign kind'),
                                  ('analysis','Unknown analysis profile'), ('duplicate','Duplicate campaign')]:
            changed = copy.deepcopy(original)
            reader = catalog.read_json
            def altered(path):
                record = reader(path)
                if path.name == 'manifest.json':
                    if mutation == 'kind':
                        record['kind'] = 'unknown'
                    if mutation == 'analysis':
                        record['analysis'] = 'unknown'
                return record
            if mutation == 'schema':
                changed['schema_version'] = 999
            if mutation == 'duplicate':
                changed['campaigns'].append(copy.deepcopy(changed['campaigns'][0]))
            with self.subTest(mutation=mutation), patch.object(catalog, 'read_json', side_effect=altered):
                with self.assertRaisesRegex(ValueError, message):
                    catalog.check_catalog_structure(changed)

    def test_path_escape_missing_and_corrupt_artifacts(self):
        for path in ['../outside', '/etc/passwd']:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'Unsafe repository path'):
                catalog.verify_file(dict(path=path, sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'Missing artifact'):
            catalog.verify_file(dict(path='missing-campaign-artifact', sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
            catalog.verify_file(dict(path='pyproject.toml', sha256='0'*64))

    def test_script_roots_mismatch(self):
        changed = copy.deepcopy(catalog.read_json(catalog.ROOT / 'catalog.json'))
        changed['campaigns'][0]['script_roots'] = []
        with self.assertRaisesRegex(ValueError, 'Script roots mismatch'):
            catalog.check_catalog_structure(changed)

    def test_catalog_does_not_require_initial_five_campaigns(self):
        original = catalog.read_json(catalog.ROOT / 'catalog.json')
        selected = [x for x in original['campaigns'] if x['id'] == 'multi-site-pr187-pr208-20261028']
        records, checked = catalog.check_catalog_structure(dict(schema_version=2, campaigns=selected))
        self.assertEqual(len(records), 1)
        self.assertGreater(checked, 0)
        self.assertEqual(catalog.ANALYSES[records[0][0]['analysis']](*records[0]), 0)

    def test_replay_refuses_source_tree_output(self):
        with self.assertRaisesRegex(ValueError, 'outside the source checkout'):
            replay.main(catalog.REPO / 'bad-replay-output', False)


if __name__ == "__main__":
    unittest.main()
