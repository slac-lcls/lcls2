"""Read-only integrity and arithmetic checks for the scaling results archive."""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from paths import find_repository


def configure(repository=None):
    global REPO, ROOT, ARCHIVE, LEGACY
    REPO = find_repository(repository)
    ROOT = REPO / 'psana/psana/gpu/docs/performance'
    ARCHIVE = REPO / 'psana/psana/gpu/notes/archives/scaling-20261116'
    LEGACY = ARCHIVE / 'legacy-root'


configure()
BYTES_PER_EVENT = 3358200


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def repository_path(relative):
    require(isinstance(relative, str) and relative, 'Empty repository path')
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'Unsafe repository path: ' + relative)
    target = REPO / p
    require(target.resolve().is_relative_to(REPO), 'Path outside repository: ' + relative)
    require(not target.is_symlink(), 'Symlinked artifact: ' + relative)
    return target


def verify_file(record):
    path = repository_path(record['path'])
    require(path.is_file(), 'Missing artifact: ' + str(path))
    if 'bytes' in record:
        require(path.stat().st_size == record['bytes'], 'Size mismatch: ' + str(path))
    require(hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], 'Hash mismatch: ' + str(path))


def near(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12),
            "Arithmetic mismatch: " + label)


def expect_cells(cells, fields, expected, label):
    keys = [tuple(cell[field] for field in fields) for cell in cells]
    require(len(keys) == len(expected) and set(keys) == set(expected), label + " cell population")


def validate_location(record, manifest_path):
    parts = manifest_path.relative_to(ROOT).parts
    require(len(parts) == 5 and parts[0] == "campaigns" and parts[-1] == "manifest.json",
            "Campaign path structure")
    require(record["site"] == parts[1] and record["workload"] == parts[2],
            "Campaign site/workload taxonomy mismatch")


def check_readme_table(text, expected, label):
    table = [line for line in text.splitlines() if line.startswith("|")]
    actual = [[cell.strip() for cell in line.strip("|").split("|")] for line in table[2:]]
    require(actual == expected, label + " README table differs from source summaries")


def check_sdf(record, report):
    # Recompute source summaries without pooling allocations or phases.
    sdf = ARCHIVE / "sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/artifacts"
    sdf_rows = rows(sdf / "runs.csv")
    require(Counter(r["phase"] for r in sdf_rows) == {"full": 102, "pilot": 48, "anchor": 6}, "SDF case inventory")
    require(len({(r["cohort"], r["case"]) for r in sdf_rows}) == 156, "SDF duplicate case")
    fields = ("cohort", "media", "gpus", "bds", "workers", "streams", "batch", "task_mib", "bounce_mib")
    groups = defaultdict(list)
    for row in sdf_rows:
        near(row["gib_s"], int(row["events"]) * BYTES_PER_EVENT / 2**30 / float(row["seconds"]), "SDF rate")
        if row["phase"] == "full":
            groups[tuple(row[k] for k in fields)].append(row)
    summaries = rows(sdf / "summary.csv")
    require(len(groups) == 48, "SDF cells")
    expect_cells(summaries, fields, set(groups), "SDF")
    for summary in summaries:
        group = groups[tuple(summary[k] for k in fields)]
        require(len(group) == int(summary["n"]), "SDF repeat count")
        near(summary["mean_gib_s"], statistics.mean(float(r["gib_s"]) for r in group), "SDF mean")

    sdf_table = []
    for media in ("weka", "nvme"):
        group = [r for r in sdf_rows if r["cohort"] == "45028778" and r["phase"] == "full" and r["media"] == media]
        controls = {r["repeat"]:float(r["gib_s"]) for r in group if r["gpus"] == "1"}
        for g in (1,2,4):
            selected = [r for r in group if int(r["gpus"]) == g]
            rates = [float(r["gib_s"]) for r in selected]
            paired = statistics.mean(float(r["gib_s"]) / controls[r["repeat"]] for r in selected)
            sdf_table.append(["Weka" if media == "weka" else "NVMe", str(g), str(len(selected)),
                              f'{statistics.mean(rates):.4f}', f'{min(rates):.4f}–{max(rates):.4f}', f'{paired:.4f}×'])
    check_readme_table(report.read_text(), sdf_table, "SDF storage/GPU")
    return len(sdf_rows)


def check_cpp(record, report):
    cpp = LEGACY / "notes/sim10_striping_20261110/reader/sweep"
    cpp_rows = rows(cpp / "trials.csv")
    require(len(cpp_rows) == 54, "C++ case count")
    require(len({r["id"] for r in cpp_rows}) == 54 and len({(r["block"],r["count"],r["threads"]) for r in cpp_rows}) == 54, "C++ duplicate case")
    for row in cpp_rows:
        near(row["gib_per_second"], 335820951264 / 2**30 / float(row["seconds"]), "C++ rate")
    cpp_cells = read_json(cpp / "audit.json")["cells"]
    expect_cells(cpp_cells, ("count", "threads"), {(c, t) for c in (1,2,4,8,16,32) for t in (5,20,40)}, "C++")
    for cell in cpp_cells:
        values = [float(r["gib_per_second"]) for r in cpp_rows if int(r["count"]) == cell["count"] and int(r["threads"]) == cell["threads"]]
        require(len(values) == 3, "C++ cell repeat count")
        near(cell["median_gib_s"], statistics.median(values), "C++ median")

    cpp_table = [[str(c)] + [format(next(x["median_gib_s"] for x in cpp_cells if x["count"] == c and x["threads"] == t), ".3f")
                             for t in (5,20,40)] for c in (1,2,4,8,16,32)]
    check_readme_table(report.read_text(), cpp_table, "C++")
    return len(cpp_rows)


def check_stripe(record, report):
    stripe = LEGACY / "notes/psana_striping_pr298_20261111/screen-20261115/evidence/screen01"
    stripe_rows = rows(stripe / "trials.csv")
    require(len(stripe_rows) == 18, "Stripe case count")
    require(len({r["id"] for r in stripe_rows}) == 18 and len({(r["block"],r["count"]) for r in stripe_rows}) == 18, "Stripe duplicate case")
    for row in stripe_rows:
        near(row["gib_s"], int(row["events"]) * BYTES_PER_EVENT / 2**30 / float(row["loop_seconds"]), "Stripe rate")
    stripe_cells = read_json(stripe / "summary.json")
    expect_cells(stripe_cells, ("count",), {(c,) for c in (1,2,4,8,16,32)}, "Stripe")
    for cell in stripe_cells:
        values = [float(r["gib_s"]) for r in stripe_rows if int(r["count"]) == cell["count"]]
        require(len(values) == 3, "Stripe repeats")
        near(cell["median_gib_s"], statistics.median(values), "Stripe median")

    stripe_table = [[str(c["count"]), f'{c["median_gib_s"]:.4f}', f'{c["min_gib_s"]:.4f}–{c["max_gib_s"]:.4f}']
                    for c in sorted(stripe_cells, key=lambda c: c["count"])]
    check_readme_table(report.read_text(), stripe_table, "Stripe")
    return len(stripe_rows)


def check_scaling(record, report):
    scaling = LEGACY / "notes/psana_gpu_scaling_20261116/evidence/screen01-results"
    audit = read_json(scaling / "audit.json")
    cases = audit["cases"]
    require(audit["state"] == "pass" and len(cases) == 12, "Scaling audit/count")
    require(len({c["id"] for c in cases}) == 12 and len({(c["id"].split("-")[0],c["gpus"]) for c in cases}) == 12, "Scaling duplicate case")
    require(all(c["events"] == 100000 and c["count"] == 1 for c in cases), "Scaling event/layout identity")
    baseline = {c["id"].split("-")[0]: c["gib_s"] for c in cases if c["gpus"] == 1}
    for c in cases:
        near(c["gib_s"], c["events"] * BYTES_PER_EVENT / 2**30 / c["seconds"], "Scaling rate")
    scaling_cells = read_json(scaling / "scaling-summary.json")["rows"]
    expect_cells(scaling_cells, ("gpus",), {(g,) for g in (1,2,3,4)}, "Scaling")
    for cell in scaling_cells:
        selected = [c for c in cases if c["gpus"] == cell["gpus"]]
        require(len(selected) == 3, "Scaling repeats")
        near(cell["median_gib_s"], statistics.median(c["gib_s"] for c in selected), "Scaling median")
        near(cell["median_paired_speedup"], statistics.median(c["gib_s"] / baseline[c["id"].split("-")[0]] for c in selected), "Scaling paired speedup")
    scaling_table = [[str(c["gpus"]), str(c["bds"]), f'{c["median_gib_s"]:.3f}',
                      f'{c["min_gib_s"]:.3f}–{c["max_gib_s"]:.3f}', f'{c["median_paired_speedup"]:.2f}×', f'{c["efficiency"]:.1%}']
                     for c in sorted(scaling_cells, key=lambda c: c["gpus"])]
    check_readme_table(report.read_text(), scaling_table, "Scaling")
    return len(cases)


def check_pending(record, report):
    require(record['status'] == 'pending' and record['full_cases'] == 0, 'Pending cube promoted to results')
    return 0


def check_artifacts(record, report):
    # Explicit structural/artifact-only profile, never a performance acceptance.
    return 0


ANALYSES = {'sdf-staging-v1': check_sdf, 'cpp-stripes-v1': check_cpp,
            'psana-stripes-v1': check_stripe, 'psana-gpus-v1': check_scaling,
            'cube-pending-v1': check_pending, 'documentary-v1': check_artifacts,
            'artifact-only-v1': check_artifacts}
KINDS = {'scaling', 'regression', 'application', 'diagnostic'}


def check_links(doc):
    for target in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", doc.read_text()):
        if '://' not in target and not target.startswith(('#', '/')):
            path = (doc.parent / target.split('#')[0]).resolve()
            require(path.is_relative_to(REPO) and path.exists(), 'Broken link: ' + str(doc) + ' -> ' + target)


def check_catalog_structure(catalog):
    require(catalog['schema_version'] == 2, 'Catalog schema')
    ids = set()
    checked = 0
    records = []
    for entry in catalog['campaigns']:
        ident = entry['id']
        require(re.fullmatch(r'[a-z0-9][a-z0-9-]*', ident), 'Unsafe campaign ID')
        require(ident not in ids, 'Duplicate campaign')
        ids.add(ident)
        manifest = repository_path(entry['manifest'])
        directory = repository_path(entry['record_path'])
        report = repository_path(entry['report'])
        require(manifest.parent == directory and report.parent == directory, 'Record path mismatch')
        require(manifest.name == 'manifest.json' and report.name == 'README.md', 'Record filenames')
        require(manifest.is_file() and report.is_file(), 'Missing campaign record')
        require(manifest.is_relative_to(ROOT), 'Manifest outside catalog')
        record = read_json(manifest)
        validate_location(record, manifest)
        require(record['schema_version'] == 2, 'Manifest schema')
        require(record['id'] == ident, 'Campaign identity mismatch')
        require(record['kind'] in KINDS and record['kind'] == entry['kind'], 'Campaign kind')
        require(record['analysis'] in ANALYSES, 'Unknown analysis profile')
        require(record['script_roots'] == entry['script_roots'], 'Script roots mismatch')
        for root in entry['script_roots']:
            path = repository_path(root)
            require(path.is_dir(), 'Missing script root')
            orchestration = REPO / 'psana/psana/gpu/scripts/campaigns'
            if path.is_relative_to(orchestration):
                require(path.relative_to(orchestration).parts[0] == ident, 'Script directory ID mismatch')
        require(record['status'] in {'planned','pending','running','complete','failed','inconclusive'}, 'Campaign status')
        require(record['review'] and record['metric']['timing'], 'Missing review or timing scope')
        require(record['lifecycle']['execution'] == record['status'], 'Execution status mismatch')
        require(set(record['lifecycle']) == {'execution','correctness','results_review','archive_verification'}, 'Lifecycle axes')
        if record['status'] == 'complete' and record['analysis'] not in {'documentary-v1','artifact-only-v1'}:
            require(record['full_cases'] > 0 and record['jobs'], 'Completed campaign lacks measurements')
        for item in record['artifacts']:
            verify_file(item)
            checked += 1
        check_links(report)
        records.append((record, report))
    return records, checked


def main(repository=None):
    if repository is not None:
        configure(repository)
    records, checked = check_catalog_structure(read_json(ROOT / 'catalog.json'))
    ledger = read_json(REPO / 'psana/psana/gpu/notes/scaling_migration_20261116/relocation.json')
    mapping = {e['old_path']: e['new_path'] for e in ledger['files']}
    for item in ledger['files']:
        if item['action'] == 'moved-byte-exact':
            verify_file(dict(path=item['new_path'], bytes=item['bytes'], sha256=item['source_sha256']))
            checked += 1
    # Original import receipt stays byte-exact; resolve its old paths through the ledger.
    imports = read_json(ARCHIVE / 'sources/IMPORTS.json')
    for item in imports['files']:
        require(item['path'] in mapping, 'Unmapped original receipt')
        verify_file(dict(item, path=mapping[item['path']]))
        checked += 1
    for doc in [ROOT/'README.md', ROOT/'FORMAT.md', ROOT/'templates/README.md', ROOT/'campaigns/README.md',
                REPO/'psana/psana/gpu/scripts/README.md', REPO/'psana/psana/gpu/scripts/performance/README.md',
                REPO/'psana/psana/gpu/scripts/performance/analysis/README.md', ARCHIVE/'README.md']:
        check_links(doc)
    cases = sum(ANALYSES[record['analysis']](record, report) for record, report in records)
    result = dict(status='pass', campaigns=len(records), file_hashes_checked=checked,
                  arithmetic_cases=cases,
                  scope='Archive hashes, authored links/tables, taxonomy, per-profile populations and arithmetic; no native raw-data re-audit')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, help='Source repository root; otherwise locate by source-tree markers')
    args = parser.parse_args()
    main(args.repository)
