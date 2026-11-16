"""Replay archived SDF normalized measurements without requiring missing cohort CSVs."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

from paths import find_repository
ROOT = find_repository()
SOURCE = ROOT / 'psana/psana/gpu/notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114'
INT_FIELDS = ("sequence", "repeat", "gpus", "bds", "workers", "streams", "batch", "events",
              "payload_bytes", "bound_physical_cores", "total_workers", "bulk_bytes")
FLOAT_FIELDS = ("task_mib", "bounce_mib", "seconds", "hz", "gib_s", "nic_rx_payload_ratio", "budget_gib_per_bd")


def validate_population(rows, cohorts):
    if Counter(r["phase"] for r in rows) != {"full": 102, "pilot": 48, "anchor": 6}:
        raise ValueError("Unexpected source population")
    if len({(r["cohort"], r["case"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate source case")
    if {r["cohort"] for r in rows} != set(cohorts):
        raise ValueError("Unexpected cohort identities")
    for cohort, (_, _, full, pilots, _) in cohorts.items():
        selected = [r for r in rows if r["cohort"] == cohort]
        if sum(r["phase"] == "full" for r in selected) != full or sum(r["phase"] == "pilot" for r in selected) != pilots:
            raise ValueError("Unexpected per-cohort population: " + cohort)


def main(output, render, repository=None):
    global ROOT, SOURCE
    if repository is not None:
        ROOT = find_repository(repository)
        SOURCE = ROOT / 'psana/psana/gpu/notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114'
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError('Replay output must be outside the source checkout')
    spec = importlib.util.spec_from_file_location("archived_sdf_report", SOURCE / "build_report.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    with (SOURCE / "artifacts/runs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in INT_FIELDS:
            row[key] = int(row[key])
        for key in FLOAT_FIELDS:
            row[key] = float(row[key])
        if row["disk_payload_ratio"]:
            row["disk_payload_ratio"] = float(row["disk_payload_ratio"])
        original.validate_row(row)
    validate_population(rows, original.COHORTS)
    cells = original.summarize(rows)
    with (SOURCE / "artifacts/summary.csv").open(newline="") as stream:
        expected = list(csv.DictReader(stream))
    if len(cells) != len(expected):
        raise ValueError("Summary row count changed")
    for cell, archived in zip(cells, expected):
        for key, value in archived.items():
            actual = cell[key]
            if isinstance(actual, (int, float)):
                if not math.isclose(actual, float(value), rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("Summary differs: " + key)
            elif str(actual) != value:
                raise ValueError("Summary key differs: " + key)
    compared = []
    for panel in original.make_panels(cells):
        for index, row in enumerate(panel["grid"]):
            for cell in row:
                if cell is not None:
                    baseline = panel["baselines"][index]
                    compared.append(dict(panel=panel["title"], configuration={k:cell[k] for k in original.CONFIG_FIELDS},
                        control={k:baseline[k] for k in original.CONFIG_FIELDS}, change_pct=original.relative_change(cell, baseline),
                        method="mean paired ratios by repeat" if cell["cohort"] == "45028778" else "ratio of mean rates"))
    expected_comparisons = json.loads((SOURCE / "artifacts/comparisons.json").read_text())
    if compared != expected_comparisons:
        raise ValueError("Comparison results changed")
    output.mkdir(parents=True, exist_ok=False)
    original.write_csv(output / "summary.csv", [{k:v for k,v in c.items() if k != "runs"} for c in cells])
    original.write_tables(cells, output)
    (output / "comparisons.json").write_text(json.dumps(compared, indent=2) + "\n")
    if render:
        original.render(cells, output)
    record = dict(status="pass", rows=len(rows), cells=len(cells), render=render,
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (SOURCE / "build_report.py", SOURCE / "artifacts/runs.csv")},
                  scope="Derivative replay of normalized archived rows; not original raw-event audit; images may differ by plotting-library version")
    (output / "replay.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory; never overwrites archived figures")
    parser.add_argument("--render", action="store_true", help="Also generate the four heatmap pages and SVG/PNG figures")
    args = parser.parse_args()
    main(args.output.resolve(), args.render, args.repository)
