"""Consolidate existing Sim10 measurements and render cohort-safe heatmaps."""
import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np


HERE = Path(__file__).resolve().parent
NOTES = HERE.parent
BYTES_PER_EVENT = 3358200
CONFIG_FIELDS = ('cohort', 'media', 'gpus', 'bds', 'workers', 'streams',
                 'batch', 'task_mib', 'bounce_mib')
COHORTS = {
    '44232284': ('BD and historical streams', 'sdfampere023', 16, 8, 'reviewed'),
    '44309691': ('Three-point batch', 'sdfampere036', 12, 6, 'reviewed'),
    '44321906': ('Five-point batch', 'sdfampere040', 20, 10, 'reviewed'),
    '44770441': ('Workers and CPU provisioning', 'sdfampere036', 20, 10, 'reviewed'),
    '44852752': ('Task size', 'sdfampere017', 8, 4, 'result review pending'),
    '44870463': ('Bounce buffers', 'sdfampere018', 8, 4, 'result review pending'),
    '45028778': ('Storage and GPU sharing', 'sdfampere036', 18, 6, 'result review pending'),
}
CSV_SOURCES = (
    NOTES / 'sim10_single_node_scaling_20261113/runs.csv',
    NOTES / 'sim10_kvikio_tasksize_20261113/results/runs.csv',
    NOTES / 'sim10_kvikio_bounce_20261113/results/runs.csv',
    NOTES / 'sim10_storage_gpu_review_20261114/results/runs.csv',
)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def normalize(source, position):
    storage = 'payload_gib_s' in source
    row = dict(
        cohort=source.get('cohort', '45028778'),
        case=source['case'], phase=source.get('phase', source.get('kind')),
        sequence=int(source.get('sequence', source.get('order', position))),
        repeat=int(source.get('repeat', 0)),
        media=source.get('media', 'weka'), gpus=int(source.get('gpus', 1)),
        bds=int(source['bds']), workers=int(source.get('workers_per_bd', 4)),
        streams=int(source.get('streams_per_bd', 1)), batch=int(source.get('batch', 200)),
        task_mib=float(source.get('task_mib', int(source.get('task_bytes', 1048576))/2**20)),
        bounce_mib=float(source.get('bounce_mib', int(source.get('bounce_bytes', 16777216))/2**20)),
        events=int(source['events']), seconds=float(source['seconds']),
        hz=float(source['events_per_second' if storage else 'hz']),
        gib_s=float(source['payload_gib_s' if storage else 'gib_s']),
        payload_bytes=int(source.get('payload_bytes', int(source['events'])*BYTES_PER_EVENT)),
        nic_rx_payload_ratio=float(source['idle_adjusted_nic_rx_payload_ratio' if storage else 'nic_rx_payload_ratio']),
        disk_payload_ratio=float(source['idle_adjusted_disk_read_payload_ratio']) if storage else '',
        raw_path=source['source_file' if storage else 'raw_path'],
        raw_sha256=source['source_sha256' if storage else 'raw_sha256'],
    )
    row['host'] = COHORTS[row['cohort']][1]
    row['bound_physical_cores'] = int(source.get('physical_cores_used', 6 + row['bds']*(row['workers']+3)))
    row['total_workers'] = row['bds']*row['workers']
    row['budget_gib_per_bd'] = float(source.get('budget_gib_per_bd', 6))
    row['bulk_bytes'] = int(source.get('bulk_bytes', 524288))
    validate_row(row)
    return row


def validate_row(row):
    assert row['phase'] in ('pilot', 'full', 'anchor')
    assert row['seconds'] > 0
    assert row['payload_bytes'] == row['events']*BYTES_PER_EVENT
    assert math.isclose(row['hz'], row['events']/row['seconds'], rel_tol=1e-10)
    assert math.isclose(row['gib_s'], row['payload_bytes']/2**30/row['seconds'], rel_tol=1e-10)
    assert row['bound_physical_cores'] == 6 + row['bds']*(row['workers']+3)
    assert row['budget_gib_per_bd'] == 6 and row['bulk_bytes'] == 524288
    if row['phase'] != 'pilot':
        assert row['events'] == 100000


def load_runs():
    rows = []
    for path in CSV_SOURCES:
        rows.extend(normalize(source, position) for position, source in enumerate(read_csv(path)))
    assert len({(row['cohort'], row['case']) for row in rows}) == len(rows) == 156
    assert Counter(row['phase'] for row in rows) == {'full': 102, 'pilot': 48, 'anchor': 6}
    for cohort, (_, _, full, pilots, _) in COHORTS.items():
        selected = [row for row in rows if row['cohort'] == cohort]
        assert sum(row['phase'] == 'full' for row in selected) == full
        assert sum(row['phase'] == 'pilot' for row in selected) == pilots
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['phase'] == 'full':
            groups[tuple(row[field] for field in CONFIG_FIELDS)].append(row)
    cells = []
    for key, runs in sorted(groups.items()):
        cell = dict(zip(CONFIG_FIELDS, key))
        rates = [row['hz'] for row in runs]
        assert len(rates) == (3 if cell['cohort'] == '45028778' else 2)
        cell.update(host=runs[0]['host'], n=len(rates), mean_hz=statistics.mean(rates),
                    min_hz=min(rates), max_hz=max(rates),
                    mean_gib_s=statistics.mean(row['gib_s'] for row in runs),
                    stdev_hz=statistics.stdev(rates),
                    spread_pct=100*(max(rates)-min(rates))/statistics.mean(rates),
                    bound_physical_cores=runs[0]['bound_physical_cores'],
                    events_per_bound_core=statistics.mean(rates)/runs[0]['bound_physical_cores'],
                    gib_s_per_active_gpu=statistics.mean(row['gib_s'] for row in runs)/cell['gpus'],
                    bounce_estimate_mib=cell['bds']*cell['workers']*cell['bounce_mib'],
                    result_review=COHORTS[cell['cohort']][4], runs=runs)
        cells.append(cell)
    assert len(cells) == 48
    return cells


def select(cells, cohort, **filters):
    matches = [cell for cell in cells if cell['cohort'] == cohort
               and all(cell[key] == value for key, value in filters.items())]
    assert len(matches) <= 1, (cohort, filters)
    return matches[0] if matches else None


def relative_change(cell, baseline):
    if cell is None:
        return float('nan')
    assert baseline is not None and cell['cohort'] == baseline['cohort']
    if cell['cohort'] == '45028778':
        controls = {row['repeat']: row['hz'] for row in baseline['runs']}
        assert len(controls) == 3
        return 100*(statistics.mean(row['hz']/controls[row['repeat']] for row in cell['runs'])-1)
    return 100*(cell['mean_hz']/baseline['mean_hz']-1)


def panel(cells, title, cohort, row_labels, column_labels, filters, controls, xlabel, ylabel, control_label):
    grid = [[select(cells, cohort, **item) for item in row] for row in filters]
    baselines = [select(cells, cohort, **item) for item in controls]
    return dict(title=title, cohort=cohort, rows=row_labels, columns=column_labels,
                grid=grid, baselines=baselines, xlabel=xlabel, ylabel=ylabel,
                control_label=control_label)


def make_panels(cells):
    panels = []
    panels.append(panel(cells, 'BD count × CUDA streams', '44232284', [str(count) for count in (1, 2, 3, 4)],
                        ['1', '2'], [[dict(bds=count, streams=streams) for streams in (1, 2)] for count in (1, 2, 3, 4)],
                        [dict(bds=count, streams=1) for count in (1, 2, 3, 4)],
                        'Streams / BD (2 = historical)', 'BDs', 'same BD count, 1 stream'))
    for cohort, title, batches in [('44309691', 'Original batch screen', [100, 200, 500]),
                                   ('44321906', 'Extended batch screen', [10, 50, 100, 200, 500])]:
        panels.append(panel(cells, title, cohort, ['1', '4'], [str(batch) for batch in batches],
                            [[dict(bds=count, batch=batch) for batch in batches] for count in (1, 4)],
                            [dict(bds=count, batch=100) for count in (1, 4)], 'Batch events', 'BDs', 'same BD count, batch 100'))
    for count in (1, 4):
        panels.append(panel(cells, f'Workers × batch · {count} BD(s)', '44770441', ['50', '100', '200'], ['4', '8', '16'],
                            [[dict(bds=count, workers=workers, batch=batch) for workers in (4, 8, 16)] for batch in (50, 100, 200)],
                            [dict(bds=count, workers=8, batch=batch) for batch in (50, 100, 200)],
                            'Workers / BD', 'Batch', 'same batch, 8 workers / BD'))
    panels.append(panel(cells, 'Fixed 16 total workers', '44770441', ['batch 100'], ['1 × 16', '2 × 8', '4 × 4'],
                        [[dict(bds=count, workers=workers, batch=100) for count, workers in [(1, 16), (2, 8), (4, 4)]]],
                        [dict(bds=1, workers=16, batch=100)], 'BDs × workers / BD', '', '1 BD × 16 workers; cores vary'))
    for cohort, title, parameter, values in [('44852752', 'KvikIO task size', 'task_mib', [1, 4]),
                                            ('44870463', 'KvikIO bounce size', 'bounce_mib', [16, 64])]:
        base_filters = [dict(bds=1, workers=8, batch=50), dict(bds=4, workers=4, batch=200)]
        panels.append(panel(cells, title, cohort, ['1 BD / 8w / b50', '4 BD / 4w / b200'], [str(value) for value in values],
                            [[dict(**profile, **{parameter:value}) for value in values] for profile in base_filters],
                            [dict(**profile, **{parameter:values[0]}) for profile in base_filters], 'MiB', '', f'same row, {values[0]} MiB'))
    panels.append(panel(cells, 'Storage × GPUs · 4 total BDs', '45028778', ['Weka', 'NVMe'], ['1', '2', '4'],
                        [[dict(media=media, gpus=gpus) for gpus in (1, 2, 4)] for media in ('weka', 'nvme')],
                        [dict(media=media, gpus=1) for media in ('weka', 'nvme')],
                        'Active GPUs (4 / 2 / 1 BDs per GPU)', '', 'same medium + repeat, 1 GPU'))
    return panels


def metric_value(cell, baseline, metric):
    if cell is None:
        return float('nan')
    return relative_change(cell, baseline) if metric == 'change_pct' else cell[metric]


def draw_panel(axis, spec, metric, cmap, norm):
    matrix = np.array([[metric_value(cell, spec['baselines'][row_index], metric)
                        for cell in row] for row_index, row in enumerate(spec['grid'])])
    image = axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect='auto')
    for row_index, row in enumerate(spec['grid']):
        for column_index, cell in enumerate(row):
            value = matrix[row_index, column_index]
            if cell is None:
                label, color = 'not run', '#55616b'
            else:
                if metric == 'mean_gib_s':
                    label = f'{value:.2f}\n{cell["mean_hz"]:,.0f}/s'
                elif metric == 'change_pct':
                    label = f'{value:+.1f}%'
                elif metric == 'spread_pct':
                    label = f'{value:.1f}%'
                else:
                    label = f'{value:.2f}' if metric == 'gib_s_per_active_gpu' else f'{value:.1f}'
                red, green, blue, _ = cmap(norm(value))
                color = 'white' if .2126*red + .7152*green + .0722*blue < .48 else '#10212d'
            axis.text(column_index, row_index, label, ha='center', va='center', fontsize=9, color=color)
    axis.set_xticks(range(len(spec['columns'])), spec['columns'])
    axis.set_yticks(range(len(spec['rows'])), spec['rows'])
    axis.set_xlabel(spec['xlabel'], fontsize=9)
    axis.set_ylabel(spec['ylabel'], fontsize=9)
    subtitle = f'{spec["cohort"]} · {COHORTS[spec["cohort"]][1]} · n={3 if spec["cohort"] == "45028778" else 2}'
    if metric == 'change_pct':
        subtitle += '\nControl: ' + spec['control_label']
    axis.set_title(spec['title']+'\n'+subtitle, fontsize=10, loc='left', pad=9)
    axis.tick_params(axis='both', length=0)
    axis.set_xticks(np.arange(-.5, len(spec['columns']), 1), minor=True)
    axis.set_yticks(np.arange(-.5, len(spec['rows']), 1), minor=True)
    axis.grid(which='minor', color='white', linewidth=2)
    axis.tick_params(which='minor', bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def save_figure(figure, output, name, pdf):
    for extension in ('png', 'svg'):
        figure.savefig(output/(name+'.'+extension), dpi=160, bbox_inches='tight')
    pdf.savefig(figure, bbox_inches='tight')
    plt.close(figure)


def render(cells, output):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'svg.fonttype':'none',
                         'figure.facecolor':'#fbfcfe', 'axes.facecolor':'#eeeeee'})
    panels = make_panels(cells)
    specs = [
        ('mean_gib_s', 'throughput', 'Payload throughput', 'Mean GiB/s; second line = global events/s', 'viridis', Normalize(0, 5.5)),
        ('change_pct', 'relative_change', 'Change versus matched controls', 'Percent change; descriptive, not statistical significance', 'RdBu', TwoSlopeNorm(vmin=-30, vcenter=0, vmax=160)),
        ('spread_pct', 'repeat_spread', 'Observed repeat spread', '100 × (maximum − minimum) / mean; not a confidence interval', 'magma_r', Normalize(0, 20)),
    ]
    with PdfPages(output/'heatmaps.pdf') as pdf:
        for metric, name, title, legend, color_map, normalization in specs:
            figure, axes = plt.subplots(3, 3, figsize=(18, 13.5), layout='constrained')
            cmap = plt.get_cmap(color_map).copy()
            cmap.set_bad('#e9edf2')
            for axis, spec in zip(axes.flat, panels):
                image = draw_panel(axis, spec, metric, cmap, normalization)
            figure.colorbar(image, ax=list(axes.flat), label=legend, fraction=.024, pad=.02)
            figure.suptitle(f'Sim10 single-node staging · {title}\n7 allocation cohorts · 48 measured configurations · 102 full runs', fontsize=17)
            figure.supxlabel('Separate jobs are separate experiments: never use another panel/job as a speedup denominator.\nGray = unmeasured, not zero. All data exclude pilots and anchors. No expansion/calibration/cube kernels. Three newest cohorts await independent result review.', fontsize=10)
            save_figure(figure, output, name, pdf)
        figure, axes = plt.subplots(2, 2, figsize=(14, 9.5), layout='constrained')
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('#e9edf2')
        resource_specs = [(panels[3], 'events_per_bound_core', Normalize(0,55), 'Events/s per bound physical core'),
                          (panels[4], 'events_per_bound_core', Normalize(0,55), 'Events/s per bound physical core'),
                          (panels[7], 'bounce_estimate_mib', Normalize(0,1024), 'Estimated worker bounce demand, MiB'),
                          (panels[8], 'gib_s_per_active_gpu', Normalize(0,5.5), 'Payload GiB/s per active GPU')]
        for axis, (spec, metric, normalization, legend) in zip(axes.flat, resource_specs):
            image = draw_panel(axis, spec, metric, cmap, normalization)
            figure.colorbar(image, ax=axis, label=legend, shrink=.8)
        figure.suptitle('Resource-normalized rates and memory estimates\nProvisioning indicators, not measured utilization, power or billed cost', fontsize=16)
        figure.supxlabel('Bound cores = 6 + BDs × (workers + 3); allocation rounding/exclusive resources can be larger.\nBounce demand assumes one buffer per worker, not measured pinned allocation. GPU campaign allocated four GPUs in every cell; active GPUs vary.', fontsize=10)
        save_figure(figure, output, 'resources', pdf)


def write_csv(path, rows):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(output.getvalue())


def write_tables(cells, output):
    lines = ['# All measured full-run configurations', '',
             'Each cohort stays separate. Means and observed ranges use only full cases; ranges are not confidence intervals.', '']
    for cohort, (name, host, _, _, review) in COHORTS.items():
        lines += [f'## {name} — {cohort} — {host}', '', f'Result review: {review}.', '',
                  '| Media | GPUs | BDs | Workers/BD | Streams | Batch | Task MiB | Bounce MiB | n | Events/s mean [min, max] | GiB/s | Bound cores |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
        for cell in cells:
            if cell['cohort'] == cohort:
                values = [cell[field] for field in ('media','gpus','bds','workers','streams','batch','task_mib','bounce_mib','n')]
                values += [f'{cell["mean_hz"]:.2f} [{cell["min_hz"]:.2f}, {cell["max_hz"]:.2f}]', f'{cell["mean_gib_s"]:.4f}', cell['bound_physical_cores']]
                lines.append('| '+' | '.join(map(str, values))+' |')
        lines.append('')
    (output/'tables.md').write_text('\n'.join(lines)+'\n')


def main(output, archive):
    output.mkdir(parents=True, exist_ok=False)
    rows = load_runs()
    cells = summarize(rows)
    write_csv(output/'runs.csv', rows)
    write_csv(output/'summary.csv', [{key:value for key,value in cell.items() if key!='runs'} for cell in cells])
    write_tables(cells, output)
    render(cells, output)
    panels = make_panels(cells)
    compared = []
    plotted = set()
    for spec in panels:
        for row_index, grid_row in enumerate(spec['grid']):
            for cell in grid_row:
                if cell is None:
                    continue
                plotted.add(tuple(cell[key] for key in CONFIG_FIELDS))
                baseline = spec['baselines'][row_index]
                compared.append(dict(panel=spec['title'], configuration={key:cell[key] for key in CONFIG_FIELDS},
                                     control={key:baseline[key] for key in CONFIG_FIELDS},
                                     change_pct=relative_change(cell,baseline),
                                     method='mean paired ratios by repeat' if cell['cohort']=='45028778' else 'ratio of mean rates'))
    assert len(plotted) == 48
    (output/'comparisons.json').write_text(json.dumps(compared,indent=2)+'\n')
    evidence = list(CSV_SOURCES) + [
        NOTES/'sim10_single_node_scaling_20261113/cohorts.json',
        NOTES/'sim10_single_node_scaling_20261113/validation.json',
        NOTES/'sim10_single_node_scaling_20261113/review/round-01.json',
        NOTES/'sim10_kvikio_tasksize_20261113/RESULTS.md',
        NOTES/'sim10_kvikio_tasksize_20261113/results/validation.json',
        NOTES/'sim10_kvikio_bounce_20261113/RESULTS.md',
        NOTES/'sim10_kvikio_bounce_20261113/results/validation.json',
        NOTES/'sim10_storage_gpu_review_20261114/RESULTS.md',
        NOTES/'sim10_storage_gpu_review_20261114/results/summary.json',
        NOTES/'mfx_expand10_512k_20261108/README.md',
        NOTES/'mfx_expand10_capacity_20261106/README.md',
        NOTES/'mfx_expand10_baseline10k_20261108/README.md',
        NOTES/'mfx_expand10_single_node_baseline_20261108.md',
    ]
    provenance = dict(recorded_at=datetime.now(timezone.utc).isoformat(),
                      source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in evidence},
                      raw_case_hash_policy='Retained from source CSVs, not independently rehashed during this report build',
                      counts=dict(cohorts=7, configurations=48, full=102, pilots=48, anchors=6),
                      workload='100k global-event staging, pinned PR298, compatibility ON, no application kernels',
                      report_review='Not independently reviewed; first four cohorts have prior result review',
                      figures=[dict(name=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
                               for path in sorted(output.glob('*')) if path.suffix in ('.svg','.png','.pdf')])
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if archive:
        target = HERE/'artifacts'
        target.mkdir(exist_ok=False)
        for path in sorted(output.iterdir()):
            if path.suffix != '.png':
                shutil.copyfile(path,target/path.name)
    print(json.dumps(dict(output=str(output), archived=archive, counts=provenance['counts']),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--archive',action='store_true')
    args = parser.parse_args()
    main(args.output.resolve(),args.archive)
