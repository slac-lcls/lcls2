"""Validate report grouping, metrics, missing cells and comparison boundaries."""
import math

import pytest

from build_report import load_runs, make_panels, relative_change, select, summarize, validate_row


@pytest.fixture(scope='module')
def cells():
    return summarize(load_runs())


def test_inventory_and_phase_exclusions(cells):
    assert len(cells) == 48
    assert sum(cell['n'] for cell in cells) == 102
    assert all(row['phase']=='full' for cell in cells for row in cell['runs'])


def test_rate_identity():
    row = load_runs()[0]
    row['gib_s'] *= 2
    with pytest.raises(AssertionError):
        validate_row(row)


def test_cohorts_are_not_pooled(cells):
    original = select(cells,'44309691',bds=4,batch=200)
    extended = select(cells,'44321906',bds=4,batch=200)
    assert original['mean_hz'] != extended['mean_hz']
    with pytest.raises(AssertionError):
        relative_change(original,extended)


def test_missing_worker_cells_are_missing(cells):
    assert select(cells,'44770441',bds=4,workers=16,batch=200) is None
    assert math.isnan(relative_change(None,None))


def test_storage_uses_paired_ratios(cells):
    control = select(cells,'45028778',media='weka',gpus=1)
    measured = select(cells,'45028778',media='weka',gpus=4)
    change = relative_change(measured,control)
    assert change == pytest.approx(1.15384, abs=1e-5)
    assert abs(change - 100*(measured['mean_hz']/control['mean_hz']-1)) > .5


def test_worker_and_memory_metrics(cells):
    chosen = select(cells,'44770441',bds=4,workers=4,batch=200)
    assert chosen['bound_physical_cores']==34
    assert chosen['events_per_bound_core']==pytest.approx(chosen['mean_hz']/34)
    large = select(cells,'44870463',bds=4,bounce_mib=64)
    assert large['bounce_estimate_mib']==1024


def test_all_cells_are_plotted(cells):
    seen = set()
    for panel in make_panels(cells):
        for row_index, row in enumerate(panel['grid']):
            for cell in row:
                if cell is not None:
                    seen.add(id(cell))
                    assert panel['baselines'][row_index] is not None
    assert len(seen)==48
