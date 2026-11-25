"""One historical repeat, with both acceptance checks, on one allocation."""
import json
from pathlib import Path
from historical_plan import cases as historical_cases, placement, OUTPUT_RESERVE
from support import require

REPEAT = json.loads(Path(__file__).with_name('selection.json').read_text())['repeat']
require(type(REPEAT) is int and REPEAT in (1, 2, 3), 'Reviewed repeat only')
ATTEMPT = 'grouped-replay-20261123-r%d' % REPEAT
WALLTIME = 9180
TIME_LIMIT = '02:33:00'
BUDGET = dict(preflight=300, device_checks=480, acceptance=600,
              staging=1200, compute=2880, final_audit=700, summary=120,
              safety=560, cache_conditioning=13*180)


def cases():
    return [c for c in historical_cases() if c['kind'] == 'acceptance' or c['repeat'] == REPEAT]


def validate():
    matrix = cases()
    require(sum(BUDGET.values()) == WALLTIME, 'Whole-allocation budget')
    require(len(matrix) == len({c['name'] for c in matrix}) == 13, 'Unique bounded group')
    require(sum(c['timeout'] for c in matrix) == sum(BUDGET[k] for k in
            ('acceptance', 'staging', 'compute', 'cache_conditioning')), 'Case budget')
    require(len([c for c in matrix if c['cohort'] == 'comparison']) == 8, 'Both profiles and all stages')
    require(all(c['events'] == 100000 for c in matrix if c['kind'] != 'acceptance'), 'Matched full event count')

validate()
