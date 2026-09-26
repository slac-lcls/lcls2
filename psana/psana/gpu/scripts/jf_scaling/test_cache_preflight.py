"""Reject incomplete frozen cache helpers before staging the full dataset."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).resolve().parent.parent
    monkeypatch.syspath_prepend(str(scripts/'feespec_bulk_benchmark'))
    monkeypatch.syspath_prepend(str(scripts/'jf_scaling'))
    spec = importlib.util.spec_from_file_location('jf_scaling_runner', scripts/'jf_scaling/run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('missing', ['common.py', 'warm_cache.py', 'memory_state.py', None])
def test_frozen_manifest_requires_cache_dependencies(tmp_path, runner, missing):
    helpers = tmp_path/'scripts/feespec_bulk_benchmark'
    helpers.mkdir(parents=True)
    hashes = {}
    for name in ('common.py', 'warm_cache.py', 'memory_state.py'):
        if name == missing:
            continue
        path = helpers/name
        path.write_text('# frozen helper\n')
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'hashes.json').write_text(json.dumps(hashes))
    if missing:
        with pytest.raises(RuntimeError, match=missing):
            runner.verify(tmp_path)
    else:
        runner.verify(tmp_path)
