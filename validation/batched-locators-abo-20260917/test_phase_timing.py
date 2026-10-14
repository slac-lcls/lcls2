import ast
from pathlib import Path

import pytest

from phase_timing import Recorder, transform


def test_nested_exclusive_and_exception():
    ticks = iter((0, 10, 30, 50))
    recorder = Recorder('A', clock=lambda: next(ticks))
    recorder.active = True
    with pytest.raises(ValueError):
        with recorder.phase('outer'):
            with recorder.phase('inner'):
                raise ValueError('exercise finally')
    stats = recorder.snapshot()['phases']
    assert stats['startup/outer']['total_ns'] == 50
    assert stats['startup/outer']['self_ns'] == 30
    assert stats['startup/inner']['self_ns'] == 20


def test_disabled_and_nvtx_balance():
    calls = []
    class Nvtx:
        def RangePush(self, name):
            calls.append(name)
        def RangePop(self):
            calls.append('pop')
    recorder = Recorder('B', Nvtx())
    with recorder.phase('disabled'):
        pass
    assert not calls and not recorder.stats
    recorder.active = recorder.steady = True
    with recorder.phase('enabled'):
        pass
    assert calls == ['psana.B.enabled', 'pop']
    assert recorder.stats['steady/enabled']['calls'] == 1


@pytest.mark.parametrize('variant,kind', [('A', 'detector'), ('B', 'detector'), ('B', 'parse'), ('O', 'detector'), ('O', 'parse')])
def test_historical_transform_preserves_statements(variant, kind):
    root = (Path('/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline/validation/perf-acceptance-20260916/sources') / variant.lower() if variant in 'AB' else Path(__file__).resolve().parents[2]) / 'psana/psana/gpu'
    path = root / ('gpu_detector.py' if kind == 'detector' else 'gpudgram/batch.py')
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name ==
                 ('GPUDetector' if kind == 'detector' else 'GpuXtcBatchPool'))
    original = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name ==
                    ('process_batch' if kind == 'detector' else 'parse'))
    patched, matches = transform(ast.unparse(original), kind, variant)
    compile(patched, str(path), 'exec')
    class StripRanges(ast.NodeTransformer):
        def visit_With(self, node):
            node = self.generic_visit(node)
            expr = node.items[0].context_expr
            if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == 'phase':
                return node.body
            return node
    restored = StripRanges().visit(patched).body[0]
    assert ast.dump(restored) == ast.dump(original)
    assert matches


def test_changed_function_fails_closed():
    with pytest.raises(RuntimeError, match='unexpected instrumentation'):
        transform('def parse(self):\n    return 123\n', 'parse', 'B')
