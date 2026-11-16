"""Offline pytest-discovery and package-selection checks for this import."""
import ast
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

from hatchling.builders.wheel import WheelBuilder


def run(argv, cwd, env=None, expected=(0,)):
    p = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True)
    if p.returncode not in expected:
        raise RuntimeError(f'{argv}: exit {p.returncode}\n{p.stdout}\n{p.stderr}')
    return p


def main():
    repo = Path(sys.argv[1]).resolve()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    patterns = ['gpu/notes', 'gpu/scripts/pr187_pr208', 'gpu/scripts/pr299_perlmutter']
    frozen = [f'psana/psana/{p}' for p in patterns]
    config = tomllib.loads((repo / 'pyproject.toml').read_text())
    assert config['tool']['ruff']['force-exclude']
    assert set(frozen) <= set(config['tool']['ruff']['extend-exclude'])
    hooks = (repo / '.pre-commit-config.yaml').read_text()
    for name in ['trailing-whitespace', 'end-of-file-fixer', 'mixed-line-ending', 'check-ast', 'debug-statements']:
        pattern = re.search(r'      - id: '+name+r'\n        exclude: ([^\n]+)', hooks).group(1)
        assert all(re.match(pattern, p+'/test_snapshot.py') for p in frozen)
        assert not re.match(pattern, 'psana/psana/gpu/scripts/jf_scaling/test_contract.py')

    with tempfile.TemporaryDirectory(prefix='pr-import-isolation-') as directory:
        root = Path(directory)
        (root / 'conftest.py').write_bytes((repo / 'conftest.py').read_bytes())
        for path in frozen:
            p = root / path / 'test_snapshot.py'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('raise RuntimeError("Frozen snapshot was collected")\n')
        maintained = root / 'psana/psana/gpu/scripts/jf_scaling/test_maintained.py'
        maintained.parent.mkdir(parents=True, exist_ok=True)
        maintained.write_text('def test_maintained():\n    assert True\n')
        probe = run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'psana/psana/gpu'], root, env)
        assert '1 passed' in probe.stdout

        # Traverse the actual GPU tree using pytest's discovery and ignore rules,
        # replacing module imports with empty collectors to avoid native/GPU code.
        plugin = root / 'discovery_probe.py'
        plugin.write_text('''import json, os
from pathlib import Path
import pytest
seen=[]
class Probe(pytest.File):
    def collect(self):
        return []
def pytest_pycollect_makemodule(module_path, parent):
    seen.append(str(module_path))
    return Probe.from_parent(parent, path=module_path)
def pytest_sessionfinish(session, exitstatus):
    Path(os.environ['PR_DISCOVERY_OUTPUT']).write_text(json.dumps(seen))
''')
        output = root / 'discovered.json'
        probe_env = dict(env, PYTHONPATH=str(root), PR_DISCOVERY_OUTPUT=str(output))
        run([sys.executable, '-B', '-m', 'pytest', '--collect-only', '-q', '-p', 'no:cacheprovider',
             '-p', 'discovery_probe', 'psana/psana/gpu'], repo, probe_env, expected=(5,))
        seen = [Path(p).relative_to(repo).as_posix() for p in json.loads(output.read_text())]
        assert len(seen) > 0
        assert not any(p.startswith(tuple(frozen)) for p in seen)
        assert any('/jf_scaling/' in p for p in seen)

        meson = (repo / 'psana/meson.build').read_text()
        block = re.search(r"install_subdir\(\s*'psana',.*?exclude_directories:\s*(\[.*?\])", meson, re.S).group(1)
        exclusions = ast.literal_eval(block)
        assert set(patterns) <= set(exclusions)
        project = root / 'meson-project'; project.mkdir()
        relative_files = [p+'/archive.py' for p in patterns] + [
            'gpu/docs/design.md', 'gpu/examples/input_only.py', 'gpu/scripts/jf_scaling/bench.py', 'gpu/gpu_events.py']
        for rel in relative_files:
            f = project / 'psana' / rel; f.parent.mkdir(parents=True, exist_ok=True); f.write_text('fixture\n')
        (project / 'meson.build').write_text("project('pr-import-selection')\ninstall_subdir('psana', install_dir: 'share', exclude_directories: "+repr(exclusions)+")\n")
        build = root / 'meson-build'; dest = root / 'meson-dest'
        run([sys.executable, '-m', 'mesonbuild.mesonmain', 'setup', str(build), str(project), '--prefix=/usr'], root, env)
        run([sys.executable, '-m', 'mesonbuild.mesonmain', 'install', '-C', str(build), '--destdir', str(dest)], root, env)
        installed = sorted(str(p.relative_to(dest / 'usr/share/psana')) for p in (dest / 'usr/share/psana').rglob('*') if p.is_file())
        expected = sorted(relative_files[3:])
        assert installed == expected, installed

        # Test the real Hatch configuration, both source selection and the
        # container's packages=[]/staged-install route, without building natives.
        wheelroot = root / 'wheel-project'; wheelroot.mkdir()
        paths = []
        for prefix in ['psana/psana', 'install/lib/python3.11/site-packages/psana']:
            for rel in relative_files:
                path = prefix+'/'+rel; paths.append(path)
                f = wheelroot / path; f.parent.mkdir(parents=True, exist_ok=True); f.write_text('fixture\n')
        for package_setting in (['psana', 'psalg'], []):
            cfg = copy.deepcopy(config)
            cfg['tool']['hatch']['build']['targets']['wheel']['packages'] = package_setting
            builder = WheelBuilder(str(wheelroot), config=cfg)
            for path in paths:
                is_frozen = any('/'+p+'/' in '/'+path for p in patterns)
                assert builder.config.include_path(path, explicit=True) != is_frozen, path
            selected = [f.relative_path for f in builder.recurse_selected_project_files()]
            assert not any(any('/'+p+'/' in '/'+f for p in patterns) for f in selected)
            wanted_prefix = 'psana/psana/' if package_setting else 'install/lib/python3.11/site-packages/psana/'
            assert all(wanted_prefix+p in selected for p in expected), selected

    print(json.dumps(dict(pytest_fixture='1 maintained test passed; all frozen sentinel imports excluded',
                         actual_gpu_discovered_test_files=seen,
                         actual_discovery_note='pytest traversal with import-free probe collectors; no maintained GPU tests executed',
                         meson_fixture_installed=installed,
                         hatch_selection='source and container-staged install routes passed',
                         rewrite_lint_configuration='Ruff TOML and five pre-commit exclusions checked; tools not installed here',
                         limitations=['No full native build or production wheel was built.',
                                      'Ruff/pre-commit binaries unavailable; configuration checks only.']), indent=2))


if __name__ == '__main__':
    main()
