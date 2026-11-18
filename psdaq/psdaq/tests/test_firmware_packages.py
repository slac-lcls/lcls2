"""
Tests for psdaq.utils.firmware_packages.

Two fake firmware packages are built in tmp_path.  Each carries its own copy
of a shared library ``surf`` (reporting a different VERSION) and a package
module that imports it, like the real packages under $SUBMODULEDIR.
"""
import sys
import textwrap

import pytest

from psdaq.utils import firmware_packages as fp

TOPS = {'pkg-a': 'pkga', 'pkg-b': 'pkgb'}
LIB_PATHS = ('firmware/python', 'firmware/submodules/surf/python')


@pytest.fixture
def submoduledir(tmp_path, monkeypatch):
    for pkg, top in TOPS.items():
        surf = tmp_path / pkg / 'firmware/submodules/surf/python/surf'
        surf.mkdir(parents=True)
        (surf / '__init__.py').write_text(f"VERSION = '{pkg}'\n")
        mod = tmp_path / pkg / 'firmware/python' / top
        mod.mkdir(parents=True)
        (mod / '__init__.py').write_text(textwrap.dedent("""\
            import surf
            SURF_VERSION = surf.VERSION
            """))
    monkeypatch.setenv('SUBMODULEDIR', str(tmp_path))
    monkeypatch.setattr(fp, 'PACKAGES', {pkg: LIB_PATHS for pkg in TOPS})
    monkeypatch.setattr(fp, '_packages', {})
    monkeypatch.setattr(fp, '_active', None)
    saved_path, saved_modules = list(sys.path), dict(sys.modules)
    yield tmp_path
    sys.path[:] = saved_path
    sys.modules.clear()
    sys.modules.update(saved_modules)


def _import(name):
    return __import__(name)


def test_isolated_gives_each_package_its_own_surf(submoduledir):
    with fp.isolated('pkg-a'):
        pkga = _import('pkga')
    with fp.isolated('pkg-b'):
        pkgb = _import('pkgb')
    assert pkga.SURF_VERSION == 'pkg-a'
    assert pkgb.SURF_VERSION == 'pkg-b'
    assert pkga.surf is not pkgb.surf


def test_isolated_leaves_no_trace_and_is_reenterable(submoduledir):
    with fp.isolated('pkg-a'):
        pkga = _import('pkga')
    assert 'surf' not in sys.modules
    assert 'pkga' not in sys.modules
    assert not any(str(submoduledir) in p for p in sys.path)
    with fp.isolated('pkg-a'):
        assert _import('surf') is pkga.surf
        assert _import('pkga') is pkga


def test_isolated_hides_an_enabled_package(submoduledir):
    fp.enable('pkg-a')
    pkga = _import('pkga')
    with fp.isolated('pkg-b'):
        assert _import('surf').VERSION == 'pkg-b'
        assert 'pkga' not in sys.modules
    # the enabled package is back once the block exits
    assert sys.modules['pkga'] is pkga
    assert sys.modules['surf'] is pkga.surf


def test_isolated_cannot_nest(submoduledir):
    with fp.isolated('pkg-a'):
        with pytest.raises(RuntimeError, match='nested'):
            with fp.isolated('pkg-b'):
                pass


def test_missing_submoduledir_and_path_are_reported(submoduledir, monkeypatch):
    monkeypatch.setattr(fp, 'PACKAGES', {'pkg-a': LIB_PATHS + ('software/python',)})
    with pytest.raises(RuntimeError, match='software/python'):
        fp.library_paths('pkg-a')
    monkeypatch.delenv('SUBMODULEDIR')
    with pytest.raises(RuntimeError, match='SUBMODULEDIR'):
        fp.library_paths('pkg-a')
