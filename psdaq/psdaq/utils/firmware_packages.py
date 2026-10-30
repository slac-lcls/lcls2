"""
Python library paths of the firmware packages checked out under $SUBMODULEDIR.

Every firmware package (lcls2-pgp-pcie-apps, epix-100a-gen2, ...) ships its own
copy of the shared rogue libraries (surf, axi-pcie-core, l2si-core, ...) at the
version its firmware was built and tested with.  ``pyrogue.addLibraryPath`` only
prepends a directory to ``sys.path``, and the first ``import surf`` binds the
name ``surf`` to one directory for the life of the interpreter.  A process that
uses two firmware packages therefore runs one of them against the other's copy
of surf, and which one wins depends on import order.  That works only as long as
the two versions happen to be compatible.

This module provides two ways to make a package importable:

``enable(name)``
    Process-wide: prepend the package's library paths to ``sys.path``.  This is
    what the ``psdaq.utils.enable_*`` modules do, and it is fine for a process
    that uses a single firmware package.

``isolated(name)``
    Context manager for a process that uses more than one package.  Imports done
    inside the block see only this package's library paths, and the modules they
    create belong to this package: on exit they are taken out of ``sys.modules``
    (and ``sys.path`` is restored) so that another package can be imported with
    its own copies of the shared libraries.  Objects created from these modules
    keep working outside the block, and re-entering the block puts the modules
    back for further imports::

        with isolated('epix-100a-gen2'):
            import ePixFpga as fpga
        with isolated('lcls2-epix-hr-pcie'):
            import lcls2_epix_hr_pcie

    Isolation covers imports executed inside the block, which includes the
    module-level imports of everything imported there.  An ``import`` executed
    later at run time (inside a function) is not covered and must also run
    inside an ``isolated`` block.  A package should be used through either
    ``enable`` or ``isolated`` within one process, not both.  Neither function
    is thread-safe; call them while importing, before any other python threads
    touch the import system.
"""

import contextlib
import logging
import os
import sys

log = logging.getLogger(__name__)

# Library paths of each package, relative to $SUBMODULEDIR/<package>, in
# sys.path priority order (the first entry is searched first).
PACKAGES = {
    'cameralink-gateway': (
        'firmware/python',
        'firmware/submodules/surf/python',
        'firmware/submodules/clink-gateway-fw-lib/python',
        'firmware/submodules/l2si-core/python',
        'firmware/submodules/lcls-timing-core/python',
        'firmware/submodules/lcls2-pgp-fw-lib/python',
        'firmware/submodules/axi-pcie-core/python',
    ),
    'epix': (
        'software/Epix/python',
        'firmware/python',
        'firmware/submodules/surf/python',
    ),
    'epix-100a-gen2': (
        'software/python',
        'firmware/submodules/surf/python',
    ),
    'epix-hr-m-320k': (
        'firmware/submodules/AsicRegMapping/python',
        'firmware/submodules/axi-pcie-core/python',
        'firmware/python',
        'firmware/submodules/surf/python',
        'firmware/submodules/l2si-core/python',
        'firmware/submodules/lcls-timing-core/python',
        'firmware/submodules/epix-hr-core/python',
        'firmware/submodules/epix-hr-leap-common/python',
    ),
    'epix-hr-single-10k': (
        'firmware/submodules/ePixViewer/python',
        'firmware/submodules/lcls-timing-core/python',
        'firmware/submodules/l2si-core/python',
        'software/python',
        'firmware/python',
        'firmware/submodules/surf/python',
        'firmware/submodules/epix-hr-core/python',
        'firmware/submodules/axi-pcie-core/python',
    ),
    'epix-quad': (
        'software/python',
        'firmware/python',
        'firmware/submodules/surf/python',
        'firmware/submodules/axi-pcie-core/python',
    ),
    'epix-quad-1kfps': (
        'software/python',
        'firmware/python',
        'firmware/submodules/surf/python',
        'firmware/submodules/axi-pcie-core/python',
    ),
    'epix-uhr-gtreadout-dev': (
        'firmware/python',
        'firmware/submodules/ePixViewer/python',
        'firmware/submodules/surf/python',
        'firmware/submodules/l2si-core/python',
        'firmware/submodules/lcls-timing-core/python',
        'firmware/submodules/pixel-camera-readout-common/python',
        'firmware/submodules/axi-pcie-core/python',
        'firmware/submodules/AsicRegMapping/python',
    ),
    'epixuhr-3x2-readout-testing': (
        'firmware/python',
        'firmware/submodules/lcls2-pgp-fw-lib/python',
        'firmware/submodules/hdl-cores-lib-dev/firmware/python',
        'firmware/submodules/ePixViewer/python',
        'firmware/submodules/surf/python',
        'firmware/submodules/l2si-core/python',
        'firmware/submodules/lcls-timing-core/python',
        'firmware/submodules/pixel-camera-readout-common/python',
        'firmware/submodules/axi-pcie-core/python',
        'firmware/submodules/AsicRegMapping/python',
    ),
    'high-rate-encoder-dev': (
        'firmware/python',
        'firmware/submodules/surf/python',
        'firmware/submodules/l2si-core/python',
        'firmware/submodules/lcls-timing-core/python',
        'firmware/submodules/lcls2-pgp-fw-lib/python',
        'firmware/submodules/axi-pcie-core/python',
    ),
    'l2si-drp': (
        'firmware/python',
        'firmware/submodules/surf/python',
        'firmware/submodules/l2si-core/python',
        'firmware/submodules/lcls-timing-core/python',
        'firmware/submodules/lcls2-pgp-fw-lib/python',
        'firmware/submodules/axi-pcie-core/python',
    ),
    'lcls2-epix-hr-pcie': (
        'firmware/python',
        'firmware/submodules/surf/python',
        'firmware/submodules/l2si-core/python',
        'firmware/submodules/lcls-timing-core/python',
        'firmware/submodules/lcls2-pgp-fw-lib/python',
        'firmware/submodules/axi-pcie-core/python',
    ),
    'lcls2-pgp-pcie-apps': (
        'firmware/python',
        'firmware/submodules/surf/python',
        'firmware/submodules/l2si-core/python',
        'firmware/submodules/lcls-timing-core/python',
        'firmware/submodules/lcls2-pgp-fw-lib/python',
        'firmware/submodules/axi-pcie-core/python',
    ),
    'lcls2-timetool': (
        'firmware/common/python',
        'firmware/submodules/clink-gateway-fw-lib/python',
        'firmware/submodules/l2si-core/python',
        'firmware/submodules/lcls2-pgp-fw-lib/python',
        'firmware/submodules/lcls-timing-core/python',
        'firmware/submodules/axi-pcie-core/python',
        'firmware/submodules/surf/python',
    ),
    'lcls2-udp-pcie-apps': (
        'firmware/python',
        'firmware/submodules/surf/python',
        'firmware/submodules/l2si-core/python',
        'firmware/submodules/lcls-timing-core/python',
        'firmware/submodules/lcls2-pgp-fw-lib/python',
        'firmware/submodules/axi-pcie-core/python',
    ),
}

# (package, relative path) entries that are allowed to be missing from a checkout.
OPTIONAL_PATHS = {
    ('epix-hr-single-10k', 'firmware/submodules/ePixViewer/python'),
}


def library_paths(name):
    """Absolute library paths of a package, in sys.path priority order."""
    try:
        top = os.path.join(os.environ['SUBMODULEDIR'], name)
    except KeyError:
        raise RuntimeError(f'SUBMODULEDIR is not set, cannot locate firmware package {name!r}') from None
    paths = []
    for rel in PACKAGES[name]:
        path = os.path.join(top, rel)
        if os.path.isdir(path):
            paths.append(path)
        elif (name, rel) in OPTIONAL_PATHS:
            log.warning('firmware package %s: optional library path %s not found', name, path)
        else:
            raise RuntimeError(f'firmware package {name!r}: library path {path} does not exist')
    return paths


class _Package:
    def __init__(self, name):
        self.name = name
        self.paths = library_paths(name)
        # sys.modules entries owned by this package while it is not active
        self.modules = {}
        # a package may append realpath-based entries to sys.path at import
        prefixes = [os.path.join(p, '') for p in self.paths]
        prefixes += [os.path.join(os.path.realpath(p), '') for p in self.paths]
        self._prefixes = tuple(prefixes)

    def owned_modules(self):
        """Names of the sys.modules entries loaded from this package's paths."""
        return [modname for modname, mod in sys.modules.items()
                if (getattr(mod, '__file__', None) or '').startswith(self._prefixes)]


_packages = {}
_active = None


def _package(name):
    if name not in _packages:
        _packages[name] = _Package(name)
    return _packages[name]


def enable(name):
    """Prepend the library paths of a firmware package to sys.path (process-wide)."""
    pkg = _package(name)
    for path in reversed(pkg.paths):
        sys.path.insert(0, path)


@contextlib.contextmanager
def isolated(name):
    """Import a firmware package with its own copies of the shared rogue libraries."""
    global _active
    if _active is not None:
        raise RuntimeError(f'isolated({name!r}) cannot be nested inside isolated({_active!r})')
    pkg = _package(name)
    others = [p for p in _packages.values() if p is not pkg]
    saved_path = list(sys.path)
    # Hide the other packages: their live modules and their library paths
    parked = {}
    for other in others:
        for modname in other.owned_modules():
            parked[modname] = sys.modules.pop(modname)
    other_paths = {path for other in others for path in other.paths}
    sys.path[:] = [p for p in sys.path if p not in other_paths]
    # Expose this package
    for modname in pkg.modules:
        if modname in sys.modules:
            parked[modname] = sys.modules.pop(modname)
    sys.modules.update(pkg.modules)
    for path in reversed(pkg.paths):
        sys.path.insert(0, path)
    _active = name
    try:
        yield
    finally:
        _active = None
        pkg.modules = {modname: sys.modules.pop(modname) for modname in pkg.owned_modules()}
        sys.path[:] = saved_path
        sys.modules.update(parked)
        log.debug('firmware package %s: %d modules imported in isolation', name, len(pkg.modules))
