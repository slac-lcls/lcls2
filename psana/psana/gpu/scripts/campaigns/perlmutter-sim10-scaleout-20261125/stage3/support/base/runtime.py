"""Use the exact Perlmutter PR298 install, with no Python/native mixing."""
import os
from support import PREFIX, SOURCE, PR_SHA, require, runtime_identity

def activate():
    import psana
    from pathlib import Path
    require(Path(psana.__file__).resolve().is_relative_to(PREFIX), 'Wrong native install')
    require(Path(os.environ['SIM10_PR_SOURCE']).resolve() == SOURCE, 'Wrong PR source')
    runtime_identity(('psana', 'psana.dgram', 'psana.eventbuilder', 'psana.smdreader'))

identity = runtime_identity
