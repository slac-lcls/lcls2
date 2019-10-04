# This test checks that psana doesn't import mpi4py, which obviously
# is going to break in a non-MPI environment.

# This should not assert.
from psana import DataSource

# This should assert.
try:
    import mpi4py
except AssertionError:
    pass # ok, failure expected
else:
    raise Exception('test passed, but was expecting a failure')
