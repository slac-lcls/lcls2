"""Record the order of collective calls a communicator receives.

A test aid, not production code. The unit-test fakes drive peers sequentially,
so they have no notion of two ranks blocked in different calls -- which is
exactly the failure mode that produced every hang in the device-placement and
constant-sharing work. Wrapping a communicator and comparing the recorded
sequences across ranks catches a path that skipped or reordered a step.

Used two ways:

* in unit tests, to assert the sequence a code path produces
  (``test_gpu_collectives.py``, and ``make_peers(record=True)`` in
  ``test_gpu_shared_constants.py``);
* in ``mpi_failure_paths.py``, which builds its own placement and so can wrap
  ``placement.device_comm`` before handing it to the code under test, then
  calls ``check_agreement`` at a point every rank reaches. ``run_cases.py``
  fails a case on the ``collective order diverged`` line it produces.

A driver that goes through the production ``DataSource`` path -- such as
``mpi_calib_pixel_exact.py`` -- cannot wrap anything: discovery builds the
communicator inside the ``Run``, so there is no point at which the driver
holds it before use. Those drivers assert on results instead, which is the
stronger statement where it is available.

``SharedRequestedConstants._check_order`` calls ``check_agreement`` on its
communicator when the method is present, so wrapping is all a driver has to
do -- production holds no reference to this class and reads no environment
variable to decide whether to use it.

The recorder raises confidence rather than proving correctness: a sequential
fake still cannot model true concurrency, so the hardware failure cases
remain the only real proof.
"""


class RecordingComm:
    """Communicator wrapper that records the order of collective calls.

    Delegates everything to the wrapped communicator and appends a label per
    call. ``sequence_digest`` summarises the calls made so far; comparing that
    across ranks detects a path that skipped or reordered a step.
    """

    _COLLECTIVES = ('bcast', 'allgather', 'allreduce', 'Allreduce', 'Barrier',
                    'Ibarrier', 'Iallreduce', 'gather', 'Split', 'Free')

    def __init__(self, comm, label=''):
        self._comm = comm
        self._label = label
        self.calls = []

    def __getattr__(self, name):
        attribute = getattr(self._comm, name)
        if name not in self._COLLECTIVES or not callable(attribute):
            return attribute

        def recorded(*args, **kwargs):
            self.calls.append(name)
            return attribute(*args, **kwargs)

        return recorded

    def sequence_digest(self):
        """Stable summary of the calls made so far."""
        from hashlib import blake2b
        hasher = blake2b(digest_size=8)
        hasher.update('|'.join(self.calls).encode())
        return hasher.hexdigest()

    def check_agreement(self, step, logger=None):
        """Confirm every rank made the same calls in the same order.

        Called at a point all ranks reach. Uses the wrapped communicator
        directly so the check does not record itself.
        """
        digest = self.sequence_digest()
        everyone = self._comm.allgather((digest, len(self.calls)))
        if len(set(everyone)) == 1:
            return True
        message = (f'collective order diverged at {step!r}: '
                   f'{sorted(set(everyone))}. This rank made {self.calls}.')
        if logger is not None:
            logger.error(message)
        else:
            print(f'[PSANA-GPU-ERROR] {message}', flush=True)
        return False
