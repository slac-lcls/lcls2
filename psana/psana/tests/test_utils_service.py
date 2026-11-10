"""`first_service` rejects service 0 as unsupported input.

Service 0 is `TransitionId.ClearReadout` in the DAQ enum, but enum membership
does not make it a supported INPUT datagram on psana's normal
event-processing path. The standard `l2si-core` timing receiver consumes
message 0 by resetting its FIFO and suppressing the output record, so it does
not reach DRP software as a datagram:

  https://github.com/slaclab/l2si-core/blob/cc7b7f851ce229291f1c4f9f8f29b6b1bbb0c44b/base/rtl/TriggerEventBuffer.vhd#L356-L362

A present service-0 record is therefore unsupported input and must raise
rather than be silently accepted or skipped (issue #156).

An ABSENT GPU-side SMD record is a separate thing and is represented by
`None`, not by 0 -- see `_iter_step_events` in `psana/gpu/gpu_events.py`.

This is a shared CPU-path function -- `Event.service()`, `psexp/run.py`,
`psexp/node.py`, `psexp/step.py`, `psexp/mpi_ds.py`, `drp_ds.py`,
`shmem_ds.py` and `singlefile_ds.py` all reach it -- so the accepted range is
CPU behaviour, not a GPU-local detail. It had no tests at all before this
issue, which is why the range went unexamined for so long.
"""
from types import SimpleNamespace as NS

import pytest

from psana import utils
from psana.psexp import TransitionId


def dgram(service):
    """A stand-in carrying `service` where first_service reads it."""
    return NS(env=lambda: service << 24)


def test_every_real_record_service_id_is_accepted():
    """Reset (1) through L1Accept (12): every id that names a real record.

    Deliberately stops at 12 rather than covering the whole accepted range.
    `first_service` also accepts 13, which is `TransitionId.NumberOf` -- an
    enum bound, not a record that can arrive -- and that is left exactly as
    master has it, because the revised scope of this work keeps enum-sentinel
    validation separate (monarin, PR #241, 2026-11-01). Asserting 13 here
    would pin behaviour this change does not claim is right; asserting it is
    rejected would make the change this PR is not making. So 13 is
    untested on purpose, and `TransitionId.NumberOf` is named below so a
    reader can see the gap is chosen rather than overlooked.
    """
    assert TransitionId.NumberOf == 13, 'the untested upper bound'
    for service in range(1, TransitionId.L1Accept + 1):
        assert utils.first_service([dgram(service)]) == service


def test_a_present_service_zero_record_is_rejected():
    """The contract this issue settles: unsupported input raises, and is not
    confused with an absent record (which the GPU decoder reports as None)."""
    assert TransitionId.ClearReadout == 0
    with pytest.raises(RuntimeError, match='got: 0'):
        utils.first_service([dgram(TransitionId.ClearReadout)])


def test_service_zero_is_rejected_even_behind_absent_streams():
    """Leading absent streams are skipped, so the service-0 record becomes the
    first PRESENT one -- and must still raise rather than be passed over in
    favour of a later supported record."""
    with pytest.raises(RuntimeError, match='got: 0'):
        utils.first_service(
            [None, dgram(TransitionId.ClearReadout),
             dgram(TransitionId.Enable)])


def test_a_missing_dgram_is_a_different_error():
    """Nothing present at all is distinct from a present unsupported record:
    the first cannot name a service, the second names one that is rejected."""
    with pytest.raises(RuntimeError, match='No valid service'):
        utils.first_service([])
    with pytest.raises(RuntimeError, match='No valid service'):
        utils.first_service([None, None])


def test_the_first_present_dgram_wins():
    assert utils.first_service(
        [None, dgram(TransitionId.Enable), dgram(TransitionId.Disable)]
    ) == TransitionId.Enable


def test_a_dgram_that_cannot_report_env_is_skipped():
    """Pre-existing behaviour: an unusable dgram is passed over rather than
    raising, so a later valid one can still answer."""
    def explode():
        raise ValueError('no env on this dgram')

    assert utils.first_service(
        [NS(env=explode), dgram(TransitionId.Reset)]) == TransitionId.Reset
