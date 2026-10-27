"""`first_service` accepts every real service id, including ClearReadout.

Service 0 is `TransitionId.ClearReadout`, a real control record. The original
range check rejected it as invalid, so any caller reaching a genuine
ClearReadout raised instead of handling it (issue #156, GPU-141-02).

This is a shared CPU-path function -- `Event.service()`, `psexp/run.py`,
`psexp/node.py`, `psexp/step.py`, `psexp/mpi_ds.py`, `drp_ds.py`,
`shmem_ds.py` and `singlefile_ds.py` all reach it -- so the accepted range is
CPU behaviour, not a GPU-local detail.
"""
from types import SimpleNamespace as NS

import pytest

from psana import utils
from psana.psexp import TransitionId


def dgram(service):
    """A stand-in carrying `service` where first_service reads it."""
    return NS(env=lambda: service << 24)


def test_every_real_service_id_is_accepted():
    """0 through 12 are all real records. 0 is ClearReadout, not an error."""
    for service in range(0, 13):
        assert utils.first_service([dgram(service)]) == service


def test_clearreadout_is_accepted():
    """Named explicitly, since this is the id the check used to reject."""
    assert utils.first_service(
        [dgram(TransitionId.ClearReadout)]) == TransitionId.ClearReadout


def test_the_enum_sentinel_is_still_rejected():
    """`NumberOf` = 13 bounds the enum; it is not a record that can arrive."""
    assert TransitionId.NumberOf == 13
    with pytest.raises(RuntimeError):
        utils.first_service([dgram(TransitionId.NumberOf)])


def test_a_missing_dgram_is_still_an_error():
    """No dgram at all is distinct from a dgram whose service is 0: the first
    is a failure, the second is ClearReadout."""
    with pytest.raises(RuntimeError, match='No valid service'):
        utils.first_service([])
    with pytest.raises(RuntimeError, match='No valid service'):
        utils.first_service([None, None])


def test_the_first_present_dgram_wins():
    """Absent streams are skipped, and a following service 0 is still
    returned rather than treated as "nothing found"."""
    assert utils.first_service(
        [None, dgram(TransitionId.ClearReadout), dgram(TransitionId.Enable)]
    ) == TransitionId.ClearReadout


def test_a_dgram_that_cannot_report_env_is_skipped():
    """Pre-existing behaviour: an unusable dgram is passed over rather than
    raising, so a later valid one can still answer."""
    def explode():
        raise ValueError('no env on this dgram')

    assert utils.first_service(
        [NS(env=explode), dgram(TransitionId.Reset)]) == TransitionId.Reset
