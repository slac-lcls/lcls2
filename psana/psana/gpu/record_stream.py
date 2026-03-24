from __future__ import annotations

from dataclasses import dataclass

from psana import utils
from psana.event import Event
from psana.psexp import TransitionId


@dataclass(frozen=True)
class TransitionRecord:
    dgrams: list
    service: int
    event: Event | None = None

    @property
    def is_transition(self):
        return True


@dataclass(frozen=True)
class L1Record:
    dgrams: list
    service: int
    event: Event

    @property
    def is_transition(self):
        return False


def iter_records(evt_iter, run_ctx):
    for dgrams in evt_iter:
        service = utils.first_service(dgrams)
        if TransitionId.isEvent(service):
            yield L1Record(dgrams=dgrams, service=service, event=Event(dgrams=dgrams, run=run_ctx))
        else:
            yield TransitionRecord(dgrams=dgrams, service=service)
