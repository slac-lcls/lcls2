"""CPU callback managers must also satisfy the GPU branch's EB batch interface."""

import pytest

from psana.psexp.eventbuilder_manager import CallbackEventBuilderManager


class CallbackStream:
    def __init__(self, batches, fail=False):
        self.batches = batches
        self.fail = fail
        self.closed = False

    def __iter__(self):
        yield from self.batches
        if self.fail:
            raise RuntimeError("callback failed")

    def close(self):
        self.closed = True


@pytest.mark.parametrize("method", ["batches", "batches_with_gpu"])
@pytest.mark.parametrize("ending", ["exhaust", "close", "error"])
def test_callback_batch_interface(method, ending):
    event_batch = {0: (bytearray(b"events"), [6])}
    step_batch = {0: (bytearray(b"steps"), [5])}
    final_steps = {0: (bytearray(b"endrun"), [6])}
    stream = CallbackStream(
        [(event_batch, step_batch), ({}, final_steps)], fail=ending == "error"
    )
    manager = CallbackEventBuilderManager.__new__(CallbackEventBuilderManager)
    manager.stream = stream
    batches = getattr(manager, method)()

    expected = (event_batch, {}, step_batch) if method == "batches_with_gpu" else (
        event_batch, step_batch
    )
    assert next(batches) == expected
    assert not stream.closed
    if ending == "close":
        batches.close()
    else:
        expected = ({}, {}, final_steps) if method == "batches_with_gpu" else (
            {}, final_steps
        )
        assert next(batches) == expected
        with pytest.raises(RuntimeError if ending == "error" else StopIteration):
            next(batches)
    assert stream.closed
