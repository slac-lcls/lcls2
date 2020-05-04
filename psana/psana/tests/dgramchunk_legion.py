from psana import dgramchunk, dgram
import pygion
import os

@pygion.task
def do_chunk(view):
    config = dgram.Dgram()
    offset = 0
    while offset < len(view):
        d = dgram.Dgram(config=config, view=view, offset=offset)
        offset += memoryview(d).shape[0]

@pygion.task(top_level=True)
def main():
    fd = os.open('/reg/d/psdm/xpp/xpptut15/scratch/mona/smd.xtc', os.O_RDONLY)
    config = dgram.Dgram(file_descriptor=fd)
    print(config)

    # Iterate chunks of n_events
    n_events = 10000
    dchunk = dgramchunk.DgramChunk(fd)
    displacement = memoryview(config).shape[0]
    view = dchunk.get(displacement, n_events)
    while view != 0:
        do_chunk(bytes(view)) # FIXME: Find a way to pickle without copying
        displacement += view.nbytes
        view = dchunk.get(displacement, n_events)
