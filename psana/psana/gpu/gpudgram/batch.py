"""Device ABI for stream dgrams in one GPU event batch.

The first four columns are inputs supplied by the event builder/read path.
The GPU XTC walker fills the remaining columns directly in this same table.
No CPU event/stream regrouping table is part of this ABI.
"""


DGRAM_EVENT_INDEX = 0
DGRAM_STREAM_ID = 1
DGRAM_OFFSET = 2
DGRAM_SIZE = 3
DGRAM_TIMESTAMP = 4
DGRAM_ENV = 5
DGRAM_SERVICE = 6
DGRAM_DAMAGE = 7
DGRAM_TYPE = 8
DGRAM_STATUS = 9
DGRAM_NCOLS = 10


__all__ = [name for name in globals() if name.startswith("DGRAM_")]
