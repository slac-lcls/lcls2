#ifndef PSDAQ_TIMING_HEADER_H
#define PSDAQ_TIMING_HEADER_H

#pragma pack(push,4)

  class AxiStreamBatcherHeader{
public:
    uint32_t evtCounter;
    uint32_t _opaque[2];
};

#pragma pack(pop)

#endif
