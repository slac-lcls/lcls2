#include "psalg/digitizer/HsdPython.hh"

#include <stdio.h>
#include <ctype.h>

using namespace Pds::HSD;

void ChannelPython::fill_arrays(const uint32_t *evtheader, const uint8_t *data,
                                uint16_t* waveform, uint16_t* sPos,
                                uint16_t* len, uint16_t** fexPtr)
{
    // find out if we have raw/fex or both
    unsigned streams((evtheader[0]>>20)&0x3);
    const uint8_t* p = data;
    while(streams) {
        const StreamHeader& s = *reinterpret_cast<const StreamHeader*>(p);
        if (s.stream_id() == 0) {
            _parse_waveform(s, waveform);
        }
        if (s.stream_id() == 1) {
            _parse_peaks(s, sPos, len, fexPtr);
        }
        if (!s.num_samples()) {
            printf("hsd found zero samples in stream. exiting\n");
            abort();
        }
        p += sizeof(StreamHeader)+s.num_samples()*2;
        streams &= ~(1<<s.stream_id());
      }
}
