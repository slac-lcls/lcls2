#include "psalg/digitizer/Hsd.hh"

#include <stdio.h>
#include <ctype.h>

using namespace Pds::HSD;
using namespace psalg;

Channel::Channel(Allocator *allocator, const uint32_t *evtheader, const uint8_t* data)
: m_allocator(allocator)
, numPixels(0)
, numFexPeaks(0)
, waveform(allocator, maxSize)
, sPos(allocator, maxSize)
, len(allocator, maxSize)
, fexPtr(allocator, maxSize)
{
    // find out if we have raw/fex or both
    unsigned streams((evtheader[0]>>20)&0x3);
    const uint8_t* p = data;
    while(streams) {
        const StreamHeader& s = *reinterpret_cast<const StreamHeader*>(p);
        if (s.stream_id() == 0) {
            _parse_waveform(s);
        }
        if (s.stream_id() == 1) {
            _parse_peaks(s);
        }
        if (!s.num_samples()) {
            printf("hsd found zero samples in stream. exiting\n");
            abort();
        }
        p += sizeof(StreamHeader)+s.num_samples()*2;
        streams &= ~(1<<s.stream_id());
      }
}
