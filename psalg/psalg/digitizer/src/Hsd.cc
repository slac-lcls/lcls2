#include "psalg/digitizer/Hsd.hh"

#include <stdio.h>
#include <ctype.h>

using namespace Pds::HSD;

// TODO: test with fex only data
Channel::Channel(Allocator *allocator, Pds::HSD::Hsd_v1_2_3 *vHsd, const uint32_t *evtheader, const uint8_t* data)
: m_allocator(allocator)
, numPixels(0)
, numFexPeaks(0)
, waveform(allocator, maxSize)
, sPos(allocator, maxSize)
, len(allocator, maxSize)
, fexPtr(allocator, maxSize)
{
    const uint8_t *e = reinterpret_cast<const uint8_t*>(evtheader);
    unsigned streams(e[2]>>4);
    const uint8_t* p = data;
    while(streams) {
        const StreamHeader& s = *reinterpret_cast<const StreamHeader*>(p);
        if (s.stream_id() == 0) {
            _parse_waveform(s);
        }
        if (s.stream_id() == 1) {
            _parse_peaks(s);
        }
        p += sizeof(StreamHeader)+s.num_samples()*2;
        streams &= ~(1<<s.stream_id());
      }
}

Hsd_v1_2_3::Hsd_v1_2_3(Allocator *allocator)
: m_allocator(allocator)
{
    version = "1.2.3";
}
