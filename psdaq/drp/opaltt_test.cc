#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <getopt.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <vector>
#include "OpalTTFex.hh"

namespace PdsL1 {
  class Xtc {
  public:
    char* payload() { return (char*)(this+1); }
    uint32_t damage;
    uint32_t src_log;
    uint32_t src_phy;
    uint32_t contains;
    uint32_t extent;
  };
  class FrameV1 {
  public:
    uint32_t	_width;	/**< Number of pixels in a row. */
    uint32_t	_height;	/**< Number of pixels in a column. */
    uint32_t	_depth;	/**< Number of bits per pixel. */
    uint32_t	_offset;	/**< Fixed offset/pedestal value of pixel data. */
    //uint8_t	_pixel_data[this->_width*this->_height*((this->_depth+7)/8)];
  };
  class FIFOEvent {
  public:
    uint32_t	_timestampHigh;	/**< 119 MHz timestamp (fiducial) */
    uint32_t	_timestampLow;	/**< 360 Hz timestamp */
    uint32_t	_eventCode;	/**< event code (range 0-255) */
  };
  class EvrDataV4 {
  public:
    uint32_t	_u32NumFifoEvents;	/**< length of FIFOEvent list */
    FIFOEvent*  _events() { return reinterpret_cast<FIFOEvent*>(this+1); }
    //EvrData::FIFOEvent	_fifoEvents[this->_u32NumFifoEvents];
  };
  class TimeToolDataV1 {
  public:
    uint32_t	_event_type;	/**< Event designation */
    uint32_t	_z;
    double	_amplitude;	/**< Amplitude of the edge */
    double	_position_pixel;	/**< Filtered pixel position of the edge */
    double	_position_time;	/**< Filtered time position of the edge */
    double	_position_fwhm;	/**< Full-width half maximum of filtered edge (in pixels) */
    double	_nxt_amplitude;	/**< Amplitude of the next largest edge */
    double	_ref_amplitude;	/**< Amplitude of reference at the edge */
    //int32_t	_projected_signal[cfg.signal_projection_size()];
    //int32_t	_projected_sideband[cfg.sideband_projection_size()];
  };
};

static void _load_xtc(std::vector<uint8_t>&, const char*);

static void usage(const char* p)
{
  printf("Usage: %p -f <filename> [-n <events>]\n",p);
}

int main(int argc, char* argv[])
{
  char* filename = 0;
  unsigned nevt = 100;
  int c;
  while( (c = getopt(argc, argv, "f:n:")) != EOF) {
    switch(c) {
    case 'f':
      filename = optarg;
      break;
    case 'n':
      nevt = strtoul(optarg,NULL,0);
      break;
    default:
      usage(argv[0]);
      exit(1);
    }
  }

  if (!filename) {
    usage(argv[0]);
    exit(2);
  }

  unsigned _evtindex = 0;
  std::vector<uint8_t> _evtbuffer;
  _load_xtc(_evtbuffer, filename);

  while(nevt--) {
    printf("index: %u\n",_evtindex);
#define L1PAYLOAD(ptype,f)                                              \
    ptype& f = *reinterpret_cast<ptype*>( reinterpret_cast<PdsL1::Xtc*>(&_evtbuffer[_evtindex])->payload() ); \
    _evtindex += reinterpret_cast<PdsL1::Xtc*>(&_evtbuffer[_evtindex])->extent

    L1PAYLOAD(PdsL1::FrameV1       ,f);
    L1PAYLOAD(PdsL1::EvrDataV4     ,e);
    L1PAYLOAD(PdsL1::TimeToolDataV1,t);
    if (_evtindex >= _evtbuffer.size())
      _evtindex=0;

    printf("Read amp [%f] pos [%f] pos_t [%f] fwhm [%f] nxta [%f] refa [%f]\n",
           t._amplitude,
           t._position_pixel,
           t._position_time,
           t._position_fwhm,
           t._nxt_amplitude,
           t._ref_amplitude);

    printf("Image width [%u] height [%u]\n", f._width, f._height);

    // transfer event codes into EventInfo
    Drp::EventInfo info;
    memset(info._seqInfo, 0, sizeof(info._seqInfo));
    for(unsigned i=0; i<e._u32NumFifoEvents; i++) {
      unsigned ec = e._events()[i]._eventCode;
      info._seqInfo[ec>>4] |= (1<<(ec&0x1f));
    }
    printf("EventInfo seq [");
    for(unsigned i=0; i<16; i++)
      printf(" %04x", info._seqInfo[i]);
    printf("\n");
    printf("---\n");
  }
  return 0;
}

void _load_xtc(std::vector<uint8_t>& buffer, const char* filename)
{
  int fd = open(filename, O_RDONLY);
  if (fd < 0)
    throw "Error opening file";

  struct stat s;
  if (fstat(fd, &s)) {
    perror("Error fetching file size");
    exit(3);
  }

  buffer.resize(s.st_size);
  int bytes = read(fd, buffer.data(), s.st_size);
  if (bytes != s.st_size) {
    perror("Error reading all bytes");
    exit(4);
  }
}
