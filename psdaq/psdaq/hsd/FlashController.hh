#ifndef Pds_HSD_FlashController_hh
#define Pds_HSD_FlashController_hh

#include "Globals.hh"

#include <stdio.h>
#include <vector>

namespace Pds {
  namespace HSD {
    class FlashController {
    public:
      void write (const char*);
      void verify(const char*);
      std::vector<uint8_t> read(unsigned nwords);
      static void verbose(bool);
      static void useFifo(bool);
    public:
      void _write (const unsigned* p, unsigned nwords);
      void _write (const std::vector<uint8_t>&);
      int  _verify(const unsigned* p, unsigned nwords);
      int  _verify(const std::vector<uint8_t>&);
      void _read  (std::vector<uint8_t>&, unsigned nwords);
    private:
      vuint32_t _reserved0[3];
      vuint32_t _destn;  // user=0, safe=0xff
      vuint32_t _bytes_to_prog;
      vuint32_t _prog_fifo_cnt;
      vuint32_t _bytes_to_read;
      vuint32_t _reserved7[9];
      vuint32_t _command;
      //  b0  = flash_init/fifo_reset
      //  b15 = start_read (unnecessary)
      //  b17 = start_prog (unnecessary)
      vuint32_t _reserved17[15];
      vuint32_t _data;
    };
  };
};

#endif
