#ifndef HSD_I2cSwitch_hh
#define HSD_I2cSwitch_hh

#include <stdint.h>

namespace Pds {
  namespace HSD {
    class I2cSwitch {
    public:
      enum Port { PrimaryFmc=1, SecondaryFmc=2, SFP=4, LocalBus=8 };
      void select(Port);
      void dump () const;
    private:
      volatile uint32_t _control;
      uint32_t          _reserved[255];
    };
  };
};

#endif
