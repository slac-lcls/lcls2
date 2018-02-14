#ifndef HSD_I2cDevice_hh
#define HSD_I2cDevice_hh

namespace Pds {
  namespace HSD {
    class I2cSwitch {
    public:
      void    set(uint8_t);
      uint8_t get() const;
    private:
      volatile uint32_t _control;
      uint32_t          _reserved[255];
    };
  };
};

#endif
