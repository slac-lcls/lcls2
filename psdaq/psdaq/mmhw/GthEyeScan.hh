#ifndef Pds_GthEyeScan_hh
#define Pds_GthEyeScan_hh

#include <stdint.h>

namespace Pds {
  namespace Mmhw {
    class GthEyeScan {
    public:
      bool enabled() const;
      void enable(bool);
      void scan(const char* ofile,
                unsigned    prescale=0,
                unsigned    xscale=0,
                bool        lsparse=false);
      void run(unsigned& error_count,
               uint64_t& sample_count);
      static void progress(unsigned& row,
                           unsigned& col);
    public:
      uint32_t _reserved_3c[0x3c];
      uint32_t _es_control;  // [15:10] control, [9] errdet_en, [8], eye_scan_en, [4:0] prescale
      uint32_t _reserved_3f[0x3f-0x3d];
      uint32_t _es_qualifier[5];
      uint32_t _es_qual_mask[5];
      uint32_t _es_sdata_mask[5];
      uint32_t _reserved_rf[0x4f-0x4e];
      uint32_t _es_horz_offset; // [15:4]
      uint32_t _reserved_97[0x97-0x50];
      uint32_t _rx_eyescan_vs; // [10] neg_dir, [9] ut_sign, [8:2] code (vert_offset), [1:0] range
      uint32_t _reserved_151[0x151-0x98];
      uint32_t _es_error_count;         
      uint32_t _es_sample_count;         
      uint32_t _es_control_status;
      uint32_t _reserved_200 [0x200-0x154];
    };
  };
};

#endif

