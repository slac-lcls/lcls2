#ifndef Pgp_Reg_GthEyeScan_hh
#define Pgp_Reg_GthEyeScan_hh

#include "Reg.hh"

#include <stdio.h>

namespace Kcu {
    class GthEyeScan {
    public:
      bool enabled() const;
      void enable(bool);
      void scan(const char* ofile,
                unsigned    prescale=0,
                unsigned    xscale=0,
                bool        lsparse=false,
                bool        lhscan=false);
      void run(unsigned& error_count,
               uint64_t& sample_count);
      static void progress(unsigned& row,
                           unsigned& col);
    private:
      void _vscan(FILE*, unsigned, bool);
      void _hscan(FILE*, unsigned, bool);
    public:
      uint32_t _reserved_3c[0x3c];
      Reg      _es_control;  // [15:10] control, [9] errdet_en, [8], eye_scan_en, [4:0] prescale
      uint32_t _reserved_3f[0x3f-0x3d];
      Reg      _es_qualifier[5];
      Reg      _es_qual_mask[5];
      Reg      _es_sdata_mask[5];
      uint32_t _reserved_rf[0x4f-0x4e];
      Reg      _es_horz_offset; // [15:4]
      uint32_t _reserved_97[0x97-0x50];
      Reg      _rx_eyescan_vs; // [10] ut_sign, [9] neg_dir, [8:2] code (vert_offset), [1:0] range
      uint32_t _reserved_151[0x151-0x98];
      Reg      _es_error_count;         
      Reg      _es_sample_count;         
      Reg      _es_control_status;
      uint32_t _reserved_200 [0x200-0x154];
    };
};

#endif

