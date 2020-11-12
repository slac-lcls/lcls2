#ifndef Fmc134CPld_hh
#define Fmc134CPld_hh

#include "Globals.hh"
#include <string>

namespace Pds {
    namespace HSD {
        class Fmc134Cpld {
        public:
            Fmc134Cpld() {}
        public:
            void enableClock(bool);
            void initialize(bool lDualChannel=false,
                            bool lInternalRef=false);
            void enable_adc_prbs(bool);
            void enable_mon     (bool);
        public:
            void lmk_dump();
            void lmx_dump();
            void adc_dump(unsigned);
            std::string adc_cal_dump(unsigned);
            void        adc_cal_load(unsigned,const std::string&);
            //      void adc_range(unsigned,float fs_vpp);
            void adc_range(unsigned,unsigned fsrng);
        private:
            void _hmc_init();
            void _lmk_init();
            void _lmx_init(bool);
            void _adc_init(unsigned,bool);
        public:
            enum DevSel { ADC0=0x01, ADC1=0x02, ADC_BOTH=0x03,
                          LMX =0x04, LMK =0x08, HMC     =0x10 };
            void     writeRegister( DevSel   dev,
                                    unsigned address,
                                    unsigned data );
            unsigned readRegister ( DevSel   dev,
                                    unsigned address );
        private:
            unsigned _read();
        public:
            void dump() const;
        public:
            enum { CLOCKTREE_CLKSRC_INTERNAL=0, 
                   CLOCKTREE_CLKSRC_EXTERNAL=1,
                   CLOCKTREE_REFSRC_EXTERNAL=2,
                   CLOCKTREE_REFSRC_STACKED =3 };
            int32_t default_clocktree_init(unsigned clockmode = CLOCKTREE_CLKSRC_INTERNAL);  // factory default config
            enum AdcCalibMode { NO_CAL, FG_CAL, BG_CAL };
            int32_t default_adc_init      (AdcCalibMode,
                                           std::string& adc0_alib, 
                                           std::string& adc1_calib);
            int32_t config_prbs           (unsigned);
        private:
            int32_t internal_ref_and_lmx_enable(uint32_t i2c_unit, uint32_t clockmode);
            int32_t reset_clock_chip(int32_t);
        private:
            vuint32_t _command; // device select
            vuint32_t _control0; //
            vuint32_t _control1; //
            uint32_t  _reserved3;
            vuint32_t _status;
            vuint32_t _version;
            vuint32_t _i2c_data[4]; // write cache
            vuint32_t _i2c_read[4]; // read cache
            uint32_t  _reserved[0x100-14];
        };
    };
};

#endif
