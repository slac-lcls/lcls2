/**
 * @file
 * @brief EBeamTebData, the TEB input record for EBeam BLD data.
 */
#ifndef Pds_Trg_EBeamTebData_hh
#define Pds_Trg_EBeamTebData_hh

namespace Pds {
  namespace Trg {
    /** TEB input record for EBeam BLD data, filled by bldTebPrimitive.cc from ebeamL3Energy. */
    struct EBeamTebData
    {
        /** Store l3Energy_. */
        EBeamTebData(double l3Energy_) {
            l3Energy = l3Energy_;
        };
        double l3Energy;  ///< ebeamL3Energy value of the EBeam BLD record.
    };
  };
};

#endif
