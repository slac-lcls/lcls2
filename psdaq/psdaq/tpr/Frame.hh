#ifndef Pds_Tpr_Frame
#define Pds_Tpr_Frame

namespace Pds {
  namespace Tpr {
    class Frame {
    public:
      uint64_t pulseId;
      uint64_t timeStamp;
      uint16_t rates;
      uint16_t acTimeSlot;
      uint32_t beamRequest;
      uint16_t beamEnergy[4];
      uint16_t photonWavelen[2];
      uint16_t reserved;
      uint16_t mpsLimit;
      uint16_t mpsClass[4];
      uint16_t control[18];
    };
  };
};

#endif
