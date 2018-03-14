#ifndef Pds_Tpr_Frame
#define Pds_Tpr_Frame

namespace Pds {
  namespace Tpr {
    class Frame {
    public:
      uint16_t channels;  // bit mask of channels receiving this event
      uint16_t msg_type;  // 0=Event, 1=BsaControl, 2=BsaChannel
      uint32_t msg_size;  // size of the message in 32-bit words
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
