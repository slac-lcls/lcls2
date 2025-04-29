#ifndef Pds_Trg_TripperTebData_hh
#define Pds_Trg_TripperTebData_hh
#include <cstdint>
#include <stdint.h>
#include <string>
#include <cstring>

namespace Pds
{
namespace Trg
{
#pragma pack(push,1)
struct TripperTebData {
    TripperTebData(const uint16_t _hotPixelThresh,
                   const uint32_t _numHotPixels,
                   const uint32_t _maxHotPixels,
                   const char* _detType)
      : hotPixelThresh{_hotPixelThresh}
      , numHotPixels{_numHotPixels}
      , maxHotPixels{_maxHotPixels}
    {
        memcpy(detType, _detType, strlen(_detType));
    };
    const uint16_t hotPixelThresh; ///< ADU Threshold for considering a pixel "hot"
    const uint32_t numHotPixels; ///< Number of hot pixels found in this contribution
    const uint32_t maxHotPixels; ///< Max number of hot pixels across ALL contributions before tripping
    char detType[30]; ///< Detector type so TEB decides if contributes to tripping
};
#pragma pack(pop)
}; // namespace Trg
}; // namespace Pds

#endif
