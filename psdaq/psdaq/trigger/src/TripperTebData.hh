#ifndef Pds_Trg_TripperTebData_hh
#define Pds_Trg_TripperTebData_hh

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace Pds
{
namespace Trg
{
#pragma pack(push,1)
struct TripperTebData {
    TripperTebData(uint16_t _hotPixelThresh,
                   uint32_t _numHotPixels,
                   uint32_t _maxHotPixels,
                   const char* _detType)
    {
        new (&smalldata) SmallData {
            { _hotPixelThresh, _numHotPixels, _maxHotPixels }
        };
        size_t len = std::min(std::strlen(_detType), sizeof(detType) - 1);
        std::memcpy(detType, _detType, len);
        detType[len] = '\0';
    };

    TripperTebData(uint16_t* _seqInfo, const char* _detType)
    {
        new (&smalldata) SmallData{};
        std::memcpy(smalldata.seqInfo, _seqInfo, sizeof(smalldata.seqInfo));

        size_t len = std::min(std::strlen(_detType), sizeof(detType) - 1);
        std::memcpy(detType, _detType, len);
        detType[len] = '\0';
    }

    union SmallData {
        // For the Jungfrau. Dummy contributions from unused detectors also use struct
        struct {
            uint16_t hotPixelThresh; ///< ADU Threshold for considering a pixel "hot"
            uint32_t numHotPixels;   ///< Number of hot pixels found in this contribution
            uint32_t maxHotPixels;   ///< Max number of hot pixels across ALL contributions before tripping
        };
        // For the timing detector
        uint16_t seqInfo[18];       ///< Holds the event code/sequence info
    } smalldata;

    // Functions as the union tag. TEB uses to decide contribution to "tripping"
    // or event code selection.
    char detType[30];
};
#pragma pack(pop)
}; // namespace Trg
}; // namespace Pds

#endif
