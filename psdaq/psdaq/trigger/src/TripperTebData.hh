/**
 * @file
 * @brief TripperTebData, the packed TEB input record of the MFX tripper trigger (Jungfrau hot-pixel counts or timing sequence words, tagged by detector type).
 */
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
/** Packed TEB input record for the MFX tripper trigger: either hot-pixel information (Jungfrau and dummy contributions) or the timing sequence words, with detType acting as the tag. */
struct TripperTebData {
    /** Fill the hot-pixel fields and copy _detType (truncated to 29 characters and NUL-terminated). */
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

    /** Zero the union, copy 18 sequence words from _seqInfo and copy _detType (truncated to 29 characters and NUL-terminated). */
    TripperTebData(uint16_t* _seqInfo, const char* _detType)
    {
        new (&smalldata) SmallData{};
        std::memcpy(smalldata.seqInfo, _seqInfo, sizeof(smalldata.seqInfo));

        size_t len = std::min(std::strlen(_detType), sizeof(detType) - 1);
        std::memcpy(detType, _detType, len);
        detType[len] = '\0';
    }

    /** Union of the hot-pixel fields and the 18 timing sequence words. */
    union SmallData {
        // For the Jungfrau. Dummy contributions from unused detectors also use struct
        /** Hot-pixel fields used for the Jungfrau and dummy contributions (per the code comment). */
        struct {
            uint16_t hotPixelThresh; ///< ADU Threshold for considering a pixel "hot"
            uint32_t numHotPixels;   ///< Number of hot pixels found in this contribution
            uint32_t maxHotPixels;   ///< Max number of hot pixels across ALL contributions before tripping
        };
        // For the timing detector
        uint16_t seqInfo[18];       ///< Holds the event code/sequence info
    } smalldata;  ///< The detector-specific data, selected by detType.

    // Functions as the union tag. TEB uses to decide contribution to "tripping"
    // or event code selection.
    /** NUL-terminated detector type; the code comment says the TEB uses it as the union tag to choose between tripping and event code selection. */
    char detType[30];
};
#pragma pack(pop)
}; // namespace Trg
}; // namespace Pds

#endif
