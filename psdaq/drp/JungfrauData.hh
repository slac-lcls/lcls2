#pragma once

#include <cstdint>

namespace Drp {
namespace JungfrauData {

constexpr size_t Rows { 512 };
constexpr size_t Cols { 1024 };
constexpr size_t PacketNum { 128 };
constexpr size_t PixelNum { Rows * Cols };
constexpr size_t PixelPerPacket { PixelNum / PacketNum };
constexpr size_t PayloadSize { PixelPerPacket * sizeof(uint16_t) };
constexpr size_t FrameSize { PixelNum * sizeof(uint16_t) };

#pragma pack(push)
#pragma pack(2)
struct Header {
    uint64_t framenum;
    uint32_t exptime;
    uint32_t packetnum;
    uint64_t bunchid;
    uint64_t timestamp;
    uint16_t moduleID;
    uint16_t xCoord;
    uint16_t yCoord;
    uint16_t zCoord;
    uint32_t debug;
    uint16_t roundRobin;
    uint8_t detectortype;
    uint8_t headerVersion;
};

struct JungfrauPacket {
  Header header;
  uint16_t data[PixelPerPacket];
};
#pragma pack(pop)

constexpr size_t PacketSize = sizeof(JungfrauPacket);

} // namespace JungfrauData
} // namespace Drp
