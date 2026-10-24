/**
 * @file
 * @brief Frame constants and the packed UDP packet layout for Jungfrau data, used by Jungfrau.cc to assemble frames.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace Drp {
/** Frame constants and packed UDP packet layout for Jungfrau data. */
namespace JungfrauData {

/** Rows in one frame (512); used for the raw image shape in Jungfrau.cc. */
constexpr size_t Rows { 512 };
/** Columns in one frame (1024). */
constexpr size_t Cols { 1024 };
/** Number of UDP packets per frame (128). */
constexpr size_t PacketNum { 128 };
/** Pixels per frame (Rows * Cols). */
constexpr size_t PixelNum { Rows * Cols };
/** Pixels carried by one packet (PixelNum / PacketNum). */
constexpr size_t PixelPerPacket { PixelNum / PacketNum };
/** Payload bytes per packet (PixelPerPacket 16-bit pixels). */
constexpr size_t PayloadSize { PixelPerPacket * sizeof(uint16_t) };
/** Bytes per frame (PixelNum 16-bit pixels). */
constexpr size_t FrameSize { PixelNum * sizeof(uint16_t) };

#pragma pack(push)
#pragma pack(2)
/** Packet header, packed to 2-byte alignment, that precedes the pixel data of each packet. */
struct Header {
    uint64_t framenum;  ///< Frame number; Jungfrau.cc requires all packets of a frame to carry the same value.
    uint32_t exptime;  ///< Exposure time. Inferred from the name; not verified in code.
    uint32_t packetnum;  ///< Packet index within the frame; Jungfrau.cc copies the payload to offset packetnum * PayloadSize.
    uint64_t bunchid;  ///< Bunch ID. Inferred from the name; not verified in code.
    uint64_t timestamp;  ///< Timestamp; Jungfrau.cc requires all packets of a frame to carry the same value.
    uint16_t moduleID;  ///< Module ID; Jungfrau.cc checks it against the expected module.
    uint16_t xCoord;  ///< X coordinate. Inferred from the name; not verified in code.
    uint16_t yCoord;  ///< Y coordinate. Inferred from the name; not verified in code.
    uint16_t zCoord;  ///< Z coordinate. Inferred from the name; not verified in code.
    uint32_t debug;  ///< Debug word. Inferred from the name; not verified in code.
    uint16_t roundRobin;  ///< Round-robin value. Inferred from the name; not verified in code.
    uint8_t detectortype;  ///< Detector type code. Inferred from the name; not verified in code.
    uint8_t headerVersion;  ///< Header version. Inferred from the name; not verified in code.
};

/** One UDP packet: a Header followed by PixelPerPacket 16-bit pixel values. */
struct JungfrauPacket {
  Header header;  ///< Packet header.
  uint16_t data[PixelPerPacket];  ///< Pixel values carried by this packet.
};
#pragma pack(pop)

/** Size in bytes of one JungfrauPacket; Jungfrau.cc rejects packets of any other size. */
constexpr size_t PacketSize = sizeof(JungfrauPacket);

} // namespace JungfrauData
} // namespace Drp
