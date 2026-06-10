#pragma once

namespace psana_gpu {

static constexpr unsigned int XTC_HEADER_NBYTES = 12;
static constexpr unsigned int DGRAM_HEADER_NBYTES = 24;

static constexpr unsigned int DGRAM_TIME_LOW_OFFSET = 0;
static constexpr unsigned int DGRAM_TIME_HIGH_OFFSET = 4;
static constexpr unsigned int DGRAM_ENV_OFFSET = 8;
static constexpr unsigned int DGRAM_XTC_OFFSET = 12;

static constexpr unsigned int XTC_SRC_OFFSET = 0;
static constexpr unsigned int XTC_DAMAGE_OFFSET = 4;
static constexpr unsigned int XTC_CONTAINS_OFFSET = 6;
static constexpr unsigned int XTC_EXTENT_OFFSET = 8;

__device__ inline unsigned short load_u16_le(const unsigned char* p)
{
    return static_cast<unsigned short>(p[0]) |
           (static_cast<unsigned short>(p[1]) << 8);
}

__device__ inline unsigned int load_u32_le(const unsigned char* p)
{
    return static_cast<unsigned int>(p[0]) |
           (static_cast<unsigned int>(p[1]) << 8) |
           (static_cast<unsigned int>(p[2]) << 16) |
           (static_cast<unsigned int>(p[3]) << 24);
}

struct GpuXtcLite {
    const unsigned char* ptr;

    __device__ explicit GpuXtcLite(const unsigned char* xtc_ptr) : ptr(xtc_ptr) {}

    __device__ unsigned int src() const
    {
        return load_u32_le(ptr + XTC_SRC_OFFSET);
    }

    __device__ unsigned short damage() const
    {
        return load_u16_le(ptr + XTC_DAMAGE_OFFSET);
    }

    __device__ unsigned short contains() const
    {
        return load_u16_le(ptr + XTC_CONTAINS_OFFSET);
    }

    __device__ unsigned int extent() const
    {
        return load_u32_le(ptr + XTC_EXTENT_OFFSET);
    }

    __device__ const unsigned char* payload() const
    {
        return ptr + XTC_HEADER_NBYTES;
    }

    __device__ unsigned int payload_size() const
    {
        const unsigned int xtc_extent = extent();
        return xtc_extent >= XTC_HEADER_NBYTES ?
               xtc_extent - XTC_HEADER_NBYTES : 0;
    }
};

struct GpuDgramLite {
    const unsigned char* ptr;

    __device__ explicit GpuDgramLite(const unsigned char* dgram_ptr) : ptr(dgram_ptr) {}

    __device__ unsigned long long timestamp() const
    {
        const unsigned int low = load_u32_le(ptr + DGRAM_TIME_LOW_OFFSET);
        const unsigned int high = load_u32_le(ptr + DGRAM_TIME_HIGH_OFFSET);
        return (static_cast<unsigned long long>(high) << 32) | low;
    }

    __device__ unsigned int env() const
    {
        return load_u32_le(ptr + DGRAM_ENV_OFFSET);
    }

    __device__ unsigned int service() const
    {
        return (env() >> 24) & 0xf;
    }

    __device__ GpuXtcLite xtc() const
    {
        return GpuXtcLite(ptr + DGRAM_XTC_OFFSET);
    }

    __device__ unsigned int extent() const
    {
        return xtc().extent();
    }

    __device__ const unsigned char* payload() const
    {
        return xtc().payload();
    }

    __device__ unsigned int payload_size() const
    {
        return xtc().payload_size();
    }
};

} // namespace psana_gpu
