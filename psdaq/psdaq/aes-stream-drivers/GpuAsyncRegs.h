/**
 * ----------------------------------------------------------------------------
 * Company    : SLAC National Accelerator Laboratory
 * ----------------------------------------------------------------------------
 * Description:
 *    Defines register locations and bit offsets into AxiPcieGpuAsyncCore
 * ----------------------------------------------------------------------------
 * This file is part of the aes_stream_drivers package. It is subject to
 * the license terms in the LICENSE.txt file found in the top-level directory
 * of this distribution and at:
 *    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
 * No part of the aes_stream_drivers package, including this file, may be
 * copied, modified, propagated, or distributed except according to the terms
 * contained in the LICENSE.txt file.
 * ----------------------------------------------------------------------------
**/

#ifndef __GPU_ASYNC_REGS_H__
#define __GPU_ASYNC_REGS_H__

#include <linux/types.h>

#ifdef DMA_IN_KERNEL
#include <linux/io.h>
#endif

// Offset of AxiPcieGpuAsyncCore. Must match the value within the firmware
#define GPU_ASYNC_CORE_OFFSET     0x00028000

struct GpuAsyncRegister {
    uint32_t offset;
    uint32_t bitOffset;
    uint32_t bitMask;
};

static inline uint32_t readGpuAsyncReg(const volatile void* baseptr, const struct GpuAsyncRegister* reg) {
#ifdef DMA_IN_KERNEL
    return (readl((uint8_t*)baseptr + reg->offset) & reg->bitMask) >> reg->bitOffset;
#else
    uint32_t val = reg->bitMask & *(const volatile uint32_t*)(((const volatile uint8_t*)baseptr) + reg->offset);
    return val >> reg->bitOffset;
#endif
}

static inline void writeGpuAsyncReg(volatile void* baseptr, const struct GpuAsyncRegister* reg, uint32_t value) {
    volatile uint32_t* regp = (volatile uint32_t*)(volatile uint8_t*)baseptr + reg->offset;
#ifdef DMA_IN_KERNEL
    writel((readl(regp) & ~reg->bitMask) | ((value << reg->bitOffset) & reg->bitMask), regp);
#else
    *regp = (*regp & ~reg->bitMask) | ((value << reg->bitOffset) & reg->bitMask);
#endif
}

#define GPU_ASYNC_DEF_REG(_name, _off, _bitOff, _bitMask)           \
static const struct GpuAsyncRegister GpuAsyncReg_ ## _name = {      \
    .offset = _off,                                                 \
    .bitOffset = _bitOff,                                           \
    .bitMask = _bitMask                                             \
};

GPU_ASYNC_DEF_REG(ArCache,      0x4, 0,  0xFF);
GPU_ASYNC_DEF_REG(AwCache,      0x4, 8,  0xFF00);
GPU_ASYNC_DEF_REG(DmaDataBytes, 0x4, 16, 0xFF0000);
GPU_ASYNC_DEF_REG(MaxBuffers,   0x4, 24, 0x1F000000);

GPU_ASYNC_DEF_REG(WriteCount,   0x8, 0,  0xFF);
GPU_ASYNC_DEF_REG(WriteEnable,  0x8, 8,  0xFF00);
GPU_ASYNC_DEF_REG(ReadCount,    0x8, 16, 0xFF0000);
GPU_ASYNC_DEF_REG(ReadEnable,   0x8, 24, 0xFF000000);

GPU_ASYNC_DEF_REG(RxFrameCnt,       0x10, 0, 0xFFFFFFFF);
GPU_ASYNC_DEF_REG(TxFrameCnt,       0x14, 0, 0xFFFFFFFF);
GPU_ASYNC_DEF_REG(AxiWriteErrorCnt, 0x18, 0, 0xFFFFFFFF);
GPU_ASYNC_DEF_REG(AxiReadErrorCnt,  0x1C, 0, 0xFFFFFFFF);

GPU_ASYNC_DEF_REG(CntRst,           0x20, 0, 0xFFFFFFFF);
GPU_ASYNC_DEF_REG(AxiWriteErrorVal, 0x24, 0, 0xFFFFFFFF);
GPU_ASYNC_DEF_REG(AxiReadErrorVal,  0x28, 0, 0xFFFFFFFF);

GPU_ASYNC_DEF_REG(DynamicRouteMasks0, 0x2C, 0,  0xFF);
GPU_ASYNC_DEF_REG(DynamicRouteDests0, 0x2C, 8,  0xFF00);
GPU_ASYNC_DEF_REG(DynamicRouteMasks1, 0x2C, 16, 0xFF0000);
GPU_ASYNC_DEF_REG(DynamicRouteDests1, 0x2C, 24, 0xFF000000);

// The following register defintiions are firmware specific. GpuAsyncCore can have up to 16 buffers, but defaults to 4.
// You must check the MaxBuffers register for the true value

/*********************** Write Buffers ************************/

#define GPU_ASYNC_REG_WRITE_BASE 256

#define GPU_ASYNC_REG_WRITE_ADDR_L_OFFSET(_i) (GPU_ASYNC_REG_WRITE_BASE + _i * 16 + 0)
#define GPU_ASYNC_REG_WRITE_ADDR_H_OFFSET(_i) (GPU_ASYNC_REG_WRITE_BASE + _i * 16 + 4)
#define GPU_ASYNC_REG_WRITE_SIZE_OFFSET(_i) (GPU_ASYNC_REG_WRITE_BASE + _i * 16 + 8)

#define GPU_ASYNC_DEF_WRITE_REGISTER(_i) \
    GPU_ASYNC_DEF_REG(WriteBuffer ## _i ## _WriteAddrL, GPU_ASYNC_REG_WRITE_ADDR_L_OFFSET(_i), 0, 0xFFFFFFFF) \
    GPU_ASYNC_DEF_REG(WriteBuffer ## _i ## _WriteAddrH, GPU_ASYNC_REG_WRITE_ADDR_H_OFFSET(_i), 0, 0xFFFFFFFF) \
    GPU_ASYNC_DEF_REG(WriteBuffer ## _i ## _WriteSize,  GPU_ASYNC_REG_WRITE_SIZE_OFFSET(_i),   0, 0xFFFFFFFF)

GPU_ASYNC_DEF_WRITE_REGISTER(0)
GPU_ASYNC_DEF_WRITE_REGISTER(1)
GPU_ASYNC_DEF_WRITE_REGISTER(2)
GPU_ASYNC_DEF_WRITE_REGISTER(3)
GPU_ASYNC_DEF_WRITE_REGISTER(4)
GPU_ASYNC_DEF_WRITE_REGISTER(5)
GPU_ASYNC_DEF_WRITE_REGISTER(6)
GPU_ASYNC_DEF_WRITE_REGISTER(7)
GPU_ASYNC_DEF_WRITE_REGISTER(8)
GPU_ASYNC_DEF_WRITE_REGISTER(9)
GPU_ASYNC_DEF_WRITE_REGISTER(10)
GPU_ASYNC_DEF_WRITE_REGISTER(11)
GPU_ASYNC_DEF_WRITE_REGISTER(12)
GPU_ASYNC_DEF_WRITE_REGISTER(13)
GPU_ASYNC_DEF_WRITE_REGISTER(14)
GPU_ASYNC_DEF_WRITE_REGISTER(15)

#undef GPU_ASYNC_DEF_WRITE_REGISTER

/*********************** Read Buffers ************************/

#define GPU_ASYNC_REG_READ_BASE 512

#define GPU_ASYNC_REG_READ_ADDR_L_OFFSET(_i) (GPU_ASYNC_REG_READ_BASE + _i * 16 + 0)
#define GPU_ASYNC_REG_READ_ADDR_H_OFFSET(_i) (GPU_ASYNC_REG_READ_BASE + _i * 16 + 4)

#define GPU_ASYNC_DEF_READ_REGISTER(_i) \
    GPU_ASYNC_DEF_REG(ReadBuffer ## _i ## _ReadAddrL, GPU_ASYNC_REG_READ_ADDR_L_OFFSET(_i), 0, 0xFFFFFFFF) \
    GPU_ASYNC_DEF_REG(ReadBuffer ## _i ## _ReadAddrH, GPU_ASYNC_REG_READ_ADDR_H_OFFSET(_i), 0, 0xFFFFFFFF)

GPU_ASYNC_DEF_READ_REGISTER(0)
GPU_ASYNC_DEF_READ_REGISTER(1)
GPU_ASYNC_DEF_READ_REGISTER(2)
GPU_ASYNC_DEF_READ_REGISTER(3)
GPU_ASYNC_DEF_READ_REGISTER(4)
GPU_ASYNC_DEF_READ_REGISTER(5)
GPU_ASYNC_DEF_READ_REGISTER(6)
GPU_ASYNC_DEF_READ_REGISTER(7)
GPU_ASYNC_DEF_READ_REGISTER(8)
GPU_ASYNC_DEF_READ_REGISTER(9)
GPU_ASYNC_DEF_READ_REGISTER(10)
GPU_ASYNC_DEF_READ_REGISTER(11)
GPU_ASYNC_DEF_READ_REGISTER(12)
GPU_ASYNC_DEF_READ_REGISTER(13)
GPU_ASYNC_DEF_READ_REGISTER(14)
GPU_ASYNC_DEF_READ_REGISTER(15)

#undef GPU_ASYNC_DEF_READ_REGISTER

/*********************** Write Detect ************************/

#define GPU_ASYNC_REG_WRITE_DETECT_BASE 768

#define GPU_ASYNC_REG_WRITE_DETECT_OFFSET(_i) (GPU_ASYNC_REG_WRITE_DETECT_BASE + _i * 4)

#define GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(_i) \
    GPU_ASYNC_DEF_REG(WriteBuffer ## _i ## _WriteEn, GPU_ASYNC_REG_WRITE_DETECT_OFFSET(_i), 0, 0xFFFFFFFF)

GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(0)
GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(1)
GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(2)
GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(3)
GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(4)
GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(5)
GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(6)
GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(7)
GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(8)
GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(9)
GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(10)
GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(11)
GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(12)
GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(13)
GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(14)
GPU_ASYNC_DEF_WRITE_DETECT_REGISTER(15)

#undef GPU_ASYNC_DEF_WRITE_DETECT_REGISTER

/*********************** Read Detect ************************/

#define GPU_ASYNC_REG_READ_DETECT_BASE 1024

#define GPU_ASYNC_REG_REMOTE_READ_SIZE_OFFSET(_i) (GPU_ASYNC_REG_READ_DETECT_BASE + _i * 4 + 0)
#define GPU_ASYNC_REG_REMOTE_READ_DETECT_OFFSET(_i) (GPU_ASYNC_REG_READ_DETECT_BASE + _i * 4 + 4)

#define GPU_ASYNC_DEF_READ_DETECT_REGISTER(_i) \
    GPU_ASYNC_DEF_REG(ReadBuffer ## _i ## _RemoteReadSize,  GPU_ASYNC_REG_REMOTE_READ_SIZE_OFFSET(_i), 0, 0xFFFFFFFF) \
    GPU_ASYNC_DEF_REG(ReadBuffer ## _i ## _RemoteReadEn,    GPU_ASYNC_REG_REMOTE_READ_DETECT_OFFSET(_i), 0, 0xFFFFFFFF)

GPU_ASYNC_DEF_READ_DETECT_REGISTER(0)
GPU_ASYNC_DEF_READ_DETECT_REGISTER(1)
GPU_ASYNC_DEF_READ_DETECT_REGISTER(2)
GPU_ASYNC_DEF_READ_DETECT_REGISTER(3)
GPU_ASYNC_DEF_READ_DETECT_REGISTER(4)
GPU_ASYNC_DEF_READ_DETECT_REGISTER(5)
GPU_ASYNC_DEF_READ_DETECT_REGISTER(6)
GPU_ASYNC_DEF_READ_DETECT_REGISTER(7)
GPU_ASYNC_DEF_READ_DETECT_REGISTER(8)
GPU_ASYNC_DEF_READ_DETECT_REGISTER(9)
GPU_ASYNC_DEF_READ_DETECT_REGISTER(10)
GPU_ASYNC_DEF_READ_DETECT_REGISTER(11)
GPU_ASYNC_DEF_READ_DETECT_REGISTER(12)
GPU_ASYNC_DEF_READ_DETECT_REGISTER(13)
GPU_ASYNC_DEF_READ_DETECT_REGISTER(14)
GPU_ASYNC_DEF_READ_DETECT_REGISTER(15)

#undef GPU_ASYNC_DEF_READ_DETECT_REGISTER

/*********************** Buffer Latencies ************************/

#define GPU_ASYNC_REG_LATENCY_BASE 1280

#define GPU_ASYNC_REG_LATENCY_TOTAL_OFFSET(_i) (GPU_ASYNC_REG_LATENCY_BASE + _i * 16 + 0)
#define GPU_ASYNC_REG_LATENCY_GPU_OFFSET(_i)   (GPU_ASYNC_REG_LATENCY_BASE + _i * 16 + 4)
#define GPU_ASYNC_REG_LATENCY_WRITE_OFFSET(_i) (GPU_ASYNC_REG_LATENCY_BASE + _i * 16 + 8)

#define GPU_ASYNC_DEF_LATENCY_REGISTER(_i) \
    GPU_ASYNC_DEF_REG(Latency ## _i ## _Total,  GPU_ASYNC_REG_LATENCY_TOTAL_OFFSET(_i), 0, 0xFFFFFFFF) \
    GPU_ASYNC_DEF_REG(Latency ## _i ## _Gpu,    GPU_ASYNC_REG_LATENCY_GPU_OFFSET(_i),   0, 0xFFFFFFFF) \
    GPU_ASYNC_DEF_REG(Latency ## _i ## _Write,  GPU_ASYNC_REG_LATENCY_WRITE_OFFSET(_i), 0, 0xFFFFFFFF)

GPU_ASYNC_DEF_LATENCY_REGISTER(0)
GPU_ASYNC_DEF_LATENCY_REGISTER(1)
GPU_ASYNC_DEF_LATENCY_REGISTER(2)
GPU_ASYNC_DEF_LATENCY_REGISTER(3)
GPU_ASYNC_DEF_LATENCY_REGISTER(4)
GPU_ASYNC_DEF_LATENCY_REGISTER(5)
GPU_ASYNC_DEF_LATENCY_REGISTER(6)
GPU_ASYNC_DEF_LATENCY_REGISTER(7)
GPU_ASYNC_DEF_LATENCY_REGISTER(8)
GPU_ASYNC_DEF_LATENCY_REGISTER(9)
GPU_ASYNC_DEF_LATENCY_REGISTER(10)
GPU_ASYNC_DEF_LATENCY_REGISTER(11)
GPU_ASYNC_DEF_LATENCY_REGISTER(12)
GPU_ASYNC_DEF_LATENCY_REGISTER(13)
GPU_ASYNC_DEF_LATENCY_REGISTER(14)
GPU_ASYNC_DEF_LATENCY_REGISTER(15)

#undef GPU_ASYNC_DEF_LATENCY_REGISTER

#undef GPU_ASYNC_DEF_REG

#endif  // __GPU_ASYNC_REGS_H__
