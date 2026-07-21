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

// Size of AxiPcieGpuAsyncCore. Must match the value within the firmware
#define GPU_ASYNC_CORE_SIZE       0x00008000  // 0x0003_0000 - 0x0002_8000

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
   volatile uint32_t* regp = (volatile uint32_t*)(((volatile uint8_t*)baseptr) + reg->offset);
#ifdef DMA_IN_KERNEL
   writel((readl(regp) & ~reg->bitMask) | ((value << reg->bitOffset) & reg->bitMask), regp);
#else
   *regp = (*regp & ~reg->bitMask) | ((value << reg->bitOffset) & reg->bitMask);
#endif
}

#define GPU_ASYNC_DEF_REG(_name, _off, _bitOff, _bitMask)           \
static const struct GpuAsyncRegister GpuAsyncReg_ ## _name = {      \
   .offset = (_off),                                               \
   .bitOffset = (_bitOff),                                         \
   .bitMask = (_bitMask)                                           \
};

// V4 Configuration
GPU_ASYNC_DEF_REG(MaxBuffersV4,   0x0, 0, 0x7FF);

GPU_ASYNC_DEF_REG(ArCache,      0x4, 0,  0xFF);
GPU_ASYNC_DEF_REG(AwCache,      0x4, 8,  0xFF00);
GPU_ASYNC_DEF_REG(DmaDataBytes, 0x4, 16, 0xFF0000);
GPU_ASYNC_DEF_REG(MaxBuffersV1, 0x4, 24, 0x1F000000);  // V1 Configuration

// V1 Configuration
GPU_ASYNC_DEF_REG(WriteCountV1,   0x8, 0,  0xFF);
GPU_ASYNC_DEF_REG(WriteEnableV1,  0x8, 8,  0xFF00);
GPU_ASYNC_DEF_REG(ReadCountV1,    0x8, 16, 0xFF0000);
GPU_ASYNC_DEF_REG(ReadEnableV1,   0x8, 24, 0xFF000000);

// V4 Configuration
GPU_ASYNC_DEF_REG(WriteCountV4,   0x8, 0,  0x7FFF);
GPU_ASYNC_DEF_REG(WriteEnableV4,  0x8, 15,  0x8000);
GPU_ASYNC_DEF_REG(ReadCountV4,    0x8, 16, 0x7FFF0000);
GPU_ASYNC_DEF_REG(ReadEnableV4,   0x8, 31, 0x80000000);

GPU_ASYNC_DEF_REG(RxFrameCnt,       0x10, 0, 0xFFFFFFFF);
GPU_ASYNC_DEF_REG(TxFrameCnt,       0x14, 0, 0xFFFFFFFF);
GPU_ASYNC_DEF_REG(AxiWriteErrorCnt, 0x18, 0, 0xFFFFFFFF);
GPU_ASYNC_DEF_REG(AxiReadErrorCnt,  0x1C, 0, 0xFFFFFFFF);

GPU_ASYNC_DEF_REG(CntRst,           0x20, 0, 0xFFFFFFFF);
GPU_ASYNC_DEF_REG(AxiWriteErrorVal, 0x24, 0, 0xFFFFFFFF);
GPU_ASYNC_DEF_REG(AxiReadErrorVal,  0x28, 0, 0xFFFFFFFF);

// V1 Configuration -- Removed in V4
GPU_ASYNC_DEF_REG(DynamicRouteMasks0V1, 0x2C, 0,  0xFF);
GPU_ASYNC_DEF_REG(DynamicRouteDests0V1, 0x2C, 8,  0xFF00);
GPU_ASYNC_DEF_REG(DynamicRouteMasks1V1, 0x2C, 16, 0xFF0000);
GPU_ASYNC_DEF_REG(DynamicRouteDests1V1, 0x2C, 24, 0xFF000000);

GPU_ASYNC_DEF_REG(Version, 0x30, 0, 0xFF);

GPU_ASYNC_DEF_REG(AxiWriteTimeoutCnt, 0x34, 0, 0xFFFFFFFF);

GPU_ASYNC_DEF_REG(AxisDeMuxSelect, 0x38, 0, 0x1);

GPU_ASYNC_DEF_REG(MinWriteBuffer, 0x3C, 0, 0xFFFFFFFF);
GPU_ASYNC_DEF_REG(MinReadBuffer,  0x40, 0, 0xFFFFFFFF);

// V4 Configuration
GPU_ASYNC_DEF_REG(TotLatencyV4, 0x48, 0, 0xFFFFFFFF);
GPU_ASYNC_DEF_REG(GpuLatencyV4, 0x50, 0, 0xFFFFFFFF);
GPU_ASYNC_DEF_REG(WrLatencyV4,  0x58, 0, 0xFFFFFFFF);

GPU_ASYNC_DEF_REG(RemoteWriteMaxSizeV4, 0x60, 0, 0xFFFFFFFF);


// The following register defintiions are firmware specific. GpuAsyncCore can have up to 16 buffers, but defaults to 4.
// You must check the MaxBuffers register for the true value

/*********************** Write Buffers ************************/

#define GPU_ASYNC_REG_WRITE_BASE_V1 0x0100
#define GPU_ASYNC_REG_WRITE_BASE_V4 0x4000

// V1 Configuration
#define GPU_ASYNC_REG_WRITE_ADDR_L_OFFSET_V1(_i) (GPU_ASYNC_REG_WRITE_BASE_V1 + (_i) * 16 + 0)
#define GPU_ASYNC_REG_WRITE_ADDR_H_OFFSET_V1(_i) (GPU_ASYNC_REG_WRITE_BASE_V1 + (_i) * 16 + 4)
#define GPU_ASYNC_REG_WRITE_SIZE_OFFSET_V1(_i) (GPU_ASYNC_REG_WRITE_BASE_V1 + (_i) * 16 + 8)

// V4 Configuration. Write size has been removed and migrated to a separate register RemoteWriteMaxSize
#define GPU_ASYNC_REG_WRITE_ADDR_L_OFFSET_V4(_i) (GPU_ASYNC_REG_WRITE_BASE_V4 + (_i) * 8 + 0)
#define GPU_ASYNC_REG_WRITE_ADDR_H_OFFSET_V4(_i) (GPU_ASYNC_REG_WRITE_BASE_V4 + (_i) * 8 + 4)

/*********************** Read Buffers ************************/

#define GPU_ASYNC_REG_READ_BASE_V1 0x0200
#define GPU_ASYNC_REG_READ_BASE_V4 0x6000

// V1 Configuration
#define GPU_ASYNC_REG_READ_ADDR_L_OFFSET_V1(_i) (GPU_ASYNC_REG_READ_BASE_V1 + (_i) * 16 + 0)
#define GPU_ASYNC_REG_READ_ADDR_H_OFFSET_V1(_i) (GPU_ASYNC_REG_READ_BASE_V1 + (_i) * 16 + 4)

// V4 Configuration
#define GPU_ASYNC_REG_READ_ADDR_L_OFFSET_V4(_i) (GPU_ASYNC_REG_READ_BASE_V4 + (_i) * 8 + 0)
#define GPU_ASYNC_REG_READ_ADDR_H_OFFSET_V4(_i) (GPU_ASYNC_REG_READ_BASE_V4 + (_i) * 8 + 4)

#define GPU_ASYNC_REG_READ_ADDR_L_OFFSET(_version, _i) \
   (((_version) < 4) ? GPU_ASYNC_REG_READ_ADDR_L_OFFSET_V1(_i) : GPU_ASYNC_REG_READ_ADDR_L_OFFSET_V4(_i))
#define GPU_ASYNC_REG_READ_ADDR_H_OFFSET(_version, _i) \
   (((_version) < 4) ? GPU_ASYNC_REG_READ_ADDR_H_OFFSET_V1(_i) : GPU_ASYNC_REG_READ_ADDR_H_OFFSET_V4(_i))

/*********************** Write Detect ************************/

#define GPU_ASYNC_REG_WRITE_DETECT_BASE_V1 0x0300
#define GPU_ASYNC_REG_WRITE_DETECT_BASE_V4 0x2000

#define GPU_ASYNC_REG_WRITE_DETECT_OFFSET_V1(_i) (GPU_ASYNC_REG_WRITE_DETECT_BASE_V1 + (_i) * 4)
#define GPU_ASYNC_REG_WRITE_DETECT_OFFSET_V4(_i) (GPU_ASYNC_REG_WRITE_DETECT_BASE_V4 + (_i) * 4)

/*********************** Read Detect ************************/

#define GPU_ASYNC_REG_READ_DETECT_BASE_V1 0x0400
#define GPU_ASYNC_REG_READ_DETECT_BASE_V4 0x3000

#define GPU_ASYNC_REG_REMOTE_READ_SIZE_OFFSET_V1(_i) (GPU_ASYNC_REG_READ_DETECT_BASE_V1 + (_i) * 4 + 0)
#define GPU_ASYNC_REG_REMOTE_READ_SIZE_OFFSET_V4(_i) (GPU_ASYNC_REG_READ_DETECT_BASE_V4 + (_i) * 4 + 0)

#define GPU_ASYNC_REG_REMOTE_READ_SIZE_OFFSET(_ver, _i) \
   (((_ver) == 4) ? GPU_ASYNC_REG_REMOTE_READ_SIZE_OFFSET_V4(_i) : GPU_ASYNC_REG_REMOTE_READ_SIZE_OFFSET_V1(_i))

/*
========================================
Buffer latencies -- Consolidated in V4
========================================
*/

#define GPU_ASYNC_REG_LATENCY_BASE_V1 1280

#define GPU_ASYNC_REG_LATENCY_TOTAL_OFFSET_V1(_i) (GPU_ASYNC_REG_LATENCY_BASE_V1 + (_i) * 16 + 0)
#define GPU_ASYNC_REG_LATENCY_GPU_OFFSET_V1(_i)   (GPU_ASYNC_REG_LATENCY_BASE_V1 + (_i) * 16 + 4)
#define GPU_ASYNC_REG_LATENCY_WRITE_OFFSET_V1(_i) (GPU_ASYNC_REG_LATENCY_BASE_V1 + (_i) * 16 + 8)

#undef GPU_ASYNC_DEF_REG

#endif  // __GPU_ASYNC_REGS_H__
