/**
 *-----------------------------------------------------------------------------
 * Title      : GPU Async Header
 * ----------------------------------------------------------------------------
 * File       : GpuAsync.h
 * ----------------------------------------------------------------------------
 * Description:
 * Defintions and inline functions for using GPU Async features.
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
#ifndef __GPU_ASYNC_H__
#define __GPU_ASYNC_H__
#include "DmaDriver.h"

// Commands
#define GPU_Add_Nvidia_Memory 0x8002
#define GPU_Rem_Nvidia_Memory 0x8003

// NVidia Data
struct GpuNvidiaData {
   uint32_t   write;
   uint64_t   address;
   uint32_t   size;
};

// Everything below is hidden during kernel module compile
#ifndef DMA_IN_KERNEL

// Add NVIDIA Memory
static inline ssize_t gpuAddNvidiaMemory(int32_t fd, uint32_t write, uint64_t address, uint32_t size) {
   struct GpuNvidiaData dat;

   dat.write    = write;
   dat.address  = address;
   dat.size     = size;

   return(ioctl(fd,GPU_Add_Nvidia_Memory,&dat));
}

// Rem NVIDIA Memory
static inline ssize_t gpuRemNvidiaMemory(int32_t fd) {
   return(ioctl(fd,GPU_Rem_Nvidia_Memory,0));
}

#endif
#endif

