/**
 *-----------------------------------------------------------------------------
 * Title      : AXI Version 
 * ----------------------------------------------------------------------------
 * File       : AxiVersion.h
 * Created    : 2017-03-17
 * ----------------------------------------------------------------------------
 * Description:
 * AXI Version Record
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
#ifndef __AXI_VERSION_H__
#define __AXI_VERSION_H__

#ifdef DMA_IN_KERNEL
#include <linux/types.h>
#else
#include <stdint.h>
#endif

// Commands
#define AVER_Get 0x1200

// AXI Version Data
struct AxiVersion {
   uint32_t firmwareVersion;
   uint32_t scratchPad;
   uint32_t upTimeCount;
   uint64_t fdValue;
   uint32_t userValues[64];
   uint32_t deviceId;
   uint8_t  gitHash[160];
   uint8_t  dnaValue[16];
   uint8_t  buildString[256];
};

// Everything below is hidden during kernel module compile
#ifndef DMA_IN_KERNEL
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/signal.h>
#include <sys/fcntl.h>

// Read AXI Version
static inline ssize_t axiVersionGet(int32_t fd, struct AxiVersion * aVer ) {
   return(ioctl(fd,AVER_Get,aVer));
}

#endif
#endif

