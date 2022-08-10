/**
 *-----------------------------------------------------------------------------
 * Title      : TEM Card Driver, Shared Header
 * ----------------------------------------------------------------------------
 * File       : TemDriver.h
 * Author     : Ryan Herbst, rherbst@slac.stanford.edu
 * Created    : 2016-08-08
 * Last update: 2016-08-08
 * ----------------------------------------------------------------------------
 * Description:
 * Defintions and inline functions for interacting with TEM driver.
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
#ifndef __TEM_DRIVER_H__
#define __TEM_DRIVER_H__
#include <DmaDriver.h>
#include <FpgaProm.h>

// Card Info
struct TemInfo {
   uint64_t serial;
   uint32_t version;
   uint32_t promPrgEn;
   char     buildStamp[256];
};

// PCI Info
struct PciStatus {
   uint32_t pciCommand;
   uint32_t pciStatus;
   uint32_t pciDCommand;
   uint32_t pciDStatus;
   uint32_t pciLCommand;
   uint32_t pciLStatus;
   uint32_t pciLinkState;
   uint32_t pciFunction;
   uint32_t pciDevice;
   uint32_t pciBus;
   uint32_t pciLanes;
   uint32_t pad;
};

// Error values
#define TEM_ERR_EOFE 0x10

// Commands
#define TEM_Read_Info   0x2001
#define TEM_Read_Pci    0x2002
#define TEM_Set_Loop    0x2004
#define TEM_Count_Reset 0x2005

// Destination
#define TEM_DEST_CMD  0
#define TEM_DEST_DATA 1

// Everything below is hidden during kernel module compile
#ifndef DMA_IN_KERNEL

// Enable command reads, call only once
static inline int32_t temEnableCmdRead(int32_t fd) {
   return(dmaSetMask(fd,1 << TEM_DEST_CMD));
}

// Enable data reads, call only once
static inline int32_t temEnableDataRead(int32_t fd) {
   return(dmaSetMask(fd,1 << TEM_DEST_DATA));
}

// Write to TEM command channel
static inline ssize_t temWriteCmd(int32_t fd, const void *buf, size_t count) {
   return(dmaWrite(fd,buf,count,0,TEM_DEST_CMD));
}

// Write to TEM data channel
static inline ssize_t temWriteData(int32_t fd, const void *buf, size_t count) {
   return(dmaWrite(fd,buf,count,0,TEM_DEST_DATA));
}

// Read from TEM channel
static inline ssize_t temRead(int fd, void *buf, size_t count) {
   uint32_t error;
   uint32_t ret;
   ret = dmaRead(fd,buf,count,NULL,&error,NULL);
   return(error==0?ret:-1);
}

// Read Card Info
static inline ssize_t temGetInfo(int32_t fd, struct TemInfo * info) {
   return(ioctl(fd,TEM_Read_Info,info));
}

// Read PCI Status
static inline ssize_t temGetPci(int32_t fd, struct PciStatus * status) {
   return(ioctl(fd,TEM_Read_Pci,status));
}

// Set Loopback State
static inline ssize_t temSetLoop(int32_t fd, uint32_t state) {
   uint32_t temp;

   temp = 0x3;
   temp |= ((state << 8) & 0x100);

   return(ioctl(fd,TEM_Set_Loop,temp));
}

#endif
#endif

