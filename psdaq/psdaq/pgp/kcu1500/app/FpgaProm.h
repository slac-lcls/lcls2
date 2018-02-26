/**
 *-----------------------------------------------------------------------------
 * Title      : FPGA Prom Driver
 * ----------------------------------------------------------------------------
 * File       : FpgaProm.h
 * Created    : 2016-08-08
 * ----------------------------------------------------------------------------
 * Description:
 * FPGA Prom Data
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
#ifndef __FPGA_PROM_H__
#define __FPGA_PROM_H__

#ifdef DMA_IN_KERNEL
#include <linux/types.h>
#else
#include <stdint.h>
#endif

// Commands
#define FPGA_Write_Prom  0x1100
#define FPGA_Read_Prom   0x1101

// Prom Programming 
struct FpgaPromData {
   uint32_t address;
   uint32_t cmd;
   uint32_t data;
   uint32_t pad;
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

// Write to PROM
static inline ssize_t fpgaWriteProm(int32_t fd, uint32_t address, uint32_t cmd, uint32_t data) {
   struct FpgaPromData prom;

   prom.address = address;
   prom.cmd     = cmd;
   prom.data    = data;
   return(ioctl(fd,FPGA_Write_Prom,&prom));
}

// Read from PROM
static inline ssize_t fpgaReadProm(int32_t fd, uint32_t address, uint32_t cmd, uint32_t *data) {
   struct FpgaPromData prom;
   ssize_t res;

   prom.address = address;
   prom.cmd     = cmd;
   prom.data    = 0;
   res = ioctl(fd,FPGA_Read_Prom,&prom);

   if ( data != NULL ) *data = prom.data;

   return(res);
}

#endif
#endif

