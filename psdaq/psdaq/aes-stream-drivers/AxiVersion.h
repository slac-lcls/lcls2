/**
 * ----------------------------------------------------------------------------
 * Company    : SLAC National Accelerator Laboratory
 * ----------------------------------------------------------------------------
 * Description:
 *    Defines an interface for accessing AXI version information in kernel space.
 *-----------------------------------------------------------------------------
 * This file is part of the aes_stream_drivers package. It is subject to the
 * license terms in the LICENSE.txt file found in the top-level directory of
 * this distribution and at:
 *    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
 * No part of the aes_stream_drivers package, including this file, may be
 * copied, modified, propagated, or distributed except according to the terms
 * contained in the LICENSE.txt file.
 *-----------------------------------------------------------------------------
 **/

#ifndef __AXI_VERSION_H__
#define __AXI_VERSION_H__

#ifdef DMA_IN_KERNEL
   #include <linux/types.h>
#else
   #include <stdint.h>
#endif

/**
 * Commands for AXI version operations.
 */
#define AVER_Get 0x1200

/**
 * struct AxiVersion - Represents AXI version data.
 * @firmwareVersion: Firmware version number.
 * @scratchPad: General purpose scratch pad register.
 * @upTimeCount: Counter for the uptime in ticks.
 * @fdValue: Factory default values.
 * @userValues: User-defined values for customization.
 * @deviceId: Unique identifier for the device.
 * @gitHash: Hash of the git commit for software tracking.
 * @dnaValue: Device DNA value for identification.
 * @buildString: String containing build information.
 *
 * This structure is used to hold version information and metadata related to
 * the AXI interface, including both software and hardware identifiers.
 */
struct AxiVersion {
   uint32_t firmwareVersion;
   uint32_t scratchPad;
   uint32_t upTimeCount;
   uint8_t  fdValue[8];
   uint32_t userValues[64];
   uint32_t deviceId;
   uint8_t  gitHash[160];
   uint8_t  dnaValue[16];
   uint8_t  buildString[256];
};

#ifndef DMA_IN_KERNEL
   // Everything below is hidden during kernel module compile
   #include <stdlib.h>
   #include <string.h>//NOLINT
   #include <sys/mman.h>
   #include <stdio.h>
   #include <unistd.h>
   #include <sys/ioctl.h>
   #include <sys/signal.h>
   #include <sys/fcntl.h>

   /**
    * axiVersionGet - Reads the AXI version information.
    * @fd: File descriptor for the device.
    * @aVer: Pointer to AxiVersion structure to fill with data.
    *
    * This function invokes an IOCTL call to read the AXI version information
    * from the hardware and fills the provided AxiVersion structure with the
    * data read.
    *
    * Return: The number of bytes read on success or an error code on failure.
    */
   static inline ssize_t axiVersionGet(int32_t fd, struct AxiVersion *aVer) {
      return(ioctl(fd, AVER_Get, aVer));
   }

#endif  // !DMA_IN_KERNEL
#endif  // __AXI_VERSION_H__
