/**
 * ----------------------------------------------------------------------------
 * Company    : SLAC National Accelerator Laboratory
 * ----------------------------------------------------------------------------
 * Description:
 *    Provides definitions and inline functions for utilizing GPU asynchronous
 *    features within the aes_stream_drivers package.
 *
 *    This code is specifically designed for managing NVIDIA GPU memory in a
 *    Linux kernel module, offering functionality to add and remove memory
 *    regions for GPU access.
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

/**
 * GPU command codes
 **/
#define GPU_Add_Nvidia_Memory 0x8002   // Command to add NVIDIA GPU memory
#define GPU_Rem_Nvidia_Memory 0x8003   // Command to remove NVIDIA GPU memory
#define GPU_Set_Write_Enable  0x8004   // Set Write Enable Flag
#define GPU_Is_Gpu_Async_Supp 0x8005   // Check if GPU Async is supported by firmware

/**
 * struct GpuNvidiaData - Represents NVIDIA GPU memory data.
 * @write: Write permission flag (non-zero for write access).
 * @address: GPU memory address.
 * @size: Size of the memory region in bytes.
 *
 * This structure is used for managing memory regions in NVIDIA GPUs,
 * specifically for adding or removing access to these regions.
 **/
struct GpuNvidiaData {
   uint32_t write;    // Write permission flag
   uint64_t address;  // GPU memory address
   uint32_t size;     // Size of the memory region
};

#ifndef DMA_IN_KERNEL

/**
 * gpuAddNvidiaMemory - Adds a NVIDIA GPU memory region.
 * @fd: File descriptor for the device.
 * @write: Write access flag (1 for write access, 0 for read-only).
 * @address: Memory address of the GPU region to add.
 * @size: Size of the memory region to add. This must be a multiple of 64kb
 *
 * This function adds a specified memory region to the NVIDIA GPU, allowing
 * for the region to be accessed as specified by the write flag.
 *
 * Return: On success, returns the result of the ioctl call. On failure,
 * returns a negative error code. Returns -ENOTSUPP if the firmware does not
 * support GPUDirect.
 **/
static inline ssize_t gpuAddNvidiaMemory(int32_t fd, uint32_t write, uint64_t address, uint32_t size) {
   struct GpuNvidiaData dat;

   dat.write = write;
   dat.address = address;
   dat.size = size;

   return(ioctl(fd, GPU_Add_Nvidia_Memory, &dat));
}

/**
 * gpuRemNvidiaMemory - Removes a NVIDIA GPU memory region.
 * @fd: File descriptor for the device.
 *
 * This function removes a previously added memory region from the NVIDIA GPU,
 * ceasing its accessibility.
 *
 * Return: On success, returns the result of the ioctl call. On failure,
 * returns a negative error code. Returns -ENOTSUPP if the firmware does not
 * support GPUDirect.
 **/
static inline ssize_t gpuRemNvidiaMemory(int32_t fd) {
   return(ioctl(fd, GPU_Rem_Nvidia_Memory, 0));
}

/**
 * gpuSetWriteEn - Set write enable for buffer
 * @dev: pointer to the DMA device structure
 * @arg: user space argument pointing to buffer index
 *
 * This function enables a DMA buffer for DMA operations.
 *
 * Return: 0 on success, negative error code on failure.
 * Returns -ENOTSUPP if the firmware does not support GPUDirect.
 */
static inline ssize_t gpuSetWriteEn(int32_t fd, uint32_t idx) {
   uint32_t lidx = idx;
   return(ioctl(fd, GPU_Set_Write_Enable, &lidx));
}

/**
 * gpuIsGpuAsyncSupported - Check if the firmware supports GPU Async
 * @fd: File descriptor for the device
 *
 * Return: 1 if the firmware supports GPU Async, 0 if it doesn't.
 * Returns -EINVAL if the driver was compiled without GPUAsync support
 */
static inline bool gpuIsGpuAsyncSupported(int32_t fd) {
   return ioctl(fd, GPU_Is_Gpu_Async_Supp);
}

#endif  // !DMA_IN_KERNEL
#endif  // __GPU_ASYNC_H__
