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
#define GPU_Get_Gpu_Async_Ver 0x8006   // Get the GpuAsyncCore version
#define GPU_Get_Max_Buffers   0x8007   // Get the max number of DMA buffers
#define GPU_Enable_Tx         0x8008   // Enable tx buffers (FPGA -> GPU)
#define GPU_Enable_Rx         0x8009   // Enable rx buffers (GPU -> FPGA)

/**
 * @brief Represents NVIDIA GPU memory data.
 *
 * This structure is used for managing memory regions in NVIDIA GPUs,
 * specifically for adding or removing access to these regions.
 **/
struct GpuNvidiaData {
   uint32_t write;    /**< Write permission flag (non-zero for write access). */
   uint64_t address;  /**< GPU memory address. */
   uint32_t size;     /**< Size of the memory region in bytes. */
};

#ifndef DMA_IN_KERNEL

/**
 * @brief Adds an NVIDIA GPU memory region.
 *
 * This function adds a specified memory region to the NVIDIA GPU, allowing
 * for the region to be accessed as specified by the write flag.
 *
 * For GpuAsyncCore V4 and later, this function will clear the "write enable"
 * and "read enable" bits. Userspace must call gpuEnableRx/gpuEnableTx as needed
 * after adding memory with this function.
 *
 * GpuAsyncCore state is exclusively owned by one process at a time. These IOCTLs
 * will fail with EBUSY if GpuAsyncCore state is already being managed by another
 * process.
 *
 * @param fd       File descriptor for the device.
 * @param write    Write access flag (1 for write access, 0 for read-only).
 * @param address  Memory address of the GPU region to add.
 * @param size     Size of the memory region to add.  Must be a multiple of 64 KB.
 *
 * @return On success, returns the result of the ioctl call.  On failure,
 *         returns a negative error code and sets errno:
 *          * ENOTSUPP if the firmware does not support GPUDirect.
 *          * EBUSY if the GpuAsyncCore state is locked by another process.
 **/
static inline ssize_t gpuAddNvidiaMemory(int32_t fd, uint32_t write, uint64_t address, uint32_t size) {
   struct GpuNvidiaData dat;

   dat.write = write;
   dat.address = address;
   dat.size = size;

   return(ioctl(fd, GPU_Add_Nvidia_Memory, &dat));
}

/**
 * @brief Removes an NVIDIA GPU memory region.
 *
 * This function removes a previously added memory region from the NVIDIA GPU,
 * ceasing its accessibility.
 *
 * @param fd File descriptor for the device.
 *
 * @return On success, returns the result of the ioctl call.  On failure,
 *         returns a negative error code and sets errno:
 *          * ENOTSUPP if the firmware does not support GPUDirect.
 *          * EBUSY if the GpuAsyncCore state is locked by another process.
 **/
static inline ssize_t gpuRemNvidiaMemory(int32_t fd) {
   return(ioctl(fd, GPU_Rem_Nvidia_Memory, 0));
}

/**
 * @brief Set write enable for buffer.
 *
 * This function enables a DMA buffer for DMA operations.
 *
 * @param fd  File descriptor for the device.
 * @param idx Buffer index to enable.
 *
 * @return On success, returns 0. On failure, returns a negative error code and
 *         sets errno:
 *          * ENOTSUPP if the firmware does not support GPUDirect.
 *          * EBUSY if the GpuAsyncCore state is locked by another process.
 */
static inline ssize_t gpuSetWriteEn(int32_t fd, uint32_t idx) {
   uint32_t lidx = idx;
   return(ioctl(fd, GPU_Set_Write_Enable, &lidx));
}

/**
 * @brief Check if the firmware supports GPU Async.
 *
 * @param fd File descriptor for the device.
 *
 * @return @c true if the firmware and driver support GPU Async, @c false
 *         otherwise (including when the driver was built without GPUAsync
 *         support and returns @c -ENOTSUPP, or when the ioctl itself fails
 *         with @c -1 and @c errno set, e.g. @c ENOTTY or @c ENOTSUPP).
 *         Callers should treat a @c false return as "not supported" without
 *         attempting any further GPU Async ioctls, except for gpuGetGpuAsyncVersion.
 *
 * @note The ioctl returns @c 1 (supported), @c 0 (not supported), or a
 *       negative errno on failure.  We must compare against @c >0 because
 *       casting @c -1 directly to @c bool yields @c true and would produce
 *       a false-positive "supported" result on driver/fd errors.
 */
static inline bool gpuIsGpuAsyncSupported(int32_t fd) {
   return ioctl(fd, GPU_Is_Gpu_Async_Supp) > 0;
}

/**
 * @brief Get the version of GpuAsyncCore in the firmware.
 *
 * @param fd File descriptor for the device.
 *
 * @return On success, the version of GpuAsyncCore (@c >= 0).  On failure,
 *         @c -1 with @c errno set to indicate the cause.
 *         @c ENOTSUPP or @c ENOTTY indicates that the driver was compiled without
 *         GpuAsync support.
 *
 * @note The return type is @c ssize_t (signed) so the @c -1 failure case is
 *       representable without wraparound.  Callers must explicitly check
 *       for a negative return before using the value.
 */
static inline ssize_t gpuGetGpuAsyncVersion(int32_t fd) {
   return ioctl(fd, GPU_Get_Gpu_Async_Ver);
}

/**
 * @brief Get the maximum number of DMA buffers.
 *
 * @param fd File descriptor for the device.
 *
 * @return On success, the number of DMA buffers available for use
 *         (@c >= 0).  On failure, @c -1 with @c errno set to indicate the
 *         cause (for example @c ENOTSUPP or @c ENOTTY if the driver was
 *         compiled without GPUAsync support, or another ioctl error).
 *
 * @note Same signed-return rationale as gpuGetGpuAsyncVersion() — callers
 *       must explicitly check for a negative return before using the value.
 */
static inline ssize_t gpuGetMaxBuffers(int32_t fd) {
   return ioctl(fd, GPU_Get_Max_Buffers);
}

/**
 * @brief Enables TX buffers, flowing from FPGA -> GPU
 *
 * @param fd File descriptor for the device.
 * @param enable Disable tx if 0, enable tx if != 0
 *
 * @return On success, returns 0. On failure, returns -1 the sets errno:
 *          * ENOTSUPP if the firmware does not support GPUDirect.
 *          * EBUSY if the GpuAsyncCore state is locked by another process.
 */
static inline int32_t gpuEnableTx(int32_t fd, uint32_t enable) {
   return ioctl(fd, GPU_Enable_Tx, enable);
}

/**
 * @brief Enable RX buffers, flowing from GPU -> FPGA
 *
 * @param fd File descriptor for the device.
 * @param enable Disable rx if 0, enable rx if != 0
 *
 * @return On success, returns 0. On failure, returns -1 and sets errno:
 *          * ENOTSUPP if the firmware does not support GPUDirect.
 *          * EBUSY if the GpuAsyncCore state is locked by another process.
 */
static inline int32_t gpuEnableRx(int32_t fd, uint32_t enable) {
   return ioctl(fd, GPU_Enable_Rx, enable);
}

#endif  // !DMA_IN_KERNEL
#endif  // __GPU_ASYNC_H__
