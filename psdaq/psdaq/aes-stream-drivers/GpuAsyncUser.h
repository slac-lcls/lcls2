/**
 * ----------------------------------------------------------------------------
 * Company    : SLAC National Accelerator Laboratory
 * ----------------------------------------------------------------------------
 * Description: User space API for Gpu Async support. Attempts to abstract away
 *  some of the internal implementation detail from user-space software.
 *  This file contains no handling for big-endian systems.
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
#ifndef _GPU_ASYNC_USER_H_
#define _GPU_ASYNC_USER_H_

#include "GpuAsyncRegs.h"

#if defined(__cplusplus) && __cplusplus < 201103L
#error The code in this file requires C++11
#endif

#ifdef __cplusplus

/**
 * @brief Thin wrapper around the C API and definitions in GpuAsyncRegs.h
 * The lifetime of this object must be within the lifetime of the memory mapped registers provided in the constructor.
 *
 * Code calling into this object to retrieve register values does not need to be aware of the offsets or GpuAsyncCore version.
 */
class GpuAsyncCoreRegs {
public:
   GpuAsyncCoreRegs() = delete;

   /**
    * @brief regs Pointer to the memory mapped GpuAsyncCore registers.
    */
   explicit GpuAsyncCoreRegs(volatile void* regs) :
      regs_((volatile uint8_t*)regs) {
      this->version_ = readReg(GpuAsyncReg_Version);
   }

   inline volatile uint8_t* registers() const {
      return (volatile uint8_t*)this->regs_;
   }

   inline uint32_t readReg(const GpuAsyncRegister& reg) const {
      return readGpuAsyncReg(regs_, &reg);
   }

   /**
    * @brief Read register at a specific offset, instead of using the GpuAsyncRegister struct
    */
   inline uint32_t readReg(uint32_t offset) const {
      return *(uint32_t*)(regs_ + offset);
   }

   inline void writeReg(const GpuAsyncRegister& reg, uint32_t value) {
      writeGpuAsyncReg(regs_, &reg, value);
   }

   inline void writeReg(uint32_t offset, uint32_t value) {
      *(uint32_t*)(regs_ + offset) = value;
   }

   /**
    * @brief Returns the version of GpuAsyncCore this firmware is running
    */
   inline uint32_t version() const { return version_; }

   /**
    * @brief Returns the max number of buffers supported by the firmware
    */
   uint32_t maxBuffers() const {
      return readRegV1V4(GpuAsyncReg_MaxBuffersV1, GpuAsyncReg_MaxBuffersV4);
   }

   uint32_t arCache() const {
      return readReg(GpuAsyncReg_ArCache);
   }

   uint32_t awCache() const {
      return readReg(GpuAsyncReg_AwCache);
   }

   /**
    * @brief Returns the number of dma header bytes, DMA_AXI_CONFIG_G.DATA_BYTES_C
    */
   uint32_t dmaDataBytes() const {
      return readReg(GpuAsyncReg_DmaDataBytes);
   }

   inline uint32_t writeCount() const {
      return readRegV1V4(GpuAsyncReg_WriteCountV1, GpuAsyncReg_WriteCountV4);
   }

   inline void setWriteCount(uint32_t val) {
      writeRegV1V4(GpuAsyncReg_WriteCountV1, GpuAsyncReg_WriteCountV4, val);
   }

   inline uint32_t writeEnable() const {
      return readRegV1V4(GpuAsyncReg_WriteEnableV1, GpuAsyncReg_WriteEnableV4);
   }

   inline void setWriteEnable(uint32_t val) {
      writeRegV1V4(GpuAsyncReg_WriteEnableV1, GpuAsyncReg_WriteEnableV4, val);
   }

   inline uint32_t readCount() const {
      return readRegV1V4(GpuAsyncReg_ReadCountV1, GpuAsyncReg_ReadCountV4);
   }

   inline void setReadCount(uint32_t val) {
      writeRegV1V4(GpuAsyncReg_ReadCountV1, GpuAsyncReg_ReadCountV4, val);
   }

   inline uint32_t readEnable() const {
      return readRegV1V4(GpuAsyncReg_ReadEnableV1, GpuAsyncReg_ReadEnableV4);
   }

   inline void setReadEnable(uint32_t val) {
      writeRegV1V4(GpuAsyncReg_ReadEnableV1, GpuAsyncReg_ReadEnableV4, val);
   }

   void countReset() {
      writeReg(GpuAsyncReg_CntRst, 1);
   }

   inline uint32_t rxFrameCount() const {
      return readReg(GpuAsyncReg_RxFrameCnt);
   }

   inline uint32_t txFrameCount() const {
      return readReg(GpuAsyncReg_TxFrameCnt);
   }

   inline uint32_t axiWriteErrorCount() const {
      return readReg(GpuAsyncReg_AxiWriteErrorCnt);
   }

   inline uint32_t axiReadErrorCount() const {
      return readReg(GpuAsyncReg_AxiReadErrorCnt);
   }

   inline uint32_t axiWriteErrorVal() const {
      return readReg(GpuAsyncReg_AxiWriteErrorVal);
   }

   inline uint32_t axiReadErrorVal() const {
      return readReg(GpuAsyncReg_AxiReadErrorVal);
   }

   inline uint32_t axiWriteTimeoutCount() const {
      return readReg(GpuAsyncReg_AxiWriteTimeoutCnt);
   }

   inline uint32_t axisDeMuxSelect() const {
      return readReg(GpuAsyncReg_AxisDeMuxSelect);
   }

   inline void setAxisDeMuxSelect(uint32_t val) {
      writeReg(GpuAsyncReg_AxisDeMuxSelect, val);
   }

   inline uint32_t minWriteBuffer() const {
      return readReg(GpuAsyncReg_MinWriteBuffer);
   }

   inline uint32_t minReadBuffer() const {
      return readReg(GpuAsyncReg_MinReadBuffer);
   }

   /**
    * @brief Returns the total round-trip latency, in clock cycles, reported for the buffer
    * @note For V4+, the buffer argument is ignored and should be 0.
    */
   inline uint32_t totalLatency(uint32_t buffer) const {
      switch (versionSwitch()) {
      case 0:
         return 0;
      case 1:
         return readReg(GPU_ASYNC_REG_LATENCY_TOTAL_OFFSET_V1(buffer));
      case 4:
      default:
         return readReg(GpuAsyncReg_TotLatencyV4);
      }
   }

   /**
    * @brief Returns the GPU processing latency, in clock cycles, reported for the buffer
    * @note For V4+, the buffer argument is ignored and should be 0.
    */
   inline uint32_t gpuLatency(uint32_t buffer) const {
      switch (versionSwitch()) {
      case 0:
         return 0;
      case 1:
         return readReg(GPU_ASYNC_REG_LATENCY_GPU_OFFSET_V1(buffer));
      case 4:
      default:
         return readReg(GpuAsyncReg_GpuLatencyV4);
      }
   }

   /**
    * @brief Returns the FPGA -> GPU write latency, in clock cycles, reported for the buffer
    * @note For V4+, the buffer argument is ignored and should be 0.
    */
   inline uint32_t writeLatency(uint32_t buffer) const {
      switch (versionSwitch()) {
      case 0:
         return 0;
      case 1:
         return readReg(GPU_ASYNC_REG_LATENCY_WRITE_OFFSET_V1(buffer));
      case 4:
      default:
         return readReg(GpuAsyncReg_WrLatencyV4);
      }
   }

   /**
    * @brief Gets the remote write max size, used for FPGA -> GPU transfers
    * @param buffer The buffer to set the remote size for. Ignored in version >= 4
    * @note buffer is ignored when version() >= 4, since in V4 all buffers share the same register
    */
   inline uint32_t remoteWriteSize(uint32_t buffer) const {
      switch (versionSwitch()) {
      case 0:
         return 0;
      case 1:
         return readReg(GPU_ASYNC_REG_WRITE_SIZE_OFFSET_V1(buffer));
      case 4:
      default:
         return readReg(GpuAsyncReg_RemoteWriteMaxSizeV4);
      }
   }

   /**
    * @brief Sets the remote write max size, used for FPGA -> GPU transfers
    * @param buffer The buffer to set the remote size for. Ignored in version >= 4
    * @param size The size
    * @note buffer is ignored when version() >= 4, since in V4 all buffers share the same register
    */
   inline void setRemoteWriteMaxSize(uint32_t buffer, uint32_t size) {
      switch (versionSwitch()) {
      case 0:
         return;
      case 1:
         writeReg(GPU_ASYNC_REG_WRITE_SIZE_OFFSET_V1(buffer), size);
         return;
      case 4:
      default:
         writeReg(GpuAsyncReg_RemoteWriteMaxSizeV4, size);
         return;
      }
   }

   /**
    * @brief Sets the remote write address for the specified buffer. Used for FPGA -> GPU transfers
    * @param buffer The buffer index. Must be < 16 for V1, and < 1024 for V4
    * @param addr 64-bit address in GPU device memory
    */
   inline void setRemoteWriteAddress(uint32_t buffer, uint64_t addr) {
      uint32_t l = uint32_t(addr & 0xFFFFFFFF);
      uint32_t h = uint32_t((addr >> 32) & 0xFFFFFFFF);

      switch (versionSwitch()) {
      case 0:
         return;
      case 1:
         writeReg(GPU_ASYNC_REG_WRITE_ADDR_L_OFFSET_V1(buffer), l);
         writeReg(GPU_ASYNC_REG_WRITE_ADDR_H_OFFSET_V1(buffer), h);
         break;
      case 4:
      default:
         writeReg(GPU_ASYNC_REG_WRITE_ADDR_L_OFFSET_V4(buffer), l);
         writeReg(GPU_ASYNC_REG_WRITE_ADDR_H_OFFSET_V4(buffer), h);
         break;
      }
   }

   /**
    * @brief Sets the remote read address for the specified buffer. Used for GPU -> FPGA transfers
    * @param buffer The buffer index. Must be < 16 for V1, and < 1024 for V4
    * @param addr 64-bit address in GPU device memory
    */
   inline void setRemoteReadAddress(uint32_t buffer, uint64_t addr) {
      uint32_t l = uint32_t(addr & 0xFFFFFFFF);
      uint32_t h = uint32_t((addr >> 32) & 0xFFFFFFFF);

      switch (versionSwitch()) {
      case 0:
         return;
      case 1:
         writeReg(GPU_ASYNC_REG_READ_ADDR_L_OFFSET_V1(buffer), l);
         writeReg(GPU_ASYNC_REG_READ_ADDR_H_OFFSET_V1(buffer), h);
         break;
      case 4:
      default:
         writeReg(GPU_ASYNC_REG_READ_ADDR_L_OFFSET_V4(buffer), l);
         writeReg(GPU_ASYNC_REG_READ_ADDR_H_OFFSET_V4(buffer), h);
         break;
      }
   }

   /**
    * @brief Arms free list buffer for remote write from FPGA -> GPU.
    * @see triggerRemoteWriteOffset() for something usable with CUDA
    * @param buffer Buffer index to trigger.
    */
   inline void returnFreeListIndex(uint32_t buffer) {
      writeReg(freeListOffset(buffer), 1);
   }

   /**
    * @brief Returns the offset of the free list register from the start of the GpuAsyncCore registers
    * @param buffer The buffer index.
    */
   inline uint32_t freeListOffset(uint32_t buffer) const {
      switch (versionSwitch()) {
      case 0:  // Leaving this to return same as V1 for now
      case 1:
         return GPU_ASYNC_REG_WRITE_DETECT_OFFSET_V1(buffer);
      case 4:
      default:
         return GPU_ASYNC_REG_WRITE_DETECT_OFFSET_V4(buffer);
      }
   }

   /**
    * @brief Returns the offset of the remote read size register from the start of the GpuAsyncCore registers.
    * This is usable in CUDA kernels.
    * @param buffer the buffer index.
    */
   inline uint32_t remoteReadSizeOffset(uint32_t buffer) const {
      switch (versionSwitch()) {
      case 0:  // Leaving this to return same as V1 for now
      case 1:
         return GPU_ASYNC_REG_REMOTE_READ_SIZE_OFFSET_V1(buffer);
      case 4:
      default:
         return GPU_ASYNC_REG_REMOTE_READ_SIZE_OFFSET_V4(buffer);
      }
   }

   /**
    * @brief Get the remote read size for the specified buffer
    */
   inline uint32_t remoteReadSize(uint32_t buffer) const {
      return readReg(remoteReadSizeOffset(buffer));
   }

   /**
    * @brief Set the remote read size for the buffer
    * @param buffer Buffer index
    * @param size Size of the GPU -> FPGA transfer
    */
   inline void setRemoteReadSize(uint32_t buffer, uint32_t size) {
      return writeReg(remoteReadSizeOffset(buffer), size);
   }

protected:
   // Squash version into [0, 1, 4] to make switches cleaner
   inline uint32_t versionSwitch() const {
      switch (version_) {
      case 0:
         return 0;
      case 1:
      case 2:
      case 3:
         return 1;
      case 4:
      default:
         return 4;
      }
   }

   uint32_t readRegV1V4(const GpuAsyncRegister& v1, const GpuAsyncRegister& v4) const {
      switch (versionSwitch()) {
      case 0:
         return 0;  // Unsupported
      case 1:
         return readReg(v1);
      case 4:
      default:
         return readReg(v4);
      }
   }

   void writeRegV1V4(const GpuAsyncRegister& v1, const GpuAsyncRegister& v4, uint32_t val) {
      switch (versionSwitch()) {
      case 0:
         return;  // Unsupported
      case 1:
         return writeReg(v1, val);
      case 4:
      default:
         return writeReg(v4, val);
      }
   }

   volatile uint8_t* regs_;
   uint32_t version_;
};

#endif

#endif  // _GPU_ASYNC_USER_H_
