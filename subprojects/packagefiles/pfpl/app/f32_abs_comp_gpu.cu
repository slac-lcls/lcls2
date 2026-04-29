/**
 *  This file was derived from part of the PFPL code that was published in
 *  IPDPS 2025.  PFPL is a guaranteed-error bound lossy compressor/decompressor
 *  that produces bit-for-bit identical files on CPUs and GPUs.  It has the
 *  following licence and copyright notice.
 *
 *  The latest version of the source of this code is available at
 *  https://github.com/burtscher/PFPL.
 */
/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2025, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.
*/


#include "f32_abs_comp_gpu.hh"

#ifndef NDEBUG
#define NDEBUG
#endif

using byte = unsigned char;
static const int CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]
static const int TPB = 512;  // threads per block [must be power of 2 and at least 128]
#define WS 32


#include <string>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <cuda.h>
#include <sys/time.h>
#include "include/macros.h"
#include "include/sum_reduction.h"
#include "include/max_scan.h"
#include "include/prefix_sum.h"
#include "components/d_DIFFNB_4.h"
#include "components/d_BIT_4.h"
#include "components/d_RZE_1.h"

using namespace PFPL;


// copy (len) bytes from shared memory (source) to global memory (destination)
// source must we word aligned
static inline __device__ void s2g(void* const __restrict__ destination, const void* const __restrict__ source, const int len)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  byte* const __restrict__ output = (byte*)destination;
  if (len < 128) {
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)output;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    const int* const __restrict__ in_w = (int*)input;
    if (bcnt == 0) {
      int* const __restrict__ out_w = (int*)output;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += TPB) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    } else {
      const int shift = bcnt * 8;
      const int rlen = len - bcnt;
      int* const __restrict__ out_w = (int*)&output[bcnt];
      if (tid < bcnt) output[tid] = input[tid];
      if (tid < wcnt) out_w[tid] = __funnelshift_r(in_w[tid], in_w[tid + 1], shift);
      for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
        out_w[i] = __funnelshift_r(in_w[i], in_w[i + 1], shift);
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    }
  }
}


static __device__ int g_chunk_counter;


static __global__ void d_reset()
{
  g_chunk_counter = 0;
}


static inline __device__ void propagate_carry(const int value, const int chunkID, volatile int* const __restrict__ fullcarry, int* const __restrict__ s_fullc)
{
  if (threadIdx.x == TPB - 1) {  // last thread
    fullcarry[chunkID] = (chunkID == 0) ? value : -value;
  }

  if (chunkID != 0) {
    if (threadIdx.x + WS >= TPB) {  // last warp
      const int lane = threadIdx.x % WS;
      const int cidm1ml = chunkID - 1 - lane;
      int val = -1;
      __syncwarp();  // not optional
      do {
        if (cidm1ml >= 0) {
          val = fullcarry[cidm1ml];
        }
      } while ((__any_sync(~0, val == 0)) || (__all_sync(~0, val <= 0)));
#if defined(WS) && (WS == 64)
      const long long mask = __ballot_sync(~0, val > 0);
      const int pos = __ffsll(mask) - 1;
#else
      const int mask = __ballot_sync(~0, val > 0);
      const int pos = __ffs(mask) - 1;
#endif
      int partc = (lane < pos) ? -val : 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      partc = __reduce_add_sync(~0, partc);
#else
      partc += __shfl_xor_sync(~0, partc, 1);
      partc += __shfl_xor_sync(~0, partc, 2);
      partc += __shfl_xor_sync(~0, partc, 4);
      partc += __shfl_xor_sync(~0, partc, 8);
      partc += __shfl_xor_sync(~0, partc, 16);
#endif
      if (lane == pos) {
        const int fullc = partc + val;
        fullcarry[chunkID] = fullc + value;
        *s_fullc = fullc;
      }
    }
  }
}


static __device__ inline void d_QABS_4(int& csize, byte in [CS], byte out [CS], const float errorbound, const float threshold)
{
  using ftype = float;
  using itype = int;
  const int size = csize / sizeof(ftype);
  const int tid = threadIdx.x;

  const int mantissabits = 23;
  const itype maxbin = 1 << (mantissabits - 1);  // leave 1 bit for sign

  const ftype eb2 = 2 * errorbound;
  const ftype inv_eb2 = 0.5f / errorbound;

  const ftype* const data_f = (ftype*)in;
  itype* const data_i = (itype*)out;

  for (int i = tid; i < size; i += TPB) {
    const ftype orig_f = data_f[i];
    const ftype scaled = orig_f * inv_eb2;
    const itype bin = (itype)roundf(scaled);
    const ftype recon = bin * eb2;

    itype val;
    if ((bin >= maxbin) || (bin <= -maxbin) || (fabsf(orig_f) >= threshold) || (fabsf(orig_f - recon) > errorbound) || (orig_f != orig_f)) {  // last check is to handle NaNs
      val = *((itype*)&orig_f);
    } else {
      val = (bin << 1) ^ (bin >> 31);  // TCMS encoding, 'sign' and 'exponent' fields are zero
    }
    data_i[i] = val;
  }
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_encode(const unsigned& index, const byte* const __restrict__ input_base, const int inBufSize, byte* const __restrict__ output_base, const long long outBufSize, int* const __restrict__ fullcarry, const float errorbound, const float threshold)
{
  byte const* const __restrict__ input   = &input_base[index * inBufSize];
  const int                      insize  = inBufSize;
  byte*       const __restrict__ output  = &output_base[index * outBufSize];
  long long*  const __restrict__ outsize = &((long long*)output)[-1]; // Place the size of the reduced data just before the data

  // allocate shared memory buffer
  __shared__ long long chunk [3 * (CS / sizeof(long long))];

  // split into 3 shared memory buffers
  byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
  byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
  byte* const temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

  // initialize
  const int tid = threadIdx.x;
  const int last = 3 * (CS / sizeof(long long)) - 2 - WS;
  const int chunks = (insize + CS - 1) / CS;  // round up
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[2];
  byte* const data_out = (byte*)&size_out[chunks];

  // loop over chunks
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const int chunkID = chunk[last];
    const int base = chunkID * CS;
    if (base >= insize) break;

    // load chunk
    const int osize = min(CS, insize - base);
    long long* const input_l = (long long*)&input[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      out_l[i] = input_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) out[osize - extra + tid] = input[base + osize - extra + tid];

    // encode chunk
    __syncthreads();  // chunk produced, chunk[last] consumed
    int csize = osize;
    bool good = true;
    if (good) {
      byte* tmp = in; in = out; out = tmp;
      d_QABS_4(csize, in, out, errorbound, threshold);
      __syncthreads();
    }
    if (good) {
      byte* tmp = in; in = out; out = tmp;
      d_DIFFNB_4(csize, in, out, temp);
      __syncthreads();
    }
    if (good) {
      byte* tmp = in; in = out; out = tmp;
      d_BIT_4(csize, in, out, temp);
      __syncthreads();
    }
    if (good) {
      byte* tmp = in; in = out; out = tmp;
      good = d_RZE_1(csize, in, out, temp);
      __syncthreads();
    }

    // handle carry
    if (!good || (csize >= osize)) csize = osize;
    propagate_carry(csize, chunkID, fullcarry, (int*)temp);

    // reload chunk if incompressible
    if (tid == 0) size_out[chunkID] = csize;
    if (csize == osize) {
      // store original data
      long long* const out_l = (long long*)out;
      for (int i = tid; i < osize / 8; i += TPB) {
        out_l[i] = input_l[i];
      }
      const int extra = osize % 8;
      if (tid < extra) out[osize - extra + tid] = input[base + osize - extra + tid];
    }
    __syncthreads();  // "out" done, temp produced

    // store chunk
    const int offs = (chunkID == 0) ? 0 : *((int*)temp);
    s2g(&data_out[offs], out, csize);

    // finalize if last chunk
    if ((tid == 0) && (base + CS >= insize)) {
      // output header
      head_out[0] = (long long)insize;
      float* const head_out_f = (float*)&head_out[1];
      head_out_f[0] = errorbound;
      // compute compressed size
      *outsize = &data_out[fullcarry[chunkID]] - output;
    }
  } while (true);
}


struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg); cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg); cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0); cudaEventSynchronize(end); float ms; cudaEventElapsedTime(&ms, beg, end); return 0.001 * ms;}
};


static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n\n", e, line, cudaGetErrorString(e));
    throw std::runtime_error("LC error");
  }
}


PFPL_Compressor::PFPL_Compressor(size_t insize, float errorbound, float threshold) :
  _errorbound(errorbound),
  _threshold (threshold)
{
  _initialize(insize);
}

PFPL_Compressor::PFPL_Compressor(size_t insize, float errorbound) :
  _errorbound(errorbound),
  _threshold (std::numeric_limits<float>::infinity())
{
  _initialize(insize);
}

PFPL_Compressor::~PFPL_Compressor()
{
  // clean up GPU memory
  cudaFree(_d_fullcarry);
  CheckCuda(__LINE__);
}

void PFPL_Compressor::_initialize(size_t insize)
{
  // get GPU info
  //cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "ERROR: no CUDA capable device detected\n\n");
    throw std::runtime_error("PFPL error");
  }
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  _blocks = SMs * (mTpSM / TPB);
  const int chunks = (insize + CS - 1) / CS;  // round up
  CheckCuda(__LINE__);
  _maxsize = 3 * sizeof(int) + chunks * sizeof(short) + chunks * CS;
  printf("*** SMs %d, mTpSM %d, blks %d, chunks %lld, maxSz %llu\n", SMs, mTpSM, _blocks, chunks, _maxsize);

  // allocate GPU memory
  cudaMalloc((void**)&_d_fullcarry, chunks * sizeof(int));
  CheckCuda(__LINE__);

  if (_threshold < std::numeric_limits<float>::min()) {
    fprintf(stderr, "ERROR: threshold must be a positive, normal, floating-point value\n");
    throw std::runtime_error("PFPL error");
  }
}

void PFPL_Compressor::banner() const
{
  printf("PFPL GPU Single-Precision ABS Compressor\n");
  printf("Copyright 2025 Texas State University\n\n");
}

void PFPL_Compressor::updateGraph(cudaStream_t      stream,
                                  const unsigned&   index,
                                  byte const* const d_input_base,
                                  const long long   inBufSize,
                                  byte* const       d_encoded_base,
                                  const long long   encBufSize)
{
  d_reset<<<1, 1, 0, stream>>>();
  const int chunks = (inBufSize + CS - 1) / CS;  // round up
  cudaMemsetAsync(_d_fullcarry, 0, chunks * sizeof(byte), stream);
  d_encode<<<_blocks, TPB, 0, stream>>>(index, d_input_base, inBufSize, d_encoded_base, encBufSize, _d_fullcarry, _errorbound, _threshold);
}
