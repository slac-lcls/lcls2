/**
 *  This file was derived from part of the LC framework for synthesizing
 *  high-speed parallel lossless and error-bounded lossy data compression
 *  and decompression algorithms for CPUs and GPUs, which has the following
 *  licence and copyright notice.
 *
 *  The latest version of the source of this code is available at
 *  https://github.com/burtscher/LC-framework.
 */
/*
BSD 3-Clause License

Copyright (c) 2021-2025, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher
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
*/


#include "lc-compressor-QUANT_ABS_0_f32-BIT_4-RZE_1.hh"

#ifndef NDEBUG
#define NDEBUG
#endif

using byte = unsigned char;
static const int CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]
static const int TPB = 512;  // threads per block [must be power of 2 and at least 128]
#if defined(__AMDGCN_WAVEFRONT_SIZE) && (__AMDGCN_WAVEFRONT_SIZE == 64)
#define WS 64
#else
#define WS 32
#endif

#include <string>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <cuda.h>
#include "include/macros.h"
#include "include/sum_reduction.h"
#include "include/max_scan.h"
#include "include/prefix_sum.h"
#include "preprocessors/d_QUANT_ABS_0_f32_stream.h"
#include "components/d_BIT_4.h"
#include "components/d_RZE_1.h"

using namespace LC_framework;


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


static __device__ unsigned long long g_chunk_counter;


static __global__ void d_reset()
{
  g_chunk_counter = 0LL;
}


static inline __device__ void propagate_carry(const int value, const long long chunkID, volatile long long* const __restrict__ fullcarry, long long* const __restrict__ s_fullc)
{
  if (threadIdx.x == TPB - 1) {  // last thread
    fullcarry[chunkID] = (chunkID == 0) ? (long long)value : (long long)-value;
  }

  if (chunkID != 0) {
    if (threadIdx.x + WS >= TPB) {  // last warp
      const int lane = threadIdx.x % WS;
      const long long cidm1ml = chunkID - 1 - lane;
      long long val = -1;
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
      long long partc = (lane < pos) ? -val : 0;
//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
//      partc = __reduce_add_sync(~0, partc);
//#else
      partc += __shfl_xor_sync(~0, partc, 1);
      partc += __shfl_xor_sync(~0, partc, 2);
      partc += __shfl_xor_sync(~0, partc, 4);
      partc += __shfl_xor_sync(~0, partc, 8);
      partc += __shfl_xor_sync(~0, partc, 16);
#if defined(WS) && (WS == 64)
      partc += __shfl_xor_sync(~0, partc, 32);
#endif
//#endif
      if (lane == pos) {
        const long long fullc = partc + val;
        fullcarry[chunkID] = fullc + value;
        *s_fullc = fullc;
      }
    }
  }
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_encode(const unsigned& index, const byte* const __restrict__ input, const long long inBufSize, byte* const __restrict__ out_base, const long long out_size, long long* const __restrict__ fullcarry)
{
  const int                insize = inBufSize;
  byte* const __restrict__ output = &out_base[index * out_size];

  // allocate shared memory buffer
  __shared__ long long chunk [3 * (CS / sizeof(long long))];

  // split into 3 shared memory buffers
  byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
  byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
  byte* const temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

  // initialize
  const int tid = threadIdx.x;
  const long long last = 3 * (CS / sizeof(long long)) - 2 - WS;
  const long long chunks = (insize + CS - 1) / CS;  // round up
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[1];
  byte* const data_out = (byte*)&size_out[chunks];

  // loop over chunks
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1LL);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const long long chunkID = chunk[last];
    const long long base = chunkID * CS;
    if (base >= insize) break;

    // load chunk
    const int osize = (int)min((long long)CS, insize - base);
    long long* const input_l = (long long*)&input[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      out_l[i] = input_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) out[(long long)osize - (long long)extra + (long long)tid] = input[base + (long long)osize - (long long)extra + (long long)tid];

    // encode chunk
    __syncthreads();  // chunk produced, chunk[last] consumed
    int csize = osize;
    bool good = true;
    if (good) {
      byte* tmp = in; in = out; out = tmp;
      good = d_BIT_4(csize, in, out, temp);
     __syncthreads();
    }
    if (good) {
      byte* tmp = in; in = out; out = tmp;
      good = d_RZE_1(csize, in, out, temp);
     __syncthreads();
    }

    // handle carry
    if (!good || (csize >= osize)) csize = osize;
    propagate_carry(csize, chunkID, fullcarry, (long long*)temp);

    // reload chunk if incompressible
    if (tid == 0) size_out[chunkID] = csize;
    if (csize == osize) {
      // store original data
      long long* const out_l = (long long*)out;
      for (long long i = tid; i < osize / 8; i += TPB) {
        out_l[i] = input_l[i];
      }
      const int extra = osize % 8;
      if (tid < extra) out[(long long)osize - (long long)extra + (long long)tid] = input[base + (long long)osize - (long long)extra + (long long)tid];
    }
    __syncthreads();  // "out" done, temp produced

    // store chunk
    const long long offs = (chunkID == 0) ? 0 : *((long long*)temp);
    s2g(&data_out[offs], out, csize);

    // finalize if last chunk
    if ((tid == 0) && (base + CS >= insize)) {
      // output header
      head_out[0] = insize;
      // compute compressed size
      auto outsize = &data_out[fullcarry[chunkID]] - output;

      // Place the size of the reduced data just before the data
      ((long long*)output)[-1] = outsize;
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

static __global__ void d_prepare(const unsigned&                index,
                                 byte const* const __restrict__ d_input_base,
                                 const long long                insize,
                                 byte      * const __restrict__ dpreencdata)
{
  byte const* const __restrict__ d_input = &d_input_base[index * insize];

  // @todo: Better to do this in steps of 32 or 64 bit words?
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = offset; i < insize; i += stride) {
    dpreencdata[i] = d_input[i];
  }
}


LC_Compressor::LC_Compressor(size_t insize, double paramv)
{
  // get GPU info
  //cudaSetDevice(0);  // This causes CheckCuda in the dtor to report an error
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no CUDA capable device detected\n\n"); throw std::runtime_error("LC error");}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  _blocks = SMs * (mTpSM / TPB);
  const long long chunks = (insize + CS - 1) / CS;  // round up
  CheckCuda(__LINE__);
  _maxsize = 2 * sizeof(long long) + chunks * sizeof(short) + chunks * CS;
  printf("*** SMs %d, mTpSM %d, blks %d, chunks %lld, maxSz %llu\n", SMs, mTpSM, _blocks, chunks, _maxsize);

  // allocate GPU memory
  cudaMalloc((void **)&_dpreencdata, insize);
  cudaMalloc((void **)&_d_fullcarry, chunks * sizeof(long long));
  CheckCuda(__LINE__);

  _paramv[0] = paramv;
}

LC_Compressor::~LC_Compressor()
{
  // clean up GPU memory
  cudaFree(_d_fullcarry);
  cudaFree(_dpreencdata);
  CheckCuda(__LINE__);
}

void LC_Compressor::banner() const
{
  printf("GPU LC 1.2 Algorithm: QUANT_ABS_0_f32 BIT_4 RZE_1\n");
  printf("Copyright 2024 Texas State University\n\n");
}

void LC_Compressor::updateGraph(cudaStream_t      stream,
                                const unsigned&   index,
                                byte const* const d_input_base,
                                const long long   inBufSize,
                                byte* const       d_encoded_base,
                                const long long   encBufSize)
{
  //cudaMemcpy(dpreencdata, d_input, inBufSize, cudaMemcpyDeviceToDevice);
  d_prepare<<<_blocks, TPB, 0, stream>>>(index, d_input_base, inBufSize, _dpreencdata);
  long long dpreencsize = inBufSize;

  d_QUANT_ABS_0_f32(dpreencsize, _dpreencdata, 1, _paramv, stream);
  d_reset<<<1, 1, 0, stream>>>();
  const long long chunks = (inBufSize + CS - 1) / CS;  // round up
  cudaMemsetAsync(_d_fullcarry, 0, chunks * sizeof(long long), stream);
  d_encode<<<_blocks, TPB, 0, stream>>>(index, _dpreencdata, dpreencsize, d_encoded_base, encBufSize, _d_fullcarry);
}
