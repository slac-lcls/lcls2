/**
 *  This file was derived from part of the SLEEK code that was published in
 *  IPDPS 2026.  SLEEK, a new main-memory compression algorithm for IEEE 754-
 *  -formatted single- and double-precision floating-point data that delivers
 *  speeds comparable to or exceeding the speed of CUDA memcpy on GPUs.  It
 *  has the following licence and copyright notice.
 *
 *  The latest version of the source of this code is available at
 *  https://github.com/burtscher/SLEEK.
 */

/*
This file is part of SLEEK, a set of ultra-fast lossless and guaranteed-error-bounded lossy main-memory compression algorithms for floating-point data on GPUs.

BSD 3-Clause License

Copyright (c) 2026, Anju Mongandampulath Akathoott, Andrew Rodriguez, and Martin Burtscher

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

URL: The latest version of this code is available at https://github.com/burtscher/SLEEK/.

Publication: This work is described in detail in the following paper.
Anju Mongandampulath Akathoott, Andrew Rodriguez, and Martin Burtscher. "SLEEK: Compressing Memory Copies for Floating-Point Data on GPUs." Proceedings of the 40th IEEE International Parallel and Distributed Processing Symposium (IPDPS'26). May 2026.

Sponsor: This material is based upon work supported by the U.S. National Science Foundation under Grant Number 2403380 and by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under Award Number DE-SC0022223.
*/


#include "SingleCompressorLossy.hh"

#ifndef NDEBUG
#define NDEBUG
#endif


#include <string>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include <stdexcept>

using byte = unsigned char;
using type_u = unsigned int;
using type_f = float;
static const int CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]
static const int TPB = 512;  // threads per block [must be power of 2 and at least 128]
#define WS 32


#define to_float(val) \
    (sizeof(type_f) == sizeof(float) ? std::stof(val) : std::stod(val))


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  #define __all(arg) __all_sync(~0, arg)
  #define __any(arg) __any_sync(~0, arg)
  #define __ballot(arg) __ballot_sync(~0, arg)
  #define __shfl_up(...) __shfl_up_sync(~0, __VA_ARGS__)
  #define __shfl_xor(...) __shfl_xor_sync(~0, __VA_ARGS__)
#endif





static __device__ inline type_u quantize(const type_u val, const int eb_e, const int thr_e, const int offs)
{
  const int e = 8;  // exponent bits
  const int m = 23;  // mantissa bits
  const int abs = val & (((unsigned int)1 << (e + m)) - 1);  // compute absolute value
  const int val_e = abs >> m;  // extract exponent
  int enc = 0;  // default value is 0
  if (val_e >= thr_e) {  // at or above threshold
    enc = abs - offs;  // lossless encoding
  } else if (val_e >= eb_e) {  // lossy encoding
    int mant = val & ((1 << m) - 1);  // extract mantissa
    const int shift = thr_e - val_e;  // bias cancels out
    mant |= 1 << m;  // insert implicit 1
    mant += 1 << (shift - 1);  // round to nearest, ties round away from zero
    enc = mant >> shift;  // shift out unnecessary bits
  }
  enc = (enc << 1) | (~val >> (e + m));  // magnitude ~sign
  if (enc != 0) enc--;  // -0 -> +0 and fill gap
  return enc;
}


static __device__ inline bool d_SLEEK(int& csize, byte in [CS], byte out [CS], byte temp [CS], const int eb_e, const int thr_e, const int offs)
{
  const int TB = sizeof(type_u) * 8;  // number of bits in type_u
  const int size = CS / sizeof(type_u);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;

  static_assert(sizeof(type_u) >= 4);
  static_assert(chunksize % WS == 0);
  static_assert(WS == SC);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<type_u>::value);

  const int tid = threadIdx.x;
  const int lane = tid % WS;
  const int warp = tid / WS;
  const int warps = TPB / WS;

  // clear unused part of input buffer
  type_u* const in_t = (type_u*)in;
  if (csize < CS) {
    if (csize % sizeof(type_u) == 0) {
      for (int i = csize / sizeof(type_u) + tid; i < size; i += TPB) {
        in_t[i] = 0;
      }
    } else {
      for (int i = csize + tid; i < CS; i += TPB) {
        in[i] = 0;
      }
    }
    __syncthreads();
  }

  // determine bits needed for each subchunk
  int ln = -1;
  for (int i = warp; i < SC; i += warps) {
    const int beg = i * chunksize;
    const int end = beg + chunksize;

    // max of values for each thread
    type_u max_val = 0;
    for (int j = beg + lane; j < end; j += WS) {
      type_u val = quantize(in_t[j], eb_e, thr_e, offs);
      in_t[j] = val;
      max_val = max(max_val, val);
    }

    // warp level max
    max_val = max(max_val, __shfl_xor(max_val, 1));
    max_val = max(max_val, __shfl_xor(max_val, 2));
    max_val = max(max_val, __shfl_xor(max_val, 4));
    max_val = max(max_val, __shfl_xor(max_val, 8));
    max_val = max(max_val, __shfl_xor(max_val, 16));

    // figure out number of bits needed
    if (lane == i) {
      int cnt = TB;
      if (max_val != 0) {
        cnt = (TB == 64) ? __clzll(max_val) : __clz(max_val);
      }
      ln = TB - cnt;  // logn value for each chunk
    }
  }
  if (ln >= 0) out[lane] = ln;
  __syncthreads();

  // warp prefix sum over bits
  int* const bits = (int*)temp;
  if (warp == 0) {
    const int org = out[lane] * chunksize;
    int val = org;
    int tmp = __shfl_up(val, 1);
    if (lane >= 1) val += tmp;
    tmp = __shfl_up(val, 2);
    if (lane >= 2) val += tmp;
    tmp = __shfl_up(val, 4);
    if (lane >= 4) val += tmp;
    tmp = __shfl_up(val, 8);
    if (lane >= 8) val += tmp;
    tmp = __shfl_up(val, 16);
    if (lane >= 16) val += tmp;
    bits[lane] = val - org;
    if (lane == SC - 1) bits[SC] = val;
  }
  __syncthreads();

  // check if encoded data fits
  const int tot = bits[SC];
  const int newsize = (SC * 8 + tot + 16) / 8;
  if (newsize >= CS) return false;

  // clear out buffer
  type_u* const out_t = (type_u*)&out[SC];
  for (int i = tid; i < tot / TB; i += TPB) out_t[i] = 0;
  __syncthreads();

  // encode data values
  for (int i = warp; i < SC; i += warps) {
    const int logn = out[i];
    if (logn > 0) {
      const int beg = i * chunksize;
      const int end = beg + chunksize;
      if (logn == TB) {
        const int offs = bits[i] / TB - beg;
        for (int j = beg + lane; j < end; j += WS) {
          out_t[offs + j] = in_t[j];
        }
      } else {
        const int incr = WS * 2 * logn;
        int loc = bits[i] + lane * 2 * logn;
        for (int j = beg + lane * 2; j < end; j += WS * 2) {
          const type_u val = in_t[j];
          const type_u valB = in_t[j + 1];
          const int loc2 = loc + logn;
          const int pos = loc / TB;
          const int shift = loc % TB;
          const int pos2 = loc2 / TB;
          const int shift2 = loc2 % TB;
          const int val1 = val << shift;
          const int val2 = valB << shift2;
          const int val3 = val >> (TB - shift);
          const int val4 = valB >> (TB - shift2);
          if (val1 != 0) atomicOr_block(&out_t[pos], val1);
          if (val2 != 0) atomicOr_block(&out_t[pos2], val2);
          if ((val3 != 0) && (shift != 0)) atomicOr_block(&out_t[pos + 1], val3);
          if ((val4 != 0) && (shift2 != 0)) atomicOr_block(&out_t[pos2 + 1], val4);
          loc += incr;
        }
      }
    }
  }

  // output header info
  if (tid == 0) {
    *(short*)&out[newsize - 2] = csize;
  }

  csize = newsize;
  return true;
}


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
      } while ((__any(val == 0)) || (__all(val <= 0)));
      const int mask = __ballot(val > 0);
      const int pos = __ffs(mask) - 1;
      long long partc = (lane < pos) ? -val : 0;
      partc += __shfl_xor(partc, 1);
      partc += __shfl_xor(partc, 2);
      partc += __shfl_xor(partc, 4);
      partc += __shfl_xor(partc, 8);
      partc += __shfl_xor(partc, 16);

      if (lane == pos) {
        const long long fullc = partc + val;
        fullcarry[chunkID] = fullc + value;
        *s_fullc = fullc;
      }
    }
  }
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 4)
#else
static __global__ __launch_bounds__(TPB, 3)
#endif
void d_encode(unsigned* const __restrict__ state, unsigned const* const __restrict__ index,
              const byte* const __restrict__ inputBase, const long long inBufSize, byte* const __restrict__ outputBase, long long const outBufSize, long long* const __restrict__ fullcarry, const int eb_e, const int thr_e, const int offs)
{
  // Added by RiC:
  if (state && (*state != 1))  return;  // Skip when not in the right state

  unsigned const idx{*index};           // Dereference only once
  byte const* const __restrict__ input   = &inputBase[idx * inBufSize];
  const int                      insize  = inBufSize;
  byte*       const __restrict__ output  = &outputBase[idx * outBufSize];
  long long*  const __restrict__ outsize = &((long long*)output)[-1]; // Place the size of the reduced data just before the data
  // End RiC

  // allocate shared memory buffer
  __shared__ long long chunk [2 * (CS / sizeof(long long)) + 17];

  // split into 3 shared memory buffers
  byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
  byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
  byte* const temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

  // initialize
  const int tid = threadIdx.x;
  const long long last = 2 * (CS / sizeof(long long)) + 1;
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
    byte* tmp = in; in = out; out = tmp;
    bool good = d_SLEEK(csize, in, out, temp, eb_e, thr_e, offs);
   __syncthreads();

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
      *outsize = &data_out[fullcarry[chunkID]] - output;

      // Added by RiC:
      if (state)  *state = 2;           // Advance to the next state
      // End RiC
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
    exit(-1);
  }
}


#if defined(ARTIFACT)
  static inline type_f computeNOAeb(const type_f* const input, const long long size, const type_f eb_param)
  {
    type_f min = input[0], max = input[0];
    #pragma omp parallel for default(none) shared(size, input) reduction(max: max) reduction(min: min)
    for (long long i = 1; i < size; i++) {
      const type_f val = input[i];
      if (val < min)
        min = val;
      else if (val > max)
        max = val;
    }

    printf("min_val: %.10f\n", min);
    printf("max_val: %.10f\n", max);
    printf("diff between max and min values: %.10f\n", (max - min));
    return (max - min) * eb_param;
  }
#endif

// ---

using namespace SLEEK;

SingleCompressorLossy::SingleCompressorLossy(size_t inSize, float errorBound)
{
  if (_initialize(inSize, errorBound)) {
    fprintf(stderr, "SLEEK error\n");
    abort();
  }
}

SingleCompressorLossy::~SingleCompressorLossy()
{
  // clean up GPU memory
  cudaFree(_d_fullcarry);
  CheckCuda(__LINE__);
}

int SingleCompressorLossy::_initialize(size_t insize, float errorBound)
{
  #if defined(ARTIFACT)
    const type_f parameter = errorBound;
    type_f* t_input = (type_f*)input;
    const type_f eb = computeNOAeb(t_input, insize / sizeof(type_f), parameter);
  #else
    const type_f eb = errorBound;
  #endif

  // eb variables
  const int e = 8;  // exponent bits
  const int m = 23;  // mantissa bits
  _eb_e = (*((int*)&eb) >> m) & ((1 << e) - 1);  // extract biased exponent
  _thr_e = _eb_e + (m + 1);  // biased exponent of threshold
  _offs = (_thr_e << m) - (1 << m);  // offset for lossless encoding
  if (_thr_e >= (1 << e) - 1) {fprintf(stderr, "QUANT_IABS_0_f32: ERROR: error_bound is too large\n"); return -1;}

  if (insize % sizeof(type_f) != 0) {fprintf(stderr, "ERROR: size of input must be a multiple of %ld bytes\n", sizeof(type_f)); return -1;}

  #if defined(ARTIFACT)
    // Check if the third argument is "y" to enable performance analysis
    char* perf_str = argv[4];
    bool perf = false;
    if (perf_str != nullptr && strcmp(perf_str, "y") == 0) {
      perf = true;
    } else if (perf_str != nullptr && strcmp(perf_str, "y") != 0) {
      fprintf(stderr, "ERROR: Invalid argument. Use 'y' or nothing.\n");
      return -1;
    }
  #endif

  // get GPU info
  //cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "ERROR: no CUDA capable device detected\n\n");
    return -1;
  }
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  _blocks = SMs * (mTpSM / TPB);
  _chunks = (insize + CS - 1) / CS;  // round up
  CheckCuda(__LINE__);
  _maxsize = 3 * sizeof(int) + _chunks * sizeof(short) + _chunks * CS;
  printf("*** SMs %d, mTpSM %d, blks %d, chunks %lld, maxSz %llu\n", SMs, mTpSM, _blocks, _chunks, _maxsize);

  // allocate GPU memory
  cudaMalloc((void**)&_d_fullcarry, _chunks * sizeof(long long));
  CheckCuda(__LINE__);

  return 0;
}

void SingleCompressorLossy::banner() const
{
  printf("GPU SLEEK 1.0: single-precision lossy compressor\n");
  printf("Copyright 2026 Texas State University\n\n");
}

void SingleCompressorLossy::updateGraph(cudaStream_t         stream,
                                        unsigned*      const state_d,
                                        unsigned*      const index_d,
                                        uint8_t const* const d_input_base,
                                        long long      const inBufSize,
                                        uint8_t*       const d_encoded_base,
                                        long long      const encBufSize)
{
  d_reset<<<1, 1, 0, stream>>>();
  cudaMemsetAsync(_d_fullcarry, 0, _chunks * sizeof(long long), stream);
  d_encode<<<_blocks, TPB, 0, stream>>>(state_d,
                                        index_d,
                                        d_input_base,
                                        inBufSize,
                                        d_encoded_base,
                                        encBufSize,
                                        _d_fullcarry,
                                        _eb_e,
                                        _thr_e,
                                        _offs);
}
