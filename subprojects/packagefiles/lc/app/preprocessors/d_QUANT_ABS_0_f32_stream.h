/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

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

URL: The latest version of this code is available at https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.
*/


static __global__ void d_QUANT_ABS_0_f32_kernel(const long long len, byte* const __restrict__ data, const float errorbound, const float eb2, const float inv_eb2, const float threshold)
{
  float* const data_f = (float*)data;
  int* const data_i = (int*)data;

  const int mantissabits = 23;
  const int maxbin = 1 << (mantissabits - 1);  // leave 1 bit for sign
  const long long idx = threadIdx.x + (long long)blockIdx.x * TPB;
  if (idx < len) {
    const float orig_f = data_f[idx];
    const float scaled = orig_f * inv_eb2;
    const int bin = (int)std::round(scaled);
    const float recon = bin * eb2;

    if ((bin >= maxbin) || (bin <= -maxbin) || (std::abs(orig_f) >= threshold) || (std::abs(orig_f - recon) > errorbound) || (orig_f != orig_f)) {  // last check is to handle NaNs
      assert(((data_i[idx] >> mantissabits) & 0xff) != 0);
    } else {
      data_i[idx] = (bin << 1) ^ (bin >> 31);  // TCMS encoding, 'sign' and 'exponent' fields are zero
    }
  }
}


static __global__ void d_iQUANT_ABS_0_f32_kernel(const long long len, byte* const __restrict__ data, const float eb2)
{
  float* const data_f = (float*)data;
  int* const data_i = (int*)data;

  const int mantissabits = 23;
  const long long idx = threadIdx.x + (long long)blockIdx.x * TPB;
  if (idx < len) {
    int bin = data_i[idx];
    if ((0 <= bin) && (bin < (1 << mantissabits))) {  // is encoded value
      bin = (bin >> 1) ^ (((bin << 31) >> 31));  // TCMS decoding
      data_f[idx] = bin * eb2;
    }
  }
}


static inline void d_QUANT_ABS_0_f32(long long& size, byte*& data, const int paramc, const double paramv [], cudaStream_t stream)
{
  if (size % sizeof(float) != 0) {fprintf(stderr, "QUANT_ABS_0_f32: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(float)); throw std::runtime_error("LC error");}
  const long long len = size / sizeof(float);
  if ((paramc != 1) && (paramc != 2)) {fprintf(stderr, "USAGE: QUANT_ABS_0_f32(error_bound [, threshold])\n"); throw std::runtime_error("LC error");}
  const float errorbound = paramv[0];
  const float threshold = (paramc == 2) ? paramv[1] : std::numeric_limits<float>::infinity();
  if (errorbound < std::numeric_limits<float>::min()) {fprintf(stderr, "QUANT_ABS_0_f32: ERROR: error_bound must be at least %e\n", std::numeric_limits<float>::min()); throw std::runtime_error("LC error");}  // minimum positive normalized value
  if (threshold <= errorbound) {fprintf(stderr, "QUANT_ABS_0_f32: ERROR: threshold must be larger than error_bound\n"); throw std::runtime_error("LC error");}

  const float eb2 = 2 * errorbound;
  const float inv_eb2 = 0.5f / errorbound;

  d_QUANT_ABS_0_f32_kernel<<<(len + TPB - 1) / TPB, TPB, 0, stream>>>(len, data, errorbound, eb2, inv_eb2, threshold);
}


static inline void d_iQUANT_ABS_0_f32(long long& size, byte*& data, const int paramc, const double paramv [], cudaStream_t stream)
{
  if (size % sizeof(float) != 0) {fprintf(stderr, "QUANT_ABS_0_f32: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(float)); throw std::runtime_error("LC error");}
  const long long len = size / sizeof(float);
  if ((paramc != 1) && (paramc != 2)) {fprintf(stderr, "USAGE: QUANT_ABS_0_f32(error_bound [, threshold])\n"); throw std::runtime_error("LC error");}
  const float errorbound = paramv[0];
  if (errorbound < std::numeric_limits<float>::min()) {fprintf(stderr, "QUANT_ABS_0_f32: ERROR: error_bound must be at least %e\n", std::numeric_limits<float>::min()); throw std::runtime_error("LC error");}  // minimum positive normalized value

  const float eb2 = 2 * errorbound;

  d_iQUANT_ABS_0_f32_kernel<<<(len + TPB - 1) / TPB, TPB, 0, stream>>>(len, data, eb2);
}
