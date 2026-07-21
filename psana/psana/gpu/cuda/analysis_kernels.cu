// analysis_kernels.cu — fused calibration + azimuthal-integration kernels.
//
// Heavier-weight GPU kernels for psana2: raw uint16 pixels go in, a q-bin
// histogram comes out.  The float32 calibrated array is never materialized
// in global memory — calibration (fused_calib.cuh, identical math to the
// branch's jungfrau_calib_kernel) happens in registers and feeds the
// histogram atomics directly.
//
// Algorithm provenance: the jungfrau_gpu_azint lclstreamer project
// (mfx101344525/results/jungfrau_gpu_azint) — OpenCL originals in
// analysis/jungfrau_processing.py, CUDA designs in
// INSIGHTS_AND_ADDITIONS_CUPY.md.  Fused here with psana2 calibration.
//
// Launch and verification: psana/psana/gpu/test_analysis_kernels.py.
// Compile with -I<this directory> so fused_calib.cuh resolves.
//
// The per-pixel q-bin index (bin_idx) is precomputed once per run on CPU
// from the detector geometry: q = 4*pi/lambda * sin(atan2(r, dist)/2),
// digitized into nbins; bin_idx[i] < 0 marks masked / out-of-range pixels.
// Counts accumulate as float32 (exact below 2^24) so sums and counts share
// one normalize epilogue.

#include "fused_calib.cuh"

extern "C" {

// ---------------------------------------------------------------------------
// fused_calib_azint_kernel — calibrate + 1D azimuthal integration
//
// One thread per pixel: calibrate in-register, atomicAdd into the global
// (nbins,) histogram.  sum_I / sum_N must be zeroed before each launch.
//
// Launch: 1D grid over npixels, block (256,1,1).
// ---------------------------------------------------------------------------
__global__ void fused_calib_azint_kernel(
    const unsigned short* __restrict__ raw,      // flat raw pixels
    const float*          __restrict__ peds,     // (3 * npixels,) mode-major
    const float*          __restrict__ gmask,    // (3 * npixels,) mode-major
    const int*            __restrict__ bin_idx,  // (npixels,), -1 = excluded
    float*                __restrict__ sum_I,    // (nbins,) pre-zeroed
    float*                __restrict__ sum_N,    // (nbins,) pre-zeroed
    const unsigned long long npixels)
{
    const unsigned long long idx =
        (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= npixels) return;

    const int bin = bin_idx[idx];
    if (bin < 0) return;

    const float v = psana_gpu::jungfrau_calib_pixel(
        raw[idx], peds, gmask, idx, npixels);

    atomicAdd(&sum_I[bin], v);
    atomicAdd(&sum_N[bin], 1.0f);
}

// ---------------------------------------------------------------------------
// fused_calib_cm_azint_kernel — calibrate + bank common-mode + azint
//
// One block per (segment, bank); Jungfrau 0.5M panel: 2 halves x
// nbanks_per_row banks, each bank_rows x bank_cols pixels.  Banks are
// block-local, so the whole chain runs in one launch:
//
//   pass 1: calibrate each bank pixel in-register; accumulate the mean of
//           quiet gain-mode-0 pixels (raw < 0x4000, unmasked, |v| < cormax)
//           via shared-memory tree reduction (mean, not psana's median, so
//           no per-bank sort is needed)
//   pass 2: re-calibrate (re-reads hit L2; cheaper than staging 64 MB of
//           calib to global), subtract the bank correction, atomicAdd into
//           the q-histogram
//
// Launch: grid (nsegs, nbanks_total), block (256,1,1).
// sum_I / sum_N must be zeroed before each launch.
// ---------------------------------------------------------------------------
__global__ void fused_calib_cm_azint_kernel(
    const unsigned short* __restrict__ raw,
    const float*          __restrict__ peds,     // (3 * npixels,) mode-major
    const float*          __restrict__ gmask,    // (3 * npixels,) mode-major
    const int*            __restrict__ bin_idx,  // (npixels,), -1 = excluded
    float*                __restrict__ sum_I,    // (nbins,) pre-zeroed
    float*                __restrict__ sum_N,    // (nbins,) pre-zeroed
    const int seg_pixels,        // nrows * ncols per segment
    const int ncols,             // row stride within a segment
    const int bank_rows,         // 256 for Jungfrau
    const int bank_cols,         // 64  for Jungfrau
    const int nbanks_per_row,    // 16  for Jungfrau
    const float cormax,          // |v| ceiling for the common-mode estimate
    const int min_pixels,        // min contributing pixels, else no correction
    const unsigned long long npixels)
{
    const int seg        = blockIdx.x;
    const int bank_idx   = blockIdx.y;
    const int lid        = threadIdx.x;
    const int local_size = blockDim.x;

    __shared__ float s_sum[256];
    __shared__ int   s_count[256];
    __shared__ float s_corr;

    const int half_idx        = bank_idx / nbanks_per_row;
    const int bank_in_half    = bank_idx % nbanks_per_row;
    const int row_start       = half_idx * bank_rows;
    const int col_start       = bank_in_half * bank_cols;
    const int pixels_per_bank = bank_rows * bank_cols;

    // Pass 1: bank common-mode estimate over quiet gain-mode-0 pixels.
    float my_sum   = 0.0f;
    int   my_count = 0;

    for (int i = lid; i < pixels_per_bank; i += local_size) {
        const int r = row_start + i / bank_cols;
        const int c = col_start + i % bank_cols;
        const unsigned long long pix =
            (unsigned long long)seg * seg_pixels
            + (unsigned long long)r * ncols + c;

        const unsigned short rv = raw[pix];
        if (rv >= 0x4000) continue;          // gain-mode-0 pixels only
        if (gmask[pix] == 0.0f) continue;    // masked (mode-0 gain*mask)

        const float v = psana_gpu::jungfrau_calib_pixel(
            rv, peds, gmask, pix, npixels);
        if (fabsf(v) < cormax) {
            my_sum   += v;
            my_count += 1;
        }
    }

    s_sum[lid]   = my_sum;
    s_count[lid] = my_count;
    __syncthreads();

    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            s_sum[lid]   += s_sum[lid + s];
            s_count[lid] += s_count[lid + s];
        }
        __syncthreads();
    }

    if (lid == 0)
        s_corr = (s_count[0] >= min_pixels) ? s_sum[0] / (float)s_count[0] : 0.0f;
    __syncthreads();
    const float correction = s_corr;

    // Pass 2: corrected calibration straight into the histogram.
    for (int i = lid; i < pixels_per_bank; i += local_size) {
        const int r = row_start + i / bank_cols;
        const int c = col_start + i % bank_cols;
        const unsigned long long pix =
            (unsigned long long)seg * seg_pixels
            + (unsigned long long)r * ncols + c;

        const int bin = bin_idx[pix];
        if (bin < 0) continue;

        const float v = psana_gpu::jungfrau_calib_pixel(
            raw[pix], peds, gmask, pix, npixels) - correction;

        atomicAdd(&sum_I[bin], v);
        atomicAdd(&sum_N[bin], 1.0f);
    }
}

// ---------------------------------------------------------------------------
// normalize_kernel — intensity_avg[i] = sum_I[i] / sum_N[i]  (0 where empty)
// ---------------------------------------------------------------------------
__global__ void normalize_kernel(
    const float* __restrict__ sum_I,
    const float* __restrict__ sum_N,
    float*       __restrict__ intensity_avg,
    const long long nbins)
{
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nbins) return;
    const float n = sum_N[idx];
    intensity_avg[idx] = (n > 0.0f) ? sum_I[idx] / n : 0.0f;
}

} // extern "C"
