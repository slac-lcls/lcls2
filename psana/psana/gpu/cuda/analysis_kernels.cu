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
// fused_calib_azint_sorted_kernel — calibrate + 1D azint, no atomics
//
// The pre-sorted strategy from jungfrau_gpu_azint DirectIntegrate
// (method='sorted'), fused one step further: instead of gathering into a
// bin-contiguous staging array, each bin's block reads its pixels through
// the sort_order indirection and calibrates in-register — so neither the
// calibrated array nor the gather intermediate ever exists in global
// memory.  sort_order lists the indices of all VALID (unmasked, in-range)
// pixels grouped by bin; bin_offsets[b]..bin_offsets[b+1] is bin b's range.
// Both are precomputed once per run on CPU (geometry is fixed).
//
// One block per bin: threads stride the bin's range accumulating private
// partial sums (registers), then tree-reduce in shared memory.  Every bin
// is written unconditionally, so sum_I / sum_N need no pre-zeroing.
//
// Launch: grid (nbins,1,1), block (256,1,1).
// ---------------------------------------------------------------------------
__global__ void fused_calib_azint_sorted_kernel(
    const unsigned short* __restrict__ raw,
    const float*          __restrict__ peds,        // (3 * npixels,) mode-major
    const float*          __restrict__ gmask,       // (3 * npixels,) mode-major
    const int*            __restrict__ sort_order,  // (n_valid,) grouped by bin
    const int*            __restrict__ bin_offsets, // (nbins + 1,)
    float*                __restrict__ sum_I,       // (nbins,)
    float*                __restrict__ sum_N,       // (nbins,)
    const unsigned long long npixels)
{
    const int bin        = blockIdx.x;
    const int lid        = threadIdx.x;
    const int local_size = blockDim.x;

    __shared__ float s_I[256];
    __shared__ float s_N[256];

    const int start = bin_offsets[bin];
    const int end   = bin_offsets[bin + 1];

    float my_I = 0.0f;
    float my_N = 0.0f;

    for (int i = start + lid; i < end; i += local_size) {
        const unsigned long long pix = (unsigned long long)sort_order[i];
        my_I += psana_gpu::jungfrau_calib_pixel(
            raw[pix], peds, gmask, pix, npixels);
        my_N += 1.0f;
    }

    s_I[lid] = my_I;
    s_N[lid] = my_N;
    __syncthreads();

    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            s_I[lid] += s_I[lid + s];
            s_N[lid] += s_N[lid + s];
        }
        __syncthreads();
    }

    if (lid == 0) {
        sum_I[bin] = s_I[0];
        sum_N[bin] = s_N[0];
    }
}

// ===========================================================================
// Post-calibration (float-input) variants
//
// The GPUDetector pipeline always materializes the calibrated frame into
// per-slot batch buffers (its batching/D2H currency), so registry kernels
// producing derived results consume that buffer directly.  Measured on
// A100 / 16.8 Mpix this is also faster than the fused raw-input variants
// above: every pass below is coalesced, while the fused kernels pay random
// -access amplification on peds/gmask.  The fused variants remain for
// kernel-only benchmarking and for a future reduction-only pipeline mode
// that skips the calib product entirely.
// ===========================================================================

// One thread per pixel, atomicAdd into the global histogram.
// sum_I / sum_N must be zeroed before each launch.
__global__ void azint_global_kernel(
    const float* __restrict__ data,      // flat calibrated pixels
    const int*   __restrict__ bin_idx,   // (npix,), -1 = excluded
    float*       __restrict__ sum_I,     // (nbins,) pre-zeroed
    float*       __restrict__ sum_N,     // (nbins,) pre-zeroed
    const unsigned long long npix)
{
    const unsigned long long idx =
        (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= npix) return;
    const int bin = bin_idx[idx];
    if (bin < 0) return;
    atomicAdd(&sum_I[bin], data[idx]);
    atomicAdd(&sum_N[bin], 1.0f);
}

// Gather valid pixels into bin-contiguous order:
// sorted_data[i] = data[sort_order[i]], i in [0, n_valid).
__global__ void azint_gather_kernel(
    const float* __restrict__ data,
    const int*   __restrict__ sort_order,
    float*       __restrict__ sorted_data,
    const unsigned long long n_valid)
{
    const unsigned long long i =
        (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_valid) return;
    sorted_data[i] = data[sort_order[i]];
}

// One block per bin: threads stride the bin's contiguous range in
// sorted_data and tree-reduce in shared memory.  No atomics; writes every
// bin unconditionally, so sum_I / sum_N need no pre-zeroing.
__global__ void azint_sorted_kernel(
    const float* __restrict__ sorted_data,
    const int*   __restrict__ bin_offsets,   // (nbins + 1,)
    float*       __restrict__ sum_I,
    float*       __restrict__ sum_N,
    const int nbins)
{
    const int bin        = blockIdx.x;
    const int lid        = threadIdx.x;
    const int local_size = blockDim.x;

    __shared__ float s_I[256];
    __shared__ float s_N[256];

    const int start = bin_offsets[bin];
    const int end   = bin_offsets[bin + 1];

    float my_I = 0.0f;
    float my_N = 0.0f;
    for (int i = start + lid; i < end; i += local_size) {
        my_I += sorted_data[i];
        my_N += 1.0f;
    }

    s_I[lid] = my_I;
    s_N[lid] = my_N;
    __syncthreads();

    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            s_I[lid] += s_I[lid + s];
            s_N[lid] += s_N[lid + s];
        }
        __syncthreads();
    }

    if (lid == 0) {
        sum_I[bin] = s_I[0];
        sum_N[bin] = s_N[0];
    }
}

// Bank-level common-mode correction of an already-calibrated frame, in
// place.  Same bank layout and estimate as fused_calib_cm_azint_kernel
// pass 1; raw supplies the gain bits, gmask (mode 0) the bad-pixel mask.
// Launch: grid (nsegs, nbanks_total), block (256,1,1).
__global__ void common_mode_bank_kernel(
    float*                __restrict__ data,    // flat calib, in place
    const unsigned short* __restrict__ raw,     // flat raw (gain bits)
    const float*          __restrict__ gmask,   // (3 * npixels,) mode-major
    const int seg_pixels,
    const int ncols,
    const int bank_rows,
    const int bank_cols,
    const int nbanks_per_row,
    const float cormax,
    const int min_pixels)
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

    float my_sum   = 0.0f;
    int   my_count = 0;

    for (int i = lid; i < pixels_per_bank; i += local_size) {
        const int r = row_start + i / bank_cols;
        const int c = col_start + i % bank_cols;
        const long long pix =
            (long long)seg * seg_pixels + (long long)r * ncols + c;

        if (raw[pix] >= 0x4000) continue;    // gain-mode-0 pixels only
        if (gmask[pix] == 0.0f) continue;    // masked

        const float v = data[pix];
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

    for (int i = lid; i < pixels_per_bank; i += local_size) {
        const int r = row_start + i / bank_cols;
        const int c = col_start + i % bank_cols;
        const long long pix =
            (long long)seg * seg_pixels + (long long)r * ncols + c;
        data[pix] -= correction;
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
