#pragma once

namespace psana_gpu {

// ---------------------------------------------------------------------------
// jungfrau_calib_pixel
//
// Per-pixel Jungfrau calibration helper called from the __global__ kernel
// defined in gpu_calib.py.
//
// Subtracts the mode-appropriate pedestal and scales by the gain*mask factor:
//
//   calib[i] = (raw[i] & 0x3fff  -  peds[mode * npixels + i])
//              * gmask[mode * npixels + i]
//
// Gain-bit to mode-index mapping (top 2 bits of the raw uint16 value):
//   gain_bits == 0  ->  mode 0  (g0 / low-gain)
//   gain_bits == 1  ->  mode 1  (g1 / medium-gain)
//   gain_bits == 3  ->  mode 2  (g2 / high-gain)
//   gain_bits == 2  ->  bad pixel, returns 0.0
//
// Parameters
// ----------
// raw_val  : single raw uint16 pixel value
// peds     : flat float32 array, length 3 * npixels, mode-major C order
// gmask    : flat float32 array, same layout (gmask = 1/pixel_gain * mask,
//              precomputed on CPU and transferred once per run)
// pixel_idx: flat pixel index in [0, npixels)
// npixels  : nsegs * nrows * ncols
// ---------------------------------------------------------------------------

__device__ inline float jungfrau_calib_pixel(
    unsigned short         raw_val,
    const float*           peds,
    const float*           gmask,
    unsigned long long     pixel_idx,
    unsigned long long     npixels)
{
    const unsigned int gain_bits = raw_val >> 14;

    unsigned int mode;
    if      (gain_bits == 0) { mode = 0; }
    else if (gain_bits == 1) { mode = 1; }
    else if (gain_bits == 3) { mode = 2; }
    else {
        // gain_bits == 2: undefined / bad pixel
        return 0.0f;
    }

    const float data   = (float)(raw_val & 0x3fff);
    const unsigned long long offset = (unsigned long long)mode * npixels + pixel_idx;
    return (data - peds[offset]) * gmask[offset];
}

} // namespace psana_gpu
