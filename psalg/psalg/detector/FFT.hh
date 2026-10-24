/**
 * @file
 * @brief Declares detector::fft(), an in-place radix-2 FFT of a complex vector.
 */
#ifndef PSALG_FFT_H
#define PSALG_FFT_H

#include <vector>
#include <complex>

namespace detector {

    /**
     * Replace a with its discrete Fourier transform, computed in place by an iterative radix-2 algorithm (bit-reversal permutation, then butterflies).
     * The forward transform uses twiddle factors exp(+2*pi*i/len); invert=true uses exp(-2*pi*i/len) and divides every element by the size. The size is assumed to be a power of two (not checked).
     */
    void fft(std::vector< std::complex<double> >& a, bool invert);

} // namespace detector

#endif // PSALG_DETECTOR_H

