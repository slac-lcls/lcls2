#ifndef PSALG_FFT_H
#define PSALG_FFT_H

#include <vector>
#include <complex>

namespace detector {

    void fft(std::vector< std::complex<double> >& a, bool invert);

} // namespace detector

#endif // PSALG_DETECTOR_H

