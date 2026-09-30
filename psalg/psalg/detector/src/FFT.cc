#include "psalg/detector/FFT.hh"

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

const double PI = std::acos(-1.0);

using Complex = std::complex<double>;

namespace detector {

// Iterative FFT implementation (In-place)
void fft(std::vector<Complex>& a, bool invert) {
    int n = a.size();

    // 1. Bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;

        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    // 2. Butterfly diagram computations
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        Complex wlen(std::cos(ang), std::sin(ang));
        
        for (int i = 0; i < n; i += len) {
            Complex w(1);
            for (int j = 0; j < len / 2; j++) {
                Complex u = a[i + j];
                Complex v = a[i + j + len / 2] * w;
                
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    // 3. Scaling for Inverse FFT
    if (invert) {
        for (Complex& x : a) {
            x /= n;
        }
    }
}

}; // namespace detector

