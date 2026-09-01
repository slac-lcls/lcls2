#include "psalg/detector/FFT.hh"
#include <iostream>

using Complex = std::complex<double>;
using namespace detector;

int main() {
    // Input size MUST be a power of 2
    std::vector<Complex> data = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    
    std::cout << "Original Data:\n";
    for (const auto& c : data) std::cout << c.real() << " ";
    std::cout << "\n\n";

    // Forward FFT (Time Domain -> Frequency Domain)
    fft(data, false);

    std::cout << "FFT Result (Frequency Spectrum):\n";
    for (const auto& c : data) {
        std::cout << "(" << c.real() << ", " << c.imag() << "i)\n";
    }
    std::cout << "\n";

    // Inverse FFT (Frequency Domain -> Time Domain)
    fft(data, true);

    std::cout << "Recovered Data (After IFFT):\n";
    for (const auto& c : data) std::cout << std::round(c.real()) << " ";
    std::cout << "\n";

    return 0;
}


