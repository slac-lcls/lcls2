/**
 * @file
 * @brief Declares psalgos constant-fraction discrimination functions: getcfd() and its polynomial helpers.
 */
#ifndef __CONSTFRACDISCRIM_H__
#define __CONSTFRACDISCRIM_H__

#include <cstddef>
#include <cstdint>
#include <vector>

/** Waveform and peak-finding algorithms of psana (constant-fraction discrimination, PeakFinderAlgos, ...). */
namespace psalgos {

/** std::vector<double> of waveform samples, the input of getcfd(). */
typedef std::vector<double> Waveform;

/**
 * Return y[0] followed by difference quotients taken column by column, each computed as (next - current) / (x[row] - x[row+1]) with adjacent x spacing at every order; the first entry of each column is appended.
 * At most deg+1 values are appended (fewer if x has too few points).
 */
std::vector<double> diff_table(int deg,
                               const std::vector<double>& x,
                               const std::vector<double>& y);

/** Evaluate the polynomial with coefficients coeffs, highest power first, at x. */
double eval_poly(double x, const std::vector<double> &coeffs);

/**
 * Newton iteration from x0 on the polynomial f with derivative polynomial df (both evaluated with eval_poly()).
 * Returns the first iterate that moves by less than error, or NaN after max_its + 2 steps without that.
 */
double find_root(const std::vector<double>& f, const std::vector<double>& df, double error, double x0, int max_its=1000);

/**
 * Scan waveform for the first sample i (from delay+1) where the constant-fraction signal s(k) = -(w[k]-off)*fraction + (w[k-delay]-off) crosses walk/gain between i and i+1, |w[i]-off| > threshold/gain and the polarity matches, with off = (int)(offset/gain).
 * Then take the four points i-1 .. i+2 through diff_table() and find_root() and return that root (in sample-index units), or 0.0 if no sample qualifies. sampleInterval and horpos are not used.
 */
double getcfd(const double sampleInterval,
              const double horpos,
              const double gain,
              const double offset,
              const Waveform &waveform,
              const int32_t delay,
              const double walk,
              const double threshold,
              const double fraction);

};
#endif
