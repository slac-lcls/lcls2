#ifndef __CONSTFRACDISCRIM_H__
#define __CONSTFRACDISCRIM_H__

#include <vector>

namespace psalgos {

typedef std::vector<double> Waveform;

std::vector<double> diff_table(int deg,
                               const std::vector<double>& x,
                               const std::vector<double>& y);

double eval_poly(double x, const std::vector<double> &coeffs);

double find_root(const std::vector<double>& f, const std::vector<double>& df, double error, double x0, int max_its=1000);

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
