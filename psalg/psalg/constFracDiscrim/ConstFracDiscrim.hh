#ifndef __CONSTFRACDISCRIM_H__
#define __CONSTFRACDISCRIM_H__

#include <vector>

namespace psalgos {

typedef std::vector<double> Waveform;

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
