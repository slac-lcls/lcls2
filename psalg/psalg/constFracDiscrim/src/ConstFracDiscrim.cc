//Copyright (C) 2003-2010 Lutz Foucar

/**
 * @file cfd.cpp file contains definition of class that does a constant fraction
 *               descrimination like analysis of a waveform
 *
 * @author Lutz Foucar
 */

#include <cassert>
#include <cmath>
//#include <gsl/gsl_poly.h>

#include "../ConstFracDiscrim.hh"

/** Implematation of Constant Fraction Method
 *
 * The CFD calculates a second trace from the original trace, where two
 * points of the original create one point of the cfd trace
 * \f[new_x = -orig_x*fraction + orig_x-delay \f]
 * It will then find where the new trace will cross the walk value where
 * this point defines the time.
 *
 * @tparam T type of a wavform point
 * @param[in] c the channel that contains the waveform to analyze
 * @param[in] param the user defined parameters for extracting signal in the
 *        waveform
 * @param[out] sig the container with all the found signals
 *
 * @author Lutz Foucar
 */

namespace psalgos {

double getcfd(const double sampleInterval,
              const double horpos,
              const double gain,
              const double offset,
              const Waveform &waveform,
              const int32_t delay,
              const double walk,
              const double threshold,
              const double fraction)
{
  //now extract information from the Channel
  const double sampleIntervalNS = sampleInterval*1e9;    //convert the s to ns
  const double horposNS         = horpos*1.e9;
  const double vGain            = gain;
  const int32_t vOff            = static_cast<int32_t>(offset / vGain);       //V -> ADC Bytes

  const int32_t idxToFiPoint = 0;
  const Waveform& Data = waveform;
  const size_t wLength = waveform.size();

  //--get the right cfd settings--//
  const int32_t delayNS     = static_cast<int32_t>(delay / sampleIntervalNS); //ns -> sampleinterval units
  const double walkB        = walk / vGain;                                 //V -> ADC Bytes
  const double thresholdB   = threshold / vGain;                            //V -> ADC Bytes

  //--go through the waveform--//
  for (size_t i=delay+1; i<wLength-2;++i)
  {
    const double fx  = Data[i] - static_cast<double>(vOff);         //the original Point at i
    const double fxd = Data[i-delay] - static_cast<double>(vOff);   //the delayed Point    at i
    const double fsx = -fx*fraction + fxd;                          //the calculated CFPoint at i

    const double fx_1  = Data[i+1] - static_cast<double>(vOff);        //original Point at i+1
    const double fxd_1 = Data[i+1-delay] - static_cast<double>(vOff);  //delayed Point at i+1
    const double fsx_1 = -fx_1*fraction + fxd_1;                       //calculated CFPoint at i+1

    //check wether the criteria for a Peak are fullfilled
    if (((fsx-walkB) * (fsx_1-walkB)) <= 0 ) //one point above one below the walk
      if (fabs(fx) > thresholdB)             //original point above the threshold
      {
        //--it could be that the first criteria is 0 because  --//
        //--one of the Constant Fraction Signal Points or both--//
        //--are exactly where the walk is                     --//
        if (fabs(fsx-fsx_1) < 1e-8)    //both points are on the walk
        {
          //--go to next loop until at least one is over or under the walk--//
          continue;
        }
        else if ((fsx-walkB) == 0)        //only first is on walk
        {
          //--Only the fist is on the walk, this is what we want--//
          //--so:do nothing--//
        }
        else if ((fsx_1-walkB) == 0)        //only second is on walk
        {
          //--we want that the first point will be on the walk,--//
          //--so in the next loop this point will be the first.--//
          continue;
        }
        //does the peak have the right polarity?//
        //if two pulses are close together then the cfsignal goes through the walk//
        //three times, where only two crossings are good. So we need to check for//
        //the one where it is not good//
        if (fsx     > fsx_1)   //neg polarity
          if (Data[i] > vOff)    //but pos Puls .. skip
            continue;
        if (fsx     < fsx_1)   //pos polarity
          if (Data[i] < vOff)    //but neg Puls .. skip
            continue;


        //--later we need two more points, create them here--//
        const double fx_m1 = Data[i-1] - static_cast<double>(vOff);        //the original Point at i-1
        const double fxd_m1 = Data[i-1-delay] - static_cast<double>(vOff); //the delayed Point    at i-1
        const double fsx_m1 = -fx_m1*fraction + fxd_m1;                    //the calculated CFPoint at i-1

        const double fx_2 = Data[i+2] - static_cast<double>(vOff);         //original Point at i+2
        const double fxd_2 = Data[i+2-delay] - static_cast<double>(vOff);  //delayed Point at i+2
        const double fsx_2 = -fx_2*fraction + fxd_2;                       //calculated CFPoint at i+2

        //--find x with a linear interpolation between the two points--//
        //const double m = fsx_1-fsx;                    //(fsx-fsx_1)/(i-(i+1));
        //const double xLin = i + (walk - fsx)/m;        //PSF fx = (x - i)*m + cfs[i]

        //--find x with a cubic polynomial interpolation between four points--//
        //--do this with the Newtons interpolation Polynomial--//
        const double x[4] = {static_cast<double>(i-1),
                             static_cast<double>(i),
                             static_cast<double>(i+1),
                             static_cast<double>(i+2)};          //x vector
        const double y[4] = {fsx_m1,fsx,fsx_1,fsx_2}; //y vector
        double coeff[4] = {0,0,0,0};                  //Newton coeff vector

        //gsl_poly_dd_init(coeff, x, y, 4);

        if (fabs(coeff[0]) > 1e-8)
        {
          double a = coeff[0];
          for(int i = 0; i < 4; i++)
          {
            coeff[i] /= a;
          }
        }

        double x0, x1, x2 = 0.0;

        //int num_roots = gsl_poly_solve_cubic(coeff[1], coeff[2], coeff[3] - walkB, &x0, &x1, &x2);
        //assert(num_roots == 1);

        //printf("{'Num roots': %d 'Roots': (%f, %f, %f)}\n", num_roots, x0, x1, x2);
        return x0;

        //--numericaly solve the Newton Polynomial--//
        //--give the lineare approach for x as Start Value--//
        // const double xPoly = findXForGivenY<T>(x,coeff,walkB,xLin);
        // const double pos = xPoly + static_cast<double>(idxToFiPoint) + horposNS;

        //--create a new signal--//

        //add the info//
        // signal[time] = pos*sampleIntervalNS;
        // signal[cfd]  = pos*sampleIntervalNS;
        // if (fsx > fsx_1) signal[polarity] = Negative;
        // if (fsx < fsx_1) signal[polarity] = Positive;
        // if (fabs(fsx-fsx_1) < std::sqrt(std::numeric_limits<double>::epsilon()))
        //   signal[polarity] = Bad;

        //--start and stop of the puls--//
        // startstop<T>(c,signal,param.threshold);

        //--height of peak--//
        // getmaximum<T>(c,signal,param.threshold);

        //--width & fwhm of peak--//
        // getfwhm<T>(c,signal,param.threshold);

        //--the com and integral--//
        // CoM<T>(c,signal,param.threshold);

        //--add peak to signal if it fits the conditions--//
        /** @todo make sure that is works right, since we get back a double */
        // if(fabs(signal[polarity]-param.polarity) < std::sqrt(std::numeric_limits<double>::epsilon()))  //if it has the right polarity
        // {
        //   for (CFDParameters::timeranges_t::const_iterator it (param._timeranges.begin());
        //        it != param._timeranges.end();
        //        ++it)
        //   {
        //     if(signal[time] > it->first && signal[time] < it->second) //if signal is in the right timerange
        //     {
        //       signal[isUsed] = false;
        //       sig.push_back(signal);
        //       break;
        //     }
        //   }
        // }
      }
  }

  return 0.0;
}

}
