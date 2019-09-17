//Copyright (C) 2003-2010 Lutz Foucar

/**
 * @file cfd.cpp file contains definition of class that does a constant fraction
 *               descrimination like analysis of a waveform
 *
 * @author Lutz Foucar
 */

#include <cassert>
#include <cmath>
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

std::vector<double> diff_table(int deg, const std::vector<double>& x, const std::vector<double>& y)
{
  int rows = x.size();

  std::vector<double> table(rows*(deg+1), 0);
  table.insert(table.begin(), y.begin(), y.end());

  std::vector<double> coeffs(1, y[0]);

  for(int col = 0; col < (deg+1); col++)
  {
    for(int row = 0; row < rows-(col+1); row++)
    {
      double a = table[col*rows+row];
      double b = table[col*rows+row+1];
      // printf("%d %d %g\n", row+1, col, b);
      // printf("%d %d %g\n", row, col, a);
      table[(col+1)*rows+row] = (b - a) / (x[row] - x[row+1]);
      // printf("%d %d %g\n\n", row, col+1, table[(col+1)*rows+row]);
      if(row == 0)
      {
        coeffs.push_back(table[(col+1)*rows+row]);
      }
    }
  }

  return coeffs;
}

double eval_poly(double x, const std::vector<double> &coeffs)
{
  double s = 0;
  int p = coeffs.size()-1;

  for(int i = 0; i < p+1; i++)
  {
    s += coeffs[i]*pow(x, p-i);
  }
  return s;
}

double find_root(const std::vector<double>& f, const std::vector<double>& df, double error, double x0, int max_its)
{
  std::vector<double> xs(1, x0);

  int end = 0;
  while(true)
  {
    double fx = eval_poly(xs[end], f);
    double dfx = eval_poly(xs[end], df);
    xs.push_back(xs[end] - fx/dfx);
    end++;
    if (fabs(xs[end] - xs[end-1]) < error)
    {
      return xs[end];
    }
    else if (end-1 > max_its)
      return NAN;
  }
}

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
  //const double sampleIntervalNS = sampleInterval*1e9;    //convert the s to ns
  //const double horposNS         = horpos*1.e9;
  const double vGain            = gain;
  const int32_t vOff            = static_cast<int32_t>(offset / vGain);       //V -> ADC Bytes

  //const int32_t idxToFiPoint = 0;
  const Waveform& Data = waveform;
  const size_t wLength = waveform.size();

  //--get the right cfd settings--//
  //const int32_t delayNS     = static_cast<int32_t>(delay / sampleIntervalNS); //ns -> sampleinterval units
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
        const std::vector<double> x = {static_cast<double>(i-1),
                                       static_cast<double>(i),
                                       static_cast<double>(i+1),
                                       static_cast<double>(i+2)};          //x vector
        const std::vector<double> y = {fsx_m1,fsx,fsx_1,fsx_2}; //y vector

        std::vector<double> coeffs = diff_table(3, x, y);

        if (fabs(coeffs[0]) > 1e-8)
        {
          double a = coeffs[0];
          for(int i = 0; i < 4; i++)
          {
            coeffs[i] /= a;
          }
        }

        coeffs[3] -= walkB;

        std::vector<double> dy;
        for(int i = 0, size = coeffs.size()-1; i < size; i++)
        {
          dy.push_back(coeffs[i]*(size-i));
        }

        double x0 = find_root(coeffs, dy, 1e-7, x[0]);
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
