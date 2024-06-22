#include "Piranha4TTFex.hh"
#include "drp.hh"

#include "xtcdata/xtc/DescData.hh"
#include "psalg/calib/NDArray.hh"
#include "psalg/detector/UtilsConfig.hh"
#include "psalg/utils/SysLog.hh"

#include <list>
#include <math.h>

//#define DBUG
//#define DBUG2

using namespace Drp;
using namespace XtcData;
using psalg::NDArray;
using logging = psalg::SysLog;

enum Cuts { _NCALLS, _NOLASER, _FRAMESIZE, _ROICUT,
            _NOBEAM, _NOREF, _NOFITS, _NCUTS };
static const char* cuts[] = {"NCalls",
                             "NoLaser",
                             "FrameSize",
                             "RoiCut",
                             "NoBeam",
                             "NoRef",
                             "NoFits",
                             NULL };

//static ndarray<double,1> load_reference(unsigned key, unsigned sz);
static void                read_roi(Roi& roi, DescData& descdata, const char* name,
                                    unsigned pixels);
// formerly psalg functions
static std::vector<int>    extract_roi(NDArray<uint16_t>&, Roi&, unsigned);
//static void                rolling_average(std::vector<int>& a,
//                                           std::vector<double>& avg,
//                                           double fraction);
static void                rolling_average(std::vector<double>& a,
                                           std::vector<double>& avg,
                                           double fraction);
//static void                rolling_average(NDArray<double>& a,
//                                           NDArray<double>& avg,
//                                           double fraction);
static std::vector<double> finite_impulse_response(std::vector<double>& filter,
                                                   std::vector<double>& sample);
static std::list<unsigned> find_peaks(std::vector<double>&, double, unsigned);
static std::vector<double> parab_fit(double* input, unsigned len);
static std::vector<double> parab_fit(double* qwf, unsigned ix, unsigned len, double nxta);

#define MLOOKUP(m,name,dflt) (m.find(name)==m.end() ? dflt : m[name])

static const unsigned no_shape[] = {0,0};

Piranha4TTFex::Piranha4TTFex(Parameters* para) :
  m_eventcodes_beam_incl (0),
  m_eventcodes_beam_excl (0),
  m_eventcodes_laser_incl(0),
  m_eventcodes_laser_excl(0),
  m_sig_avg_sem          (Pds::Semaphore::FULL),
  m_ref_avg_sem          (Pds::Semaphore::FULL)
{
  std::string fname = MLOOKUP(para->kwargs,"ttreffile",
                              para->detName+".ttref");
  if (fname[0]=='/') {
      m_fname = fname;
  }
  else {
      const char* dir = getenv("HOME");
      m_fname = std::string(dir ? dir : "/tmp") + "/" + fname;
  }

  m_sig_avg.resize(0);
  m_ref_avg.resize(0);

  FILE* f = fopen(m_fname.c_str(),"r");
  if (f) {
      double v;
      while( fscanf(f,"%lf",&v)==1 )
          m_ref_avg.push_back(v);
      fclose(f);
      logging::info("Piranha4TTFex read reference from %s",m_fname.c_str());
  }
}

Piranha4TTFex::~Piranha4TTFex()
{
  // Record accumulated reference
  FILE* f = fopen(m_fname.c_str(),"w");
  if (f) {
      for(unsigned i=0; i<m_ref_avg.size(); i++)
          fprintf(f," %lf",m_ref_avg[i]);
      fprintf(f,"\n");
      fclose(f);
      logging::info("Piranha4TTFex saved reference to %s",m_fname.c_str());
  }
}

void Piranha4TTFex::configure(XtcData::ConfigIter& configo,
                              unsigned             pixels) {
  m_pixels = pixels;

  XtcData::Names& names = detector::configNames(configo);
  XtcData::DescData& descdata = configo.desc_shape();
  IndexMap& nameMap = descdata.nameindex().nameMap();

  for (unsigned i = 0; i < names.num(); i++) {
      XtcData::Name& name = names.get(i);
      int data_rank = name.rank();
      int data_type = name.type();
      printf("%d: '%s' rank %d, type %d\n", i, name.name(), data_rank, data_type);
#define GET_VECTOR(a,b)                                          \
      if (strcmp(name.name(),"fex.eventcodes." #a "." #b)==0) {  \
          Array<uint16_t> t = descdata.get_array<uint16_t>(i);   \
          std::vector<uint16_t>& v = m_eventcodes_##a##_##b;     \
              unsigned len = t.num_elem();                       \
              v.resize(len);                                     \
              memcpy(v.data(),t.data(),len*sizeof(uint16_t));    \
              printf("m_eventcodes_" #a "_" #b);                 \
              for(unsigned k=0; k<len; k++)                      \
                  printf(" %u", v[k]);                           \
              printf("\n");                                      \
      }
      GET_VECTOR(beam,incl);
      GET_VECTOR(beam,excl);
      GET_VECTOR(laser,incl);
      GET_VECTOR(laser,excl);
#undef GET_VECTOR
#define GET_VECTOR(a)                                           \
      if (strcmp(name.name(),"fex." #a)==0) {                   \
        Array<double> t = descdata.get_array<double>(i);        \
        unsigned len = t.num_elem();                            \
        std::vector<double> v(len);                             \
        memcpy(v.data(),t.data(),len*sizeof(double));           \
        printf("m_" #a);                                        \
        for(unsigned k=0; k<len; k++)                           \
          printf(" %f", v[k]);                                  \
        printf("\n");                                           \
        m_##a = v;                                              \
      }
      GET_VECTOR(fir_weights);
      GET_VECTOR(calib_poly);
#undef GET_VECTOR

    {
      if (nameMap.find("fex.invert_weights:boolEnum") != nameMap.end()) {
        int invert = descdata.get_value<int32_t>("fex.invert_weights:boolEnum");
        if (invert) {
          for(unsigned k=0; k<m_fir_weights.size(); k++)
            m_fir_weights[k] = -1.*m_fir_weights[k];
          printf("weights inverted\n");
        }
      }
    }
  }

#define GET_ENUM(a,b,c) {                                                \
    m_##a##_##b = descdata.get_value<int32_t>("fex." #a "." #b ":" #c); \
      printf("m_" #a "_" #b " = %u\n", m_##a##_##b);                    \
  }
#define GET_VALUE(a,b) {                                                \
    m_##a##_##b = descdata.get_value<unsigned>("fex." #a "." #b);       \
      printf("m_" #a "_" #b " = %u\n", m_##a##_##b);                    \
  }
  GET_VALUE(signal,minvalue);
  GET_VALUE(prescale,image);
  GET_VALUE(prescale,averages);
#undef GET_VALUE
#define GET_VALUE(a,b,v) {                                      \
    if (nameMap.find("fex." #a "." #b) != nameMap.end()) {      \
      m_##a##_##b = descdata.get_value<double>("fex." #a "." #b);     \
      printf("m_" #a "_" #b " = %f\n", m_##a##_##b);            \
    } else { m_##a##_##b = v; } \
  }
  GET_VALUE(sig,convergence,1.);
  GET_VALUE(ref,convergence,1.);
#undef GET_VALUE
  m_pedestal = descdata.get_value<int>("user.black_level");
  if (nameMap.find("fex.pedestal_adj") != nameMap.end()) {
    m_pedestal -= descdata.get_value<int>("fex.pedestal_adj");
    printf("m_pedestal adjusted to subtract %d\n",m_pedestal);
  }

  read_roi(m_sig_roi, descdata, "fex.sig.roi", m_pixels);

  int32_t m_ref_record;
  GET_ENUM(ref,record,recordEnum);
  switch(m_ref_record) {
  case 0: m_record_ref_image=false; m_record_ref_average=false; break;
  case 1: m_record_ref_image=false; m_record_ref_average= true; break;
  case 2: m_record_ref_image=true ; m_record_ref_average=false; break;
  default: break;
  }
  printf("fex.ref.record %u m_record_ref_image %c  m_record_ref_average %c\n",
         descdata.get_value<int32_t>("fex.ref.record:recordEnum"),
         m_record_ref_image ? 'T':'F',
         m_record_ref_average ? 'T':'F');
#undef GET_ENUM
#define GET_ENUM(a,b) {                                                 \
      m_##a = descdata.get_value<int32_t>("fex." #a ":" #b);            \
          printf("m_" #a " = %u\n", m_##a);                             \
  }
  //  GET_ENUM(subtractAndNormalize,boolEnum);

//  m_ref_avg.resize(0); // reset reference

  m_cut.clear();
  m_cut.resize(_NCUTS,0);

  m_prescale_image_counter = 0;
  m_prescale_averages_counter = 0;

#ifdef DBUG
  printf("incl_beam size %zd  excl_beam size %zd\n",
         m_eventcodes_beam_incl.size(),
         m_eventcodes_beam_excl.size());
  printf("incl_laser size %zd  excl_laser size %zd\n",
         m_eventcodes_laser_incl.size(),
         m_eventcodes_laser_excl.size());
#endif

}

void Piranha4TTFex::unconfigure()
{
  if (m_cut.size() && m_cut[_NCALLS]>0) {
    printf("TimeTool::Fex Summary\n");
    for(unsigned i=0; i<_NCUTS; i++)
      printf("%s: %3.2f [%u]\n",
             cuts[i],
             double(m_cut[i])/double(m_cut[_NCALLS]),
             m_cut[i]);
  }
}

void Piranha4TTFex::reset()
{
  m_flt_position  = 0;
  m_flt_position_ps = 0;
  m_flt_fwhm      = 0;
  m_amplitude     = 0;
  m_ref_amplitude = 0;
  m_nxt_amplitude = -1;
}

Piranha4TTFex::TTResult Piranha4TTFex::analyze(std::vector< XtcData::Array<uint8_t> >& subframes,
                                               std::vector<double>& sigd,
                                               std::vector<double>& refout)
{
  m_cut[_NCALLS]++;

  //  EventInfo is in subframe 3
  const EventInfo& info = *reinterpret_cast<const EventInfo*>(subframes[3].data());


  bool beam = true;
  for(unsigned i=0; i<m_eventcodes_beam_incl.size(); i++)
    beam &= info.eventCode(m_eventcodes_beam_incl[i]);
  for(unsigned i=0; i<m_eventcodes_beam_excl.size(); i++)
    beam &= !info.eventCode(m_eventcodes_beam_excl[i]);

  bool laser = true;
  for(unsigned i=0; i<m_eventcodes_laser_incl.size(); i++)
    laser &= info.eventCode(m_eventcodes_laser_incl[i]);
  for(unsigned i=0; i<m_eventcodes_laser_excl.size(); i++)
    laser &= !info.eventCode(m_eventcodes_laser_excl[i]);

#ifdef DBUG
  {
    printf("EventInfo [beam %c laser %c]:",beam?'T':'F',laser?'T':'F');
    printf(" LaserIncl:");
    for(unsigned i=0; i<m_eventcodes_laser_incl.size(); i++)
        printf(" %u",m_eventcodes_laser_incl[i]);
    printf(" LaserExcl:");
    for(unsigned i=0; i<m_eventcodes_laser_excl.size(); i++)
        printf(" %u",m_eventcodes_laser_excl[i]);
    printf("\n");
    for(unsigned i=0; i<18*16; i++)
        if (info.eventCode(i))
            printf(" %u", i);
    printf("\n");
  }
#endif

  bool nobeam   = !beam;
  bool nolaser  = !laser;

  if (nolaser) {
      m_cut[_NOLASER]++;
#ifdef DBUG
      printf("-->NOLASER\n");
#endif
      return NOLASER;
  }

  unsigned shape[MaxRank];
  shape[0] = m_pixels;
  NDArray<uint16_t> f(shape, 1, subframes[2].data());

#ifdef DBUG2
  { const uint16_t* p = reinterpret_cast<const uint16_t*>(subframes[2].data());
    for(unsigned j=0; j<m_pixels; j+= 16)
      printf(" %04x",p[j]);
    printf("\n");
  }
#endif

  if (!f.size()) {
      m_cut[_FRAMESIZE]++;
#ifdef DBUG
      printf("-->INVALID1\n");
#endif
      return INVALID;
  }

  m_prescale_image_counter++;

  //
  //  Extract signal ROI
  //
  std::vector<int> m_sig = extract_roi(f, m_sig_roi, m_pedestal);

  m_prescale_averages_counter++;

  sigd.resize(m_sig.size());
  std::vector<double> refd(m_sig.size());

  // If the size stored in the file is out of date,
  // resetting the size to 0 here will cause a new m_ref_avg
  // to be initialized in rolling_average
  // (and saved to a new file)
  if (sigd.size() != m_ref_avg.size()) {
      m_ref_avg_sem.take();
      m_ref_avg.resize(0);
      m_ref_avg_sem.give();
  }

  for(unsigned i=0; i<m_sig.size(); i++)
      sigd[i] = double(m_sig[i]);

  refd = sigd;

  //
  //  Require ROI to have a minimum amplitude (else no laser)
  //
  bool lcut=true;
  for(unsigned i=0; i<sigd.size(); i++)
      if (sigd[i]>m_signal_minvalue)
          lcut=false;

  if (lcut) {
      m_cut[_ROICUT]++;
#ifdef DBUG
      printf("-->INVALID2\n");
#endif
      return INVALID;
  }

  if (nobeam) {

      // For the reference, the signal when NOBEAM is used
      _monitor_ref_sig( refd );
      m_ref_avg_sem.take();
      rolling_average(refd, m_ref_avg, m_ref_convergence);
      m_ref_avg_sem.give();

#ifdef DBUG
      printf("--refavg--\n");
      for(unsigned i=0; i<sigd.size(); i+= 16)
          printf(" %g",sigd[i]);
      printf("\n");
#endif

      m_cut[_NOBEAM]++;
#ifdef DBUG
      printf("-->NOBEAM\n");
#endif
      return NOBEAM;
  }

  _monitor_raw_sig( sigd );

  if (m_ref_avg.size()==0) {
      //  Fake a reference once
      m_ref_avg_sem.take();
      m_ref_avg.resize(refd.size());
      double avg = 0;
      for(unsigned i=0; i<refd.size(); i++)
          avg += refd[i];
      avg /= double(refd.size());
      for(unsigned i=0; i<refd.size(); i++)
          m_ref_avg[i] = avg;
      m_ref_avg_sem.give();
      //
      m_cut[_NOREF]++;
#ifdef DBUG
      printf("-->NOREF\n");
#endif
      return INVALID;
  }

  //
  //  Average the signal
  //
  m_sig_avg_sem.take();
  rolling_average(sigd, m_sig_avg, m_sig_convergence);
  sigd = m_sig_avg;
  m_sig_avg_sem.give();

  //
  //  Divide by the reference
  //
  m_ref_avg_sem.take();
  for(unsigned i=0; i<sigd.size(); i++) {
      sigd[i] = sigd[i]/m_ref_avg[i] - 1;
  }
  refout = m_ref_avg;
  m_ref_avg_sem.give();

  _monitor_sub_sig( sigd );

  //
  //  Apply the digital filter
  //
  std::vector<double> qwf = finite_impulse_response(m_fir_weights,sigd);

  _monitor_flt_sig( qwf );

#ifdef DBUG
    printf("--filtered--\n");
    for(unsigned i=0; i<sigd.size(); i+= 16)
      printf(" %g",sigd[i]);
    printf("\n");
#endif

  //
  //  Find the two highest peaks that are well-separated
  //
  const double afrac = 0.50;
  std::list<unsigned> peaks = find_peaks(qwf, afrac, 2);

  unsigned nfits = peaks.size();
  if (nfits>0) {
    unsigned ix = *peaks.begin();
    std::vector<double> pFit0 = parab_fit(qwf.data(),ix,qwf.size(),0.8);
    if (pFit0[2]>0) {
      double   xflt = pFit0[1]+m_sig_roi.x0;

      double  xfltc = 0;
      for(unsigned i=m_calib_poly.size(); i!=0; )
        xfltc = xfltc*xflt + m_calib_poly[--i];

      m_amplitude        = pFit0[0];
      m_flt_position     = xflt;
      m_flt_position_ps  = xfltc;
      m_flt_fwhm         = pFit0[2];
      m_ref_amplitude    = m_ref_avg[ix];

      if (nfits>1) {
        std::vector<double> pFit1 =
          parab_fit(qwf.data(),*(++peaks.begin()),qwf.size(),0.8);
        if (pFit1[2]>0)
          m_nxt_amplitude = pFit1[0];
      }
    }
  }
  else {
    m_cut[_NOFITS]++;
#ifdef DBUG
    printf("-->NOFITS\n");
#endif
    return INVALID;
  }

#ifdef DBUG
    printf("-->VALID\n");
#endif
  return VALID;
}

//
//  Ideally, each thread's 'm_ref' array would reference the
//  same ndarray, but then I would need to control exclusive
//  access during the reference updates
//
/*
void Piranha4TTFex::_monitor_raw_sig (std::vector<double>& a)
{
  _etype = TimeToolDataType::Signal;
  MapType::iterator it = _ref.find(_src);
  if (it != _ref.end()) {
    if (m_ref_avg.size()!=it->second.size())
      m_ref_avg = std::vector<double>(it->second.size());
    std::copy(it->second.begin(), it->second.end(), m_ref_avg.begin());
  }
}

void Piranha4TTFex::_monitor_ref_sig (std::vector<double>& ref)
{
  _etype = TimeToolDataType::Reference;
  MapType::iterator it = _ref.find(_src);
  if (it == _ref.end()) {
    ndarray<double> a(ref.shape(), 1);
    std::copy(ref.begin(), ref.end(), a.begin());
    _sem.take();
    _ref[_src] = a;
    _sem.give();
  }
  else {
    _sem.take();
    psalg::rolling_average(ref, it->second, m_ref_convergence);
    _sem.give();
  }
}
*/
void read_roi(Roi& roi, DescData& descdata, const char* name, unsigned pixels)
{
  roi.x0 = descdata.get_value<unsigned>((std::string(name)+=".x0").c_str());
  roi.x1 = descdata.get_value<unsigned>((std::string(name)+=".x1").c_str());

  std::string msg;

  if (roi.x1 >= pixels) {
    std::stringstream s;
    s << "Timetool: " << name << ".roi.x1[" << roi.x1
      << "] exceeds frame pixels [" << pixels
      << "].  Truncating.";
    msg += s.str();
    roi.x1 = pixels-1;
  }

  if (!msg.empty())
    printf("Piranha4TTFex::configure: %s\n", msg.c_str());
}


std::vector<int> extract_roi(NDArray<uint16_t>& f,
                             Roi& roi,
                             unsigned ped)
{
#ifdef DBUG2
  printf("extract roi [%u,%u]\n",
         roi.x0,roi.x1);
#endif
  std::vector<int> result(roi.x1-roi.x0+1);
  for(unsigned i=0; i<result.size(); i++) result[i]=-ped;
  for(unsigned j=roi.x0, k=0; j<=roi.x1; j++,k++)
    result[k] += f(j);
  return result;
}

void rolling_average(std::vector<int>& a, std::vector<double>& avg, double fraction)
{
  if (avg.size()==0) {
    avg.resize(a.size());
    for(unsigned i=0; i<a.size(); i++)
      avg[i] = a[i];
  } else if (avg.size()!=a.size()) {
    logging::critical("rolling average (int/double) with different sizes");
    throw std::string("rolling average with different sizes");
  } else {
    double g = (1-fraction);
    double f = fraction;
    for(unsigned i=0; i<a.size(); i++)
      avg[i] = avg[i]*g + double(a[i])*f;
  }
}

void rolling_average(std::vector<double>& a, std::vector<double>& avg, double fraction)
{
  if (avg.size()==0) {
    avg.resize(a.size());
    for(unsigned i=0; i<a.size(); i++)
      avg[i] = a[i];
  } else if (avg.size()!=a.size()) {
    logging::critical("rolling average (double/double) with different sizes");
    throw std::string("rolling average with different sizes");
  } else {
    double g = (1-fraction);
    double f = fraction;
    for(unsigned i=0; i<a.size(); i++)
      avg[i] = avg[i]*g + a[i]*f;
  }
}

//void rolling_average(NDArray<double>& a, NDArray<double>& avg, double fraction)
//{
//  if (avg.size()==0)
//    avg = a;
//  else if (avg.size()!=a.size()) {
//      throw std::string("rolling average with different sizes");
//  }
//  else {
//    double g = (1-fraction);
//    double f = fraction;
//    const size_t sz = a.size();
//    double* ad   = a  .data();
//    double* avgd = avg.data();
//    for(unsigned i=0; i<sz; i++)
//      avgd[i] = avgd[i]*g + ad[i]*f;
//  }
//}

std::vector<double> finite_impulse_response(std::vector<double>& filter,
                                            std::vector<double>& sample)
{
  unsigned nf = filter.size();
  if (sample.size()<filter.size()) {
    logging::critical("Piranha4TTFex sample size %i smaller than filter size %i", sample.size(), filter.size());
    throw("Error Piranha4TTFex sample size too small");
  } else {
    unsigned len = sample.size()-nf;
    std::vector<double> result = std::vector<double>(len);
    for(unsigned i=0; i<len; i++) {
      double v = 0;
      for(unsigned j=0; j<nf; j++)
        v += sample[i+j]*filter[j];
      result[i] = v;
    }
    return result;
  }
}

std::list<unsigned> find_peaks(std::vector<double>& a,
                               double afrac,
                               unsigned max_peaks)
{
  std::list<unsigned> peaks;

  double amax    = a[0];
  double aleft   = amax;
  double aright  = 0;
  unsigned imax  = 0;

  bool lpeak = false;

  for(unsigned i=1; i<a.size(); i++) {
    if (a[i] > amax) {
      amax = a[i];
      double af = afrac*amax;
      if (af > aleft) {
        imax = i;
        lpeak  = true;
        aright = af;
      }
    }
    else if (lpeak && a[i] < aright) {
      if (peaks.size()==max_peaks && a[peaks.back()]>amax)
        ;
      else {
        if (peaks.size()==max_peaks)
          peaks.pop_back();

        int sz = peaks.size();
        for(std::list<unsigned>::iterator it=peaks.begin(); it!=peaks.end(); it++)
          if (a[*it]<amax) {
            peaks.insert(it,imax);
            break;
          }
        if (sz == int(peaks.size()))
          peaks.push_back(imax);
      }

      lpeak = false;
      amax  = aleft = (a[i]>0 ? a[i] : 0);
    }
    else if (!lpeak && a[i] < aleft) {
      amax = aleft = (a[i] > 0 ? a[i] : 0);
    }
  }
  return peaks;
}

std::vector<double> parab_fit(double* input, unsigned len)
{
  std::vector<double> result(3);

  double xx[5], xy[3];
  memset(xx,0,5*sizeof(double));
  memset(xy,0,3*sizeof(double));

  for(unsigned ix=0; ix<len; ix++) {
    double x = double(ix);
    double qx=x;
    double y = input[ix];
    xx[0] += 1;
    xy[0] += y;
    xx[1] += x;
    xy[1] += (y*=x);
    xx[2] += (qx*=x);
    xy[2] += y*x;
    xx[3] += (qx*=x);
    xx[4] += qx*x;
  }

  double a11 = xx[0];
  double a21 = xx[1];
  double a31 = xx[2];
  double a22 = xx[2];
  double a32 = xx[3];
  double a33 = xx[4];

  double b11 = a22*a33-a32*a32;
  double b21 = a21*a33-a32*a31;
  double b31 = a21*a32-a31*a22;
  double b22 = a11*a33-a31*a31;
  double b32 = a11*a32-a21*a31;
  double b33 = a11*a22-a21*a21;

  double det = a11*b11 - a21*b21 + a31*b31;

  if (det==0) {
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
  }
  else {
    result[0] = ( b11*xy[0] - b21*xy[1] + b31*xy[2])/det;
    result[1] = (-b21*xy[0] + b22*xy[1] - b32*xy[2])/det;
    result[2] = ( b31*xy[0] - b32*xy[1] + b33*xy[2])/det;
  }
  return result;
}

std::vector<double> parab_fit(double* input,
                              unsigned ix,
                              unsigned len,
                              double afrac)
{
  enum { Amplitude, Position, FWHM, NParms };
  std::vector<double> _p(NParms);

  const double trf = afrac*input[ix];
  int ix_left(ix);
  while(--ix_left > 0) {
    if (input[ix_left] < trf)
      break;
  }

  int ix_right(ix);
  while(++ix_right < int(len)) {
    if (input[ix_right] < trf)
      break;
  }

  std::vector<double> a = parab_fit(&input[ix_left],ix_right-ix_left+1);

  if (a[2] < 0) {  // a maximum
    _p[Amplitude] = a[0] - 0.2*a[1]*a[1]/a[2];
    _p[Position ] = double(ix_left)-0.5*a[1]/a[2];

    const double hm = 0.5*input[ix];
    while(input[ix_left]>hm && ix_left>0)
      ix_left--;
    while(input[ix_right]>hm && ix_right<int(len-1))
      ix_right++;

    _p[FWHM] = sqrt(-2*_p[Amplitude]/a[2]);
  }
  else {
    _p[Amplitude] = -1;
    _p[Position ] = -1;
    _p[FWHM     ] = -1;
  }
  return _p;
}

void Piranha4TTFex::_monitor_raw_sig (std::vector<double>& a)
{
#ifdef DBUG2
  printf("raw_sig: ");
  for(unsigned i=0; i<a.size(); i+=16)
    printf("%g ",a[i]);
  printf("\n");
#endif
}
void Piranha4TTFex::_monitor_ref_sig (std::vector<double>& a)
{
#ifdef DBUG2
  printf("ref_sig: ");
  for(unsigned i=0; i<a.size(); i+=16)
    printf("%g ",a[i]);
  printf("\n");
#endif
}
void Piranha4TTFex::_monitor_sub_sig (std::vector<double>& a)
{
#ifdef DBUG2
  printf("sub_sig: ");
  for(unsigned i=0; i<a.size(); i+=16)
    printf("%g ",a[i]);
  printf("\n");
#endif
}
void Piranha4TTFex::_monitor_flt_sig (std::vector<double>& a)
{
#ifdef DBUG2
  printf("flt_sig: ");
  for(unsigned i=0; i<a.size(); i+=16)
    printf("%g ",a[i]);
  printf("\n");
#endif
}

bool   Piranha4TTFex::write_evt_image      ()
{
  return true;
  bool r = m_prescale_image_counter>=m_prescale_image;
  if (m_prescale_image_counter >= m_prescale_image)
    m_prescale_image_counter = 0;
  return r;
}
bool   Piranha4TTFex::write_evt_averages()
{
  return false;
  bool r = (m_prescale_averages>0) && m_prescale_averages_counter>=m_prescale_averages;
  if (r)
    m_prescale_averages_counter = 0;
  return r;
}
