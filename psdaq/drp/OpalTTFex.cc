#include "OpalTTFex.hh"
#include "drp.hh"

#include "xtcdata/xtc/DescData.hh"
#include "psalg/calib/NDArray.hh"
#include "psalg/detector/UtilsConfig.hh"
#include "psalg/utils/SysLog.hh"

#include <list>
#include <math.h>

//#define DBUG

using namespace Drp;
using namespace XtcData;
using psalg::NDArray;
using logging = psalg::SysLog;

enum Cuts { _NCALLS, _NOLASER, _FRAMESIZE, _PROJCUT, 
            _NOBEAM, _NOREF, _NOFITS, _NCUTS };
static const char* cuts[] = {"NCalls",
                             "NoLaser",
                             "FrameSize",
                             "ProjCut",
                             "NoBeam",
                             "NoRef",
                             "NoFits",
                             NULL };

//static ndarray<double,1> load_reference(unsigned key, unsigned sz);
static void              read_roi(Roi& roi, DescData& descdata, const char* name, 
                                  unsigned columns, unsigned rows);
// formerly psalg functions
static std::vector<int>    project_x(NDArray<uint16_t>&, Roi&, unsigned);
static std::vector<int>    project_y(NDArray<uint16_t>&, Roi&, unsigned);
static void            rolling_average(std::vector<int>& a, std::vector<double>& avg, double fraction);
static void            rolling_average(std::vector<double>& a, std::vector<double>& avg, double fraction);
static std::vector<double> finite_impulse_response(std::vector<double>& filter,
                                               std::vector<double>& sample);
static std::list<unsigned> find_peaks(std::vector<double>&, double, unsigned);
static std::vector<double> parab_fit(double* input, unsigned len);
static std::vector<double> parab_fit(double* qwf, unsigned ix, unsigned len, double nxta);

#define MLOOKUP(m,name,dflt) (m.find(name)==m.end() ? dflt : m[name])

OpalTTFex::OpalTTFex(Parameters* para) :
  m_eventcodes_beam_incl (0),
  m_eventcodes_beam_excl (0),
  m_eventcodes_laser_incl(0),
  m_eventcodes_laser_excl(0)
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

  m_ref_avg.resize(0);
  FILE* f = fopen(m_fname.c_str(),"r");
  if (f) {
      double v;
      while( fscanf(f,"%lf",&v)==1 )
          m_ref_avg.push_back(v);
      fclose(f);
      logging::info("OpalTTFex read reference from %s",m_fname.c_str());
  }
}

OpalTTFex::~OpalTTFex() 
{
  // Record accumulated reference
  FILE* f = fopen(m_fname.c_str(),"w");
  if (f) {
      for(unsigned i=0; i<m_ref_avg.size(); i++)
          fprintf(f," %lf",m_ref_avg[i]);
      fprintf(f,"\n");
      fclose(f);
      logging::info("OpalTTFex saved reference to %s",m_fname.c_str());
  }

}

void OpalTTFex::configure(XtcData::ConfigIter& configo,
                          unsigned      columns,
                          unsigned      rows) {
  
  m_columns = columns;
  m_rows    = rows;
  
  XtcData::Names& names = detector::configNames(configo);
  XtcData::DescData& descdata = configo.desc_shape();
  
  for (unsigned i = 0; i < names.num(); i++) {
      XtcData::Name& name = names.get(i);
      int data_rank = name.rank();
      int data_type = name.type();
      printf("%d: '%s' rank %d, type %d\n", i, name.name(), data_rank, data_type);
#define GET_VECTOR(a,b)                                         \
      if (strcmp(name.name(),"fex.eventcodes." #a "." #b)==0) { \
        Array<uint8_t> t = descdata.get_array<uint8_t>(i);      \
        std::vector<uint8_t>& v = m_eventcodes_##a##_##b;       \
          unsigned len = t.num_elem();                          \
          v.resize(len);                                        \
          memcpy(v.data(),t.data(),len);                        \
          printf("m_eventcodes_" #a "_" #b);                    \
          for(unsigned k=0; k<len; k++)                         \
            printf(" %u", v[k]);                                \
          printf("\n");                                         \
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

    int invert = descdata.get_value<int32_t>("fex.invert:boolEnum");
    if (invert) {
      for(unsigned k=0; k<m_fir_weights.size(); k++)
        m_fir_weights[k] = -1.*m_fir_weights[k];
      printf("weights inverted\n");
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
  GET_ENUM(project,axis,axisEnum);
  GET_VALUE(project,minvalue);
  GET_VALUE(prescale,image);
  GET_VALUE(prescale,projections);
#undef GET_VALUE
#define GET_VALUE(a,b) {                                        \
    m_##a##_##b = descdata.get_value<double>("fex." #a "." #b); \
      printf("m_" #a "_" #b " = %f\n", m_##a##_##b);            \
  }
  GET_VALUE(ref,convergence);
  GET_VALUE(sb ,convergence);
#undef GET_VALUE
  m_pedestal = descdata.get_value<unsigned>("user.black_level");

  m_use_ref_roi = descdata.get_value<uint8_t>("fex.ref.enable");
  m_use_sb_roi  = descdata.get_value<uint8_t>("fex.sb.enable");
  read_roi(m_sig_roi, descdata, "fex.sig.roi", m_columns, m_rows);
  read_roi(m_ref_roi, descdata, "fex.ref.roi", m_columns, m_rows);
  read_roi(m_sb_roi , descdata, "fex.sb.roi" , m_columns, m_rows);

  int32_t m_ref_record;
  GET_ENUM(ref,record,recordEnum);
  switch(m_ref_record) {
  case 0: m_record_ref_image=false; m_record_ref_projection=false; break;
  case 1: m_record_ref_image=false; m_record_ref_projection= true; break;
  case 2: m_record_ref_image=true ; m_record_ref_projection=false; break;
  default: break;
  }
  printf("fex.ref.record %u m_record_ref_image %c  m_record_ref_projection %c\n",
         descdata.get_value<int32_t>("fex.ref.record:recordEnum"),
         m_record_ref_image ? 'T':'F',
         m_record_ref_projection ? 'T':'F');
#undef GET_ENUM
#define GET_ENUM(a,b) {                                                 \
      m_##a = descdata.get_value<int32_t>("fex." #a ":" #b);            \
          printf("m_" #a " = %u\n", m_##a);                             \
  }
  //  GET_ENUM(subtractAndNormalize,boolEnum);
  
  m_sb_avg.resize(0);
//  m_ref_avg.resize(0); // reset reference

  m_cut.clear();
  m_cut.resize(_NCUTS,0);

  m_prescale_image_counter = 0;
  m_prescale_projections_counter = 0;
}
      
void OpalTTFex::unconfigure()
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

void OpalTTFex::reset() 
{
  m_flt_position  = 0;
  m_flt_position_ps = 0;
  m_flt_fwhm      = 0;
  m_amplitude     = 0;
  m_ref_amplitude = 0;
  m_nxt_amplitude = -1;
}

OpalTTFex::TTResult OpalTTFex::analyze(std::vector< XtcData::Array<uint8_t> >& subframes)
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
  { const uint32_t* p = reinterpret_cast<const uint32_t*>(&info);
    printf("EventInfo [beam %c]: ",beam?'T':'F');
    for(unsigned i=3; i<11; i++)
      printf("%08x ",p[i]);
    printf("\n");
  }
#endif

  bool nobeam   = !beam;
  bool nolaser  = !laser;

  if (nolaser) { m_cut[_NOLASER]++; return NOLASER; }

  unsigned shape[2];
  shape[0] = m_columns;
  shape[1] = m_rows;
  NDArray<uint16_t> f(shape, 2, subframes[2].data());

#ifdef DBUG
  { const uint16_t* p = reinterpret_cast<const uint16_t*>(subframes[2].data());
    for(unsigned i=0; i<m_rows; i+= 16) {
      for(unsigned j=0; j<m_columns; j+= 16)
        printf(" %04x",p[i*m_columns+j]);
      printf("\n");
    }
  }
#endif

  if (!f.size()) { m_cut[_FRAMESIZE]++; return INVALID; }

  m_prescale_image_counter++;

  //
  //  Project signal ROI
  //
  if (m_project_axis==0) {
    m_sig = project_x(f, m_sig_roi, m_pedestal);
    if (m_use_ref_roi)
      m_ref = project_x(f, m_ref_roi, m_pedestal);
    if (m_use_sb_roi)
      m_sb  = project_x(f, m_sb_roi , m_pedestal);
  }
  else {
    m_sig = project_y(f, m_sig_roi, m_pedestal);
    if (m_use_ref_roi)
      m_ref = project_y(f, m_ref_roi, m_pedestal);
    if (m_use_sb_roi)
      m_sb  = project_y(f, m_sb_roi , m_pedestal);
  }

#ifdef DBUG
  printf("--projection--\n");
  for(unsigned i=0; i<m_sig.size(); i+= 16)
    printf(" %d",m_sig[i]);
  printf("\n");
#endif

  m_prescale_projections_counter++;

  std::vector<double> sigd(m_sig.size());
  std::vector<double> refd(m_sig.size());

  //
  //  Correct projection for common mode found in sideband
  //
  if (m_use_sb_roi) {
    rolling_average(m_sb, m_sb_avg, m_sb_convergence);

    //    ndarray<const double,1> sbc = commonModeLROE(m_sb, m_sb_avg);
    std::vector<double>& sbc = m_sb_avg;

    if (m_use_ref_roi)
      for(unsigned i=0; i<m_sig.size(); i++) {
        sigd[i] = double(m_sig[i])-sbc[i];
        refd[i] = double(m_ref[i])-sbc[i];
      }
    else
      for(unsigned i=0; i<m_sig.size(); i++)
        sigd[i] = double(m_sig[i])-sbc[i];
  }
  else {
    if (m_use_ref_roi)
      for(unsigned i=0; i<m_sig.size(); i++) {
        sigd[i] = double(m_sig[i]);
        refd[i] = double(m_ref[i]);
      }
    else
      for(unsigned i=0; i<m_sig.size(); i++)
        sigd[i] = double(m_sig[i]);
  }

  if (!m_use_ref_roi)
    refd = sigd;

#ifdef DBUG
  printf("--sidebandcorr--\n");
  for(unsigned i=0; i<sigd.size(); i+= 16)
    printf(" %g",sigd[i]);
  printf("\n");
#endif

  //
  //  Require projection has a minimum amplitude (else no laser)
  //
  bool lcut=true;
  for(unsigned i=0; i<sigd.size(); i++)
    if (sigd[i]>m_project_minvalue)
      lcut=false;

  if (lcut) { m_cut[_PROJCUT]++; return INVALID; }

  if (nobeam) {
    _monitor_ref_sig( refd );
    rolling_average(refd, m_ref_avg, m_ref_convergence);

#ifdef DBUG
    printf("--refavg--\n");
    for(unsigned i=0; i<sigd.size(); i+= 16)
      printf(" %g",sigd[i]);
    printf("\n");
#endif

    m_cut[_NOBEAM]++;
    return NOBEAM;
  }
  else if (m_use_ref_roi) {
    _monitor_ref_sig( refd );
    rolling_average(refd, m_ref_avg, m_ref_convergence);
  }

  _monitor_raw_sig( sigd );

  if (m_ref_avg.size()==0) {
    m_cut[_NOREF]++;
    return INVALID;
  }

  //
  //  Divide by the reference
  //
  for(unsigned i=0; i<sigd.size(); i++)
    sigd[i] = sigd[i]/m_ref_avg[i] - 1;

#ifdef DBUG
    printf("--ratio--\n");
    for(unsigned i=0; i<sigd.size(); i+= 16)
      printf(" %g",sigd[i]);
    printf("\n");
#endif

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
      double   xflt = pFit0[1]+(m_project_axis==0 ? m_sig_roi.x0 : m_sig_roi.y0);

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
    return INVALID;
  }
  return VALID;
}

//
//  Ideally, each thread's 'm_ref' array would reference the
//  same ndarray, but then I would need to control exclusive
//  access during the reference updates
//
/*
void OpalTTFex::_monitor_raw_sig (std::vector<double>& a) 
{
  _etype = TimeToolDataType::Signal;
  MapType::iterator it = _ref.find(_src);
  if (it != _ref.end()) {
    if (m_ref_avg.size()!=it->second.size())
      m_ref_avg = std::vector<double>(it->second.size());
    std::copy(it->second.begin(), it->second.end(), m_ref_avg.begin());
  }
}

void OpalTTFex::_monitor_ref_sig (std::vector<double>& ref) 
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
void read_roi(Roi& roi, DescData& descdata, const char* name, unsigned columns, unsigned rows)
{
  roi.x0 = descdata.get_value<unsigned>((std::string(name)+=".x0").c_str());
  roi.y0 = descdata.get_value<unsigned>((std::string(name)+=".y0").c_str());
  roi.x1 = descdata.get_value<unsigned>((std::string(name)+=".x1").c_str());
  roi.y1 = descdata.get_value<unsigned>((std::string(name)+=".y1").c_str());

  std::string msg;

  if (roi.x1 >= columns) {
    std::stringstream s;
    s << "Timetool: " << name << ".roi.x1[" << roi.x1
      << "] exceeds frame columns [" << columns 
      << "].  Truncating.";
    msg += s.str();
    roi.x1 = columns-1;
  }
  if (roi.y1 >= rows) {
    std::stringstream s;
    s << "Timetool: " << name << " roi.y1[" << roi.y1 
      << "] exceeds frame rows [" << rows 
      << "].  Truncating.";
    msg += s.str();
    roi.y1 = rows-1;
  }

  if (!msg.empty())
    printf("OpalTTFex::configure: %s\n", msg.c_str());
}


std::vector<int> project_x(NDArray<uint16_t>& f, 
                           Roi& roi,
                           unsigned ped)
{
#ifdef DBUG
  printf("proj_x roi [%u,%u],[%u,%u]\n",
         roi.x0,roi.x1,roi.y0,roi.y1);
#endif
  std::vector<int> result(roi.x1-roi.x0+1);
  for(unsigned i=0; i<result.size(); i++) result[i]=-ped*(roi.y1-roi.y0+1);
  for(unsigned i=roi.y0; i<=roi.y1; i++) {
    for(unsigned j=roi.x0, k=0; j<=roi.x1; j++,k++)
      result[k] += f(i,j);
  }
  return result;
}

std::vector<int> project_y(NDArray<uint16_t>& f, 
                           Roi& roi,
                           unsigned ped)
{
  std::vector<int> result(roi.y1-roi.y0+1);
  for(unsigned i=roi.y0,k=0; i<=roi.y1; i++,k++) {
    int sum=0;
    for(unsigned j=roi.x0; j<=roi.x1; j++)
      sum += f(i,j);
    result[k] = sum - ped*(roi.x1-roi.x0+1);
  }
  return result;
}

void rolling_average(std::vector<int>& a, std::vector<double>& avg, double fraction)
{
  if (avg.size()==0) {
    avg.resize(a.size());
    for(unsigned i=0; i<a.size(); i++)
      avg[i] = a[i];
  }
  else {
    double g = (1-fraction);
    double f = fraction;
    for(unsigned i=0; i<a.size(); i++)
      avg[i] = avg[i]*g + double(a[i])*f;
  }
} 

void rolling_average(std::vector<double>& a, std::vector<double>& avg, double fraction)
{
  if (avg.size()==0) {
    avg = a;
  }
  else {
    double g = (1-fraction);
    double f = fraction;
    for(unsigned i=0; i<a.size(); i++)
      avg[i] = avg[i]*g + a[i]*f;
  }
} 

std::vector<double> finite_impulse_response(std::vector<double>& filter,
                                            std::vector<double>& sample)
{
  unsigned nf = filter.size();
  if (sample.size()<filter.size())
    return std::vector<double>(0);
  else {
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

void OpalTTFex::_monitor_raw_sig (std::vector<double>& a) 
{
#ifdef DBUG
  printf("raw_sig: ");
  for(unsigned i=0; i<a.size(); i+=16)
    printf("%g ",a[i]);
  printf("\n");
#endif
}
void OpalTTFex::_monitor_ref_sig (std::vector<double>& a)
{
#ifdef DBUG
  printf("ref_sig: ");
  for(unsigned i=0; i<a.size(); i+=16)
    printf("%g ",a[i]);
  printf("\n");
#endif
}
void OpalTTFex::_monitor_sub_sig (std::vector<double>& a)
{
#ifdef DBUG
  printf("sub_sig: ");
  for(unsigned i=0; i<a.size(); i+=16)
    printf("%g ",a[i]);
  printf("\n");
#endif
}
void OpalTTFex::_monitor_flt_sig (std::vector<double>& a)
{
#ifdef DBUG
  printf("flt_sig: ");
  for(unsigned i=0; i<a.size(); i+=16)
    printf("%g ",a[i]);
  printf("\n");
#endif
}

bool   OpalTTFex::write_evt_image      ()  
{
  bool r = m_prescale_image_counter==m_prescale_image;
  if (m_prescale_image_counter >= m_prescale_image)
    m_prescale_image_counter = 0;
  return r;
}
bool   OpalTTFex::write_evt_projections() 
{
  bool r = m_prescale_projections_counter==m_prescale_projections;
  if (m_prescale_projections_counter >= m_prescale_projections)
    m_prescale_projections_counter = 0;
  return r;
}

      
