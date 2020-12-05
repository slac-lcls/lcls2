#include "Opal.hh"
#include "OpalTTFex.hh"
#include "psdaq/service/Semaphore.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psalg/utils/SysLog.hh"
#include "psalg/calib/NDArray.hh"
#include "psalg/detector/UtilsConfig.hh"

#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <fcntl.h>
#include <stdint.h>

using namespace XtcData;
using logging = psalg::SysLog;
using json = nlohmann::json;

//#define DBUG

namespace Drp {

  class RawDef : public VarDef
  {
  public:
    enum index { image };
    RawDef() { NameVec.push_back({"image", Name::UINT16, 2}); }
  } rawDef;

  class FexDef : public VarDef
  {
  public:
    enum index {
      ampl, fltpos, fltpos_ps, fltposfwhm, nxtampl, refampl, proj_sig, proj_ref
    };

    FexDef()
    {
        NameVec.push_back({"ampl"      , Name::DOUBLE});
        NameVec.push_back({"fltpos"    , Name::DOUBLE});
        NameVec.push_back({"fltpos_ps" , Name::DOUBLE});
        NameVec.push_back({"fltposfwhm", Name::DOUBLE});
        NameVec.push_back({"amplnxt"   , Name::DOUBLE});
        NameVec.push_back({"refampl"   , Name::DOUBLE});
    }

    FexDef(const OpalTTFex& fex)
    {
      NameVec.push_back({"ampl"      , Name::DOUBLE});
      NameVec.push_back({"fltpos"    , Name::DOUBLE});
      NameVec.push_back({"fltpos_ps" , Name::DOUBLE});
      NameVec.push_back({"fltposfwhm", Name::DOUBLE});
      NameVec.push_back({"amplnxt"   , Name::DOUBLE});
      NameVec.push_back({"refampl"   , Name::DOUBLE});
      if (fex.write_projections()) {
        NameVec.push_back({"proj_sig"   , Name::INT32 , 1});
        NameVec.push_back({"proj_ref"   , Name::DOUBLE, 1});
      }
    }
  };

  class RefDef : public VarDef {
  public:
    enum index { image, projection };
    RefDef(const char* detname, const char* dettype) {
      char buff[128];
      sprintf(buff,"%s_%s_image",detname,dettype);
      NameVec.push_back({buff, Name::UINT16, 2});
      sprintf(buff,"%s_%s_projection",detname,dettype);
      NameVec.push_back({buff, Name::DOUBLE, 1}); }
  };

  class OpalTT {
  public:
    OpalTT(Opal& d, Parameters* para);
    ~OpalTT();
  public:
    void           slowupdate(XtcData::Xtc&);
    void           shutdown ();
    unsigned       configure(XtcData::Xtc&,XtcData::ConfigIter&);
    bool           event    (XtcData::Xtc&,
                             std::vector< XtcData::Array<uint8_t> >&);
  private:
    Opal&                 m_det;
    Parameters*           m_para;
    XtcData::NamesId      m_fexNamesId;
    XtcData::NamesId      m_refNamesId;
    Pds::Semaphore        m_background_sem;
    std::atomic<bool>     m_background_empty; // cache image for slow update transition
    OpalTTFex             m_fex;
  };

  class OpalTTSim {
  public:
    OpalTTSim(const char*, Opal&, Parameters* para);
    ~OpalTTSim();
  public:
    unsigned       configure(XtcData::Xtc&,XtcData::ConfigIter&);
    void           event    (XtcData::Xtc&,
                             std::vector< XtcData::Array<uint8_t> >&);
  private:
    Opal&                 m_det;
    Parameters*           m_para;
    XtcData::NamesId      m_simNamesId;
    std::vector<uint16_t> m_framebuffer;
    std::vector<uint8_t>  m_evtbuffer;
    unsigned              m_evtindex;
  };


};

//
//  Data types from LCLS-1
//
namespace PdsL1 {
  class Xtc {
  public:
    char* payload() { return (char*)(this+1); }
    uint32_t damage;
    uint32_t src_log;
    uint32_t src_phy;
    uint32_t contains;
    uint32_t extent;
  };
  class FrameV1 {
  public:
    uint16_t*   data() { return reinterpret_cast<uint16_t*>(this+1); }
    uint16_t&   operator()(unsigned row, unsigned col) { return data()[row*_width+col]; }
    uint32_t	_width;	/**< Number of pixels in a row. */
    uint32_t	_height;	/**< Number of pixels in a column. */
    uint32_t	_depth;	/**< Number of bits per pixel. */
    uint32_t	_offset;	/**< Fixed offset/pedestal value of pixel data. */
    //uint8_t	_pixel_data[this->_width*this->_height*((this->_depth+7)/8)];
  };
  class FIFOEvent {
  public:
    uint32_t	_timestampHigh;	/**< 119 MHz timestamp (fiducial) */
    uint32_t	_timestampLow;	/**< 360 Hz timestamp */
    uint32_t	_eventCode;	/**< event code (range 0-255) */
  };
  class EvrDataV4 {
  public:
    uint32_t	_u32NumFifoEvents;	/**< length of FIFOEvent list */
    FIFOEvent*  _events() { return reinterpret_cast<FIFOEvent*>(this+1); }
    //EvrData::FIFOEvent	_fifoEvents[this->_u32NumFifoEvents];
  };
  class TimeToolDataV1 {
  public:
    uint32_t	_event_type;	/**< Event designation */
    uint32_t	_z;
    double	_amplitude;	/**< Amplitude of the edge */
    double	_position_pixel;	/**< Filtered pixel position of the edge */
    double	_position_time;	/**< Filtered time position of the edge */
    double	_position_fwhm;	/**< Full-width half maximum of filtered edge (in pixels) */
    double	_nxt_amplitude;	/**< Amplitude of the next largest edge */
    double	_ref_amplitude;	/**< Amplitude of reference at the edge */
    //int32_t	_projected_signal[cfg.signal_projection_size()];
    //int32_t	_projected_sideband[cfg.sideband_projection_size()];
  };
};

static void _load_xtc(std::vector<uint8_t>&, const char*);


using Drp::Opal;
using Drp::OpalTT;
using Drp::OpalTTFex;
using Drp::OpalTTSim;

Opal::Opal(Parameters* para, MemPool* pool) :
    BEBDetector   (para, pool),
    m_tt          (0),
    m_sim         (0)
{
  _init(para->detName.c_str());  // an argument is required here
  _init_feb();

#define MLOOKUP(m,name,dflt) (m.find(name)==m.end() ? dflt : m[name].c_str())

  const char* simxtc = MLOOKUP(m_para->kwargs,"simxtc",0);
  if (simxtc)
    m_sim = new OpalTTSim(simxtc,*this, para);
}

Opal::~Opal()
{
  if (m_sim) delete m_sim;
  if (m_tt ) delete m_tt;
}

void Opal::_connect(PyObject* mbytes)
{
    unsigned modelnum = strtoul( _string_from_PyDict(mbytes,"model").c_str(), NULL, 10);
#define MODEL(num,rows,cols) case num: m_rows = rows; m_columns = cols; break
    switch(modelnum) {
        MODEL(1000,1024,1024);
        MODEL(1600,1200,1600);
        MODEL(2000,1080,1920);
        MODEL(4000,1752,2336);
        MODEL(8000,2472,3296);
#undef MODEL
    default:
        throw std::string("Opal camera model not recognized");
        break;
    }

    m_para->serNo = _string_from_PyDict(mbytes,"serno");
}

json Opal::connectionInfo()
{
  return BEBDetector::connectionInfo();

    // Exclude connection info until cameralink-gateway timingTxLink is fixed
    logging::error("Returning NO XPM link; implementation incomplete");
    return json({});
}

unsigned Opal::_configure(XtcData::Xtc& xtc,XtcData::ConfigIter& configo)
{
    // set up the names for L1Accept data
    m_evtNamesId = NamesId(nodeId, EventNamesIndex);
    Alg alg("raw", 2, 0, 0);
    Names& eventNames = *new(xtc) Names(m_para->detName.c_str(), alg,
                                        m_para->detType.c_str(), m_para->serNo.c_str(), m_evtNamesId, m_para->detSegment);

    eventNames.add(xtc, rawDef);
    m_namesLookup[m_evtNamesId] = NameIndex(eventNames);

    if (m_tt) { delete m_tt; m_tt = 0; }

    XtcData::DescData& descdata = configo.desc_shape();
    IndexMap& nameMap = descdata.nameindex().nameMap();
    if (nameMap.find("fex.enable")!=nameMap.end() && descdata.get_value<uint8_t>("fex.enable"))
      (m_tt = new OpalTT(*this,m_para))->configure(xtc,configo);

    if (m_sim) m_sim->configure(xtc,configo);

    return 0;
}

void Opal::_event(XtcData::Xtc& xtc, std::vector< XtcData::Array<uint8_t> >& subframes)
{
  if (m_sim) m_sim->event(xtc,subframes);
  if (m_tt && !m_tt->event(xtc,subframes))
      return;
  write_image(xtc,subframes, m_evtNamesId);
}

void Opal::write_image(XtcData::Xtc& xtc, std::vector< XtcData::Array<uint8_t> >& subframes,
                       XtcData::NamesId& namesId)
{
  CreateData cd(xtc, m_namesLookup, namesId);

  unsigned shape[MaxRank];
  shape[0] = m_rows;
  shape[1] = m_columns;
  Array<uint8_t> arrayT = cd.allocate<uint8_t>(RawDef::image, shape);
  memcpy(arrayT.data(), subframes[2].data(), subframes[2].shape()[0]);
}

void     Opal::slowupdate(XtcData::Xtc& xtc)
{
  if (m_tt) m_tt->slowupdate(xtc);
  else this->Detector::slowupdate(xtc);
}

void     Opal::shutdown()
{
  if (m_tt) m_tt->shutdown();
}

OpalTT::OpalTT(Opal& d, Parameters* para) :
  m_det             (d),
  m_para            (para),
  m_background_sem  (Pds::Semaphore::FULL),
  m_background_empty(true),
  m_fex             (para)
{
}

OpalTT::~OpalTT() {}

void     OpalTT::slowupdate(XtcData::Xtc& xtc)
{
  m_background_sem.take();
  memcpy((void*)&xtc, (const void*)&m_det.transitionXtc(), m_det.transitionXtc().extent);
  m_background_empty = true;
  m_background_sem.give();
}

void     OpalTT::shutdown() { m_fex.unconfigure(); }

unsigned OpalTT::configure(XtcData::Xtc& xtc, XtcData::ConfigIter& cfg)
{
    m_fex.configure(cfg, m_det.m_columns, m_det.m_rows);

    // set up the names for L1Accept data
    { m_fexNamesId = NamesId(m_det.nodeId, EventNamesIndex+1);
      Alg alg("ttfex", 2, 0, 0);
      Names& fexNames = *new(xtc) Names(m_para->detName.c_str(), alg,
                                        m_para->detType.c_str(), m_para->serNo.c_str(), m_fexNamesId, m_para->detSegment);
      FexDef fexDef(m_fex);
      fexNames.add(xtc, fexDef);
      m_det.namesLookup()[m_fexNamesId] = NameIndex(fexNames);
    }

    // set up the data for slow update
    { m_refNamesId = NamesId(m_det.nodeId, EventNamesIndex+3);
      Alg alg("opaltt", 2, 0, 0);
      Names& bkgNames = *new(xtc) Names("epics", alg,
                                        "epics", m_para->serNo.c_str(), m_refNamesId, m_para->detSegment);
      RefDef refDef(m_para->detName.c_str(),"opaltt");
      bkgNames.add(xtc, refDef);
      m_det.namesLookup()[m_refNamesId] = NameIndex(bkgNames);
    }

   return 0;
}

bool OpalTT::event(XtcData::Xtc& xtc, std::vector< XtcData::Array<uint8_t> >& subframes)
{
  m_fex.reset();

  OpalTTFex::TTResult result = m_fex.analyze(subframes);

  CreateData cd(xtc, m_det.namesLookup(), m_fexNamesId);

  if (result == OpalTTFex::INVALID) {
    xtc.damage.increase(Damage::UserDefined);
  }
  else if (result == OpalTTFex::VALID) {
    //  Insert the results
    cd.set_value(FexDef::ampl      , m_fex.amplitude());
    cd.set_value(FexDef::fltpos    , m_fex.filtered_position());
    cd.set_value(FexDef::fltpos_ps , m_fex.filtered_pos_ps());
    cd.set_value(FexDef::fltposfwhm, m_fex.filtered_fwhm());
    cd.set_value(FexDef::nxtampl   , m_fex.next_amplitude());
    cd.set_value(FexDef::refampl   , m_fex.ref_amplitude());

#define copy_projection(atype, src, index) {                            \
      unsigned shape[1];                                                \
      shape[0] = src.size();                                            \
      Array<atype> a = cd.allocate<atype>(index,shape);                 \
      memcpy(a.data(), src.data(), src.size()*sizeof(atype)); }

    if (m_fex.write_evt_projections()) {
      copy_projection(int   , m_fex.sig_projection(), FexDef::proj_sig);
      copy_projection(double, m_fex.ref_projection(), FexDef::proj_ref);
    }
  }
  else if (result == OpalTTFex::NOBEAM) {
    m_background_sem.take();
    // Only do this once per SlowUpdate
    if (m_background_empty) {
      m_det.transitionXtc().extent = sizeof(Xtc);
      CreateData cd(m_det.transitionXtc(), m_det.m_namesLookup, m_refNamesId);
      {
        unsigned shape[MaxRank];
        shape[0] =m_fex.write_ref_image() ?  m_det.m_rows : 0;
        shape[1] = m_det.m_columns;
        Array<uint8_t> arrayT = cd.allocate<uint8_t>(RefDef::image, shape);
        memcpy(arrayT.data(), subframes[2].data(), m_fex.write_ref_image() ? subframes[2].shape()[0] : 0);
      }
      {
        unsigned shape[MaxRank];
        shape[0] = m_fex.write_ref_projection() ? m_fex.ref_projection().size() : 0;
        Array<double> arrayT = cd.allocate<double>(RefDef::projection, shape);
        memcpy(arrayT.data(), m_fex.ref_projection().data(), shape[0]*sizeof(double));
      }
      m_background_empty = false;
    }
    m_background_sem.give();
  }

  return m_fex.write_image();
}


OpalTTSim::OpalTTSim(const char* evtxtc, Opal& d, Parameters* para) :
  m_det         (d),
  m_para        (para),
  m_simNamesId  (0,0),
  m_framebuffer  (2*1024*1024),
  m_evtindex     (0)

{
  _load_xtc(m_evtbuffer, evtxtc);
}

OpalTTSim::~OpalTTSim()
{
}

unsigned OpalTTSim::configure(XtcData::Xtc& xtc, XtcData::ConfigIter& cfg)
{
    //  Add results into the dgram
    m_simNamesId = NamesId(m_det.nodeId, EventNamesIndex+2);
    Alg alg("simfex", 2, 0, 0);
    Names& fexNames = *new(xtc) Names(m_para->detName.c_str(), alg,
                                      m_para->detType.c_str(), m_para->serNo.c_str(), m_simNamesId, m_para->detSegment);

    FexDef fexDef;
    fexNames.add(xtc, fexDef);
    m_det.namesLookup()[m_simNamesId] = NameIndex(fexNames);

    return 0;
}

void OpalTTSim::event(XtcData::Xtc& xtc, std::vector< XtcData::Array<uint8_t> >& subframes)
{
#define L1PAYLOAD(ptype,f) \
  ptype& f = *reinterpret_cast<ptype*>( reinterpret_cast<PdsL1::Xtc*>(&m_evtbuffer[m_evtindex])->payload() ); \
  m_evtindex += reinterpret_cast<PdsL1::Xtc*>(&m_evtbuffer[m_evtindex])->extent

  L1PAYLOAD(PdsL1::FrameV1       ,f);
  L1PAYLOAD(PdsL1::EvrDataV4     ,e);
  L1PAYLOAD(PdsL1::TimeToolDataV1,t);
  if (m_evtindex >= m_evtbuffer.size()) {
    printf("Resetting input events\n");
    m_evtindex=0;
  }

  CreateData cd(xtc, m_det.namesLookup(), m_simNamesId);

   //  Insert the results
   cd.set_value(FexDef::ampl      , t._amplitude);
   cd.set_value(FexDef::fltpos    , t._position_pixel);
   cd.set_value(FexDef::fltpos_ps , t._position_time);
   cd.set_value(FexDef::fltposfwhm, t._position_fwhm);
   cd.set_value(FexDef::nxtampl   , t._nxt_amplitude);
   cd.set_value(FexDef::refampl   , t._ref_amplitude);

   //  Copy the ROI into a full image
   unsigned shape[2];
   shape[0] = m_det.m_columns;
   shape[1] = m_det.m_rows;
   for(unsigned i=0; i<f._height; i++)
     memcpy(m_framebuffer.data()+i*m_det.m_columns,&f(i,0),f._width*sizeof(uint16_t));

   shape[0] = m_det.m_rows*m_det.m_columns*sizeof(uint16_t);
   subframes[2] = Array<uint8_t>(m_framebuffer.data(), shape, 1);

#ifdef DBUG
   printf("Copied %d/%d rows x %d/%d cols into subframes\n", 
          f._height, shape[1], f._width, shape[0]);
#endif

  // transfer event codes into EventInfo
  { EventInfo& info = *reinterpret_cast<EventInfo*>(subframes[3].data());
    memset(info._seqInfo, 0, sizeof(info._seqInfo));
    for(unsigned i=0; i<e._u32NumFifoEvents; i++) {
      unsigned ec = e._events()[i]._eventCode;
      info._seqInfo[ec>>4] |= (1<<(ec&0x1f));
    } }

}

void _load_xtc(std::vector<uint8_t>& buffer, const char* filename)
{
  int fd = open(filename, O_RDONLY);
  if (fd < 0) {
    perror("Error opening file");
    exit(1);
  }
  struct stat s;
  if (fstat(fd, &s)) {
    perror("Error fetching file size");
    exit(2);
  }
  buffer.resize(s.st_size);
  int bytes = read(fd, buffer.data(), s.st_size);
  if (bytes != s.st_size) {
    perror("Error reading all bytes");
    exit(3);
  }
}
