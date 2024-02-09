#include "Opal.hh"
#include "OpalTTFex.hh"
#include "psdaq/service/Semaphore.hh"
#include "psdaq/epicstools/EpicsPVA.hh"
#include "psdaq/epicstools/EpicsProviders.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/XtcFileIterator.hh"
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
            ampl, fltpos, fltpos_ps, fltposfwhm, nxtampl, refampl
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
    };

    class ProjDef : public VarDef
    {
    public:
        enum index {
            proj_sig, proj_ref
        };

        ProjDef()
        {
            NameVec.push_back({"proj_sig"   , Name::DOUBLE, 1});
            NameVec.push_back({"proj_ref"   , Name::DOUBLE, 1});
        }
    };

    //
    //  This Names doesn't satisfy our general rules:
    //    its structure is configuration-dependent
    //    its data also only appears on SlowUpdate
    //    there is no detector interface for it yet
    //
    class RefDef : public VarDef {
    public:
        //        enum index { image, projection };
        RefDef(const char* detname, const char* dettype,
               bool write_image,
               bool write_projection) {
            char buff[128];
            if (write_image) {
                sprintf(buff,"%s_%s_image",detname,dettype);
                NameVec.push_back({buff, Name::UINT16, 2});
            }
            if (write_projection) {
                sprintf(buff,"%s_%s_projection",detname,dettype);
                NameVec.push_back({buff, Name::DOUBLE, 1});
            }
        }
    };

    class OpalTT {
    public:
        OpalTT(Opal& d, Parameters* para);
        ~OpalTT();
    public:
        void           slowupdate(XtcData::Xtc&, const void* bufEnd);
        void           shutdown ();
        unsigned       configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&);
        bool           event    (XtcData::Xtc&, const void* bufEnd,
                                 std::vector< XtcData::Array<uint8_t> >&);
    private:
        Opal&                 m_det;
        Parameters*           m_para;
        XtcData::NamesId      m_fexNamesId;
        XtcData::NamesId      m_projNamesId;
        XtcData::NamesId      m_refNamesId;
        Pds::Semaphore        m_background_sem;
        std::atomic<bool>     m_background_empty; // cache image for slow update transition
        OpalTTFex             m_fex;
        pvac::ClientChannel   m_fex_pv;
        pvd::PVStructure::const_shared_pointer m_request;
        double                *m_vec;
        const char            *m_ttpv;
    };

    class OpalTTSim {
    public:
        virtual ~OpalTTSim() {}
        virtual unsigned       configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) = 0;
        virtual void           event    (XtcData::Xtc&, const void* bufEnd,
                                         std::vector< XtcData::Array<uint8_t> >&) = 0;
    };

    class OpalTTSimL1 : public OpalTTSim {
    public:
        OpalTTSimL1(const char*, Opal&, Parameters* para);
        ~OpalTTSimL1();
    public:
        unsigned       configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&);
        void           event    (XtcData::Xtc&, const void* bufEnd,
                                 std::vector< XtcData::Array<uint8_t> >&);
    private:
        Opal&                 m_det;
        Parameters*           m_para;
        XtcData::NamesId      m_simNamesId;
        std::vector<uint16_t> m_framebuffer;
        std::vector<uint8_t>  m_evtbuffer;
        unsigned              m_evtindex;
    };

    class L2Iter : public XtcIterator
    {
    public:
        enum { Stop, Continue };
        L2Iter() : XtcIterator() {}

        void get_value(int i, Name& name, DescData& descdata);
        int process(Xtc*, const void* bufEnd);
    public:
        NamesLookup namesLookup;
        std::unordered_map<unsigned,ShapesData*> shapesdata;
    };

    class OpalTTSimL2 : public OpalTTSim {
    public:
        OpalTTSimL2(const char*,const char*, Opal&, Parameters* para);
        ~OpalTTSimL2();
    public:
        unsigned       configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&);
        void           event    (XtcData::Xtc&, const void* bufEnd,
                                 std::vector< XtcData::Array<uint8_t> >&);
    private:
        Opal&                 m_det;
        Parameters*           m_para;
        XtcData::NamesId      m_simNamesId;
        std::vector<uint16_t> m_framebuffer;
        Pds::Semaphore        m_filesem;
        XtcFileIterator*      m_iter;
        XtcFileIterator*      m_timiter;
        L2Iter                m_input;
        L2Iter                m_timinput;
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
using Drp::OpalTTSimL1;
using Drp::OpalTTSimL2;

Opal::Opal(Parameters* para, MemPool* pool) :
    BEBDetector   (para, pool),
    m_tt          (0),
    m_sim         (0),
    m_notifySocket{&m_context, ZMQ_PUSH}
{
    // ZMQ socket for reporting errors
    m_notifySocket.connect({"tcp://" + para->collectionHost + ":" + std::to_string(CollectionApp::zmq_base_port + para->partition)});

    _init(para->detName.c_str());  // an argument is required here
    _init_feb();

#define MLOOKUP(m,name,dflt) (m.find(name)==m.end() ? dflt : m[name].c_str())

    const char* simxtc  = MLOOKUP(m_para->kwargs,"simxtc" ,0);
    const char* simxtc2 = MLOOKUP(m_para->kwargs,"simxtc2",0);
    const char* simtime = MLOOKUP(m_para->kwargs,"simtime",0);
    if (simxtc) {
        m_sim = new OpalTTSimL1(simxtc ,*this, para);
    }
    else if (simxtc2) {
        m_sim = new OpalTTSimL2(simxtc2,simtime,*this, para);
    }
}

Opal::~Opal()
{
    if (m_sim) delete m_sim;
    if (m_tt ) delete m_tt;
}

void Opal::_fatal_error(std::string errMsg)
{
    logging::critical("%s", errMsg.c_str());
    json msg = createAsyncErrMsg(m_para->alias, errMsg);
    m_notifySocket.send(msg.dump());
    throw errMsg;
}

void Opal::_connectionInfo(PyObject* mbytes)
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
        _fatal_error("Opal camera model " + std::to_string(modelnum) +
                     " not recognized");
        break;
    }

    m_para->serNo = _string_from_PyDict(mbytes,"serno");
}

unsigned Opal::_configure(XtcData::Xtc& xtc,const void* bufEnd,XtcData::ConfigIter& configo)
{
    // set up the names for L1Accept data
    m_evtNamesId = NamesId(nodeId, EventNamesIndex);
    Alg alg("raw", 2, 0, 0);
    Names& eventNames = *new(xtc,bufEnd) Names(bufEnd,
                                               m_para->detName.c_str(), alg,
                                               m_para->detType.c_str(), m_para->serNo.c_str(), m_evtNamesId, m_para->detSegment);

    eventNames.add(xtc, bufEnd, rawDef);
    m_namesLookup[m_evtNamesId] = NameIndex(eventNames);

    if (m_tt) { delete m_tt; m_tt = 0; }

    XtcData::DescData& descdata = configo.desc_shape();
    IndexMap& nameMap = descdata.nameindex().nameMap();
    if (nameMap.find("fex.enable")!=nameMap.end() && descdata.get_value<uint8_t>("fex.enable"))
        (m_tt = new OpalTT(*this,m_para))->configure(xtc,bufEnd,configo);

    if (m_sim) m_sim->configure(xtc,bufEnd,configo);

    return 0;
}

void Opal::_event(XtcData::Xtc& xtc, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >& subframes)
{
    if (m_sim) m_sim->event(xtc,bufEnd,subframes);
    if (m_tt && !m_tt->event(xtc,bufEnd,subframes))
        return;
    write_image(xtc,bufEnd,subframes, m_evtNamesId);
}

void Opal::write_image(XtcData::Xtc& xtc, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >& subframes,
                       XtcData::NamesId& namesId)
{
    CreateData cd(xtc, bufEnd, m_namesLookup, namesId);

    unsigned shape[MaxRank];
    shape[0] = m_rows;
    shape[1] = m_columns;
    Array<uint16_t> arrayT = cd.allocate<uint16_t>(RawDef::image, shape);
    memcpy(arrayT.data(), subframes[2].data(), subframes[2].shape()[0]);
}

void     Opal::slowupdate(XtcData::Xtc& xtc, const void* bufEnd)
{
    logging::debug("%s: m_tt = %s", __PRETTY_FUNCTION__, m_tt ? "true" : "false");
    if (m_tt) m_tt->slowupdate(xtc, bufEnd);
    else this->Detector::slowupdate(xtc, bufEnd);
}

void     Opal::shutdown()
{
    if (m_tt) m_tt->shutdown();
    this->BEBDetector::shutdown();
}

OpalTT::OpalTT(Opal& d, Parameters* para) :
    m_det             (d),
    m_para            (para),
    m_background_sem  (Pds::Semaphore::FULL),
    m_background_empty(true),
    m_fex             (para)
{
    m_ttpv = MLOOKUP(m_para->kwargs,"ttpv",0);
    if (m_ttpv) {
        logging::info("Connecting to pv %s\n", m_ttpv);
        try {
            m_fex_pv = pvac::ClientChannel(Pds_Epics::EpicsProviders::ca().connect(m_ttpv));
            m_request = pvd::createRequest("field(value)");
            pvd::PVStructure::const_shared_pointer cpv =
                m_fex_pv.get(3.0,m_request);
            const pvd::PVFieldPtrArray& fields = cpv->getPVFields();
            for(unsigned i=0; i<fields.size(); i++) {
                const pvd::PVFieldPtr field = fields[i];
                logging::info("%s [%s] [%s]\n",
                              field->getFieldName().c_str(),
                              field->getFullName().c_str(),
                              field->getField()->getID().c_str());
            }
            logging::info("Connection complete\n");
        } catch(...) {
            d._fatal_error("Error connecting to feedback PV");
        }
    }
    else {
        logging::info("No feedback pv specified\n");
    }
}

OpalTT::~OpalTT() {}

void     OpalTT::slowupdate(XtcData::Xtc& xtc, const void* bufEnd)
{
    logging::debug("%s: m_background_empty = %s", __PRETTY_FUNCTION__, m_background_empty ? "true" : "false");
    m_background_sem.take();
    if (!m_background_empty) {
        XtcData::Xtc& trXtc = m_det.transitionXtc();
        xtc = trXtc; // Preserve header info, but allocate to check fit
        auto payload = xtc.alloc(trXtc.sizeofPayload(), bufEnd);
        memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());
        m_background_empty = true;
    }
    else {
        // no payload
        xtc = {{XtcData::TypeId::Parent, 0}, {m_det.nodeId}};
    }
    m_background_sem.give();
}

void     OpalTT::shutdown() { m_fex.unconfigure(); }

unsigned OpalTT::configure(XtcData::Xtc& xtc, const void* bufEnd, XtcData::ConfigIter& cfg)
{
    m_fex.configure(cfg, m_det.m_columns, m_det.m_rows);

    // set up the names for L1Accept data
    { m_fexNamesId = NamesId(m_det.nodeId, EventNamesIndex+1);
        Alg alg("ttfex", 2, 1, 0);
        Names& fexNames = *new(xtc, bufEnd) Names(bufEnd,
                                                  m_para->detName.c_str(), alg,
                                                  m_para->detType.c_str(), m_para->serNo.c_str(), m_fexNamesId, m_para->detSegment);
        FexDef fexDef;
        fexNames.add(xtc, bufEnd, fexDef);
        m_det.namesLookup()[m_fexNamesId] = NameIndex(fexNames);
    }

    // and the conditional projections
    { m_projNamesId = NamesId(m_det.nodeId, EventNamesIndex+2);
        Alg alg("ttproj", 2, 0, 0);
        Names& fexNames = *new(xtc, bufEnd) Names(bufEnd,
                                                  m_para->detName.c_str(), alg,
                                                  m_para->detType.c_str(), m_para->serNo.c_str(), m_projNamesId, m_para->detSegment);
        ProjDef fexDef;
        fexNames.add(xtc, bufEnd, fexDef);
        m_det.namesLookup()[m_projNamesId] = NameIndex(fexNames);
    }

    // set up the data for slow update
    if (m_fex.write_ref_image() ||
        m_fex.write_ref_projection()) {
        m_refNamesId = NamesId(m_det.nodeId, EventNamesIndex+3);
        Alg alg("opaltt", 2, 0, 0);
        // cpo: rename this away from "epics" for now because the
        // segment number can conflict with epicsarch.
        Names& bkgNames = *new(xtc, bufEnd) Names(bufEnd,
                                                  "epics_dontuse", alg,
                                                  "epics_dontuse", m_para->serNo.c_str(), m_refNamesId, m_para->detSegment);
        RefDef refDef(m_para->detName.c_str(),"opaltt",
                      m_fex.write_ref_image(),
                      m_fex.write_ref_projection());
        bkgNames.add(xtc, bufEnd, refDef);
        m_det.namesLookup()[m_refNamesId] = NameIndex(bkgNames);
    }

    return 0;
}

bool OpalTT::event(XtcData::Xtc& xtc, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >& subframes)
{
    m_fex.reset();

    std::vector<double> sig, ref;
    OpalTTFex::TTResult result = m_fex.analyze(subframes,sig,ref);

    if (result == OpalTTFex::INVALID) {
        xtc.damage.increase(Damage::UserDefined);
    }
    else if (result == OpalTTFex::VALID) {
        //  Live feedback
	m_vec = new double[6];
        pvd::shared_vector<const double> ttvec(m_vec,0,6);
        m_vec[0] = m_fex.filtered_position();
        m_vec[1] = m_fex.filtered_pos_ps();
        m_vec[2] = m_fex.amplitude();
        m_vec[3] = m_fex.next_amplitude();
        m_vec[4] = m_fex.ref_amplitude();
        m_vec[5] = m_fex.filtered_fwhm();
        if (m_ttpv) {
            m_fex_pv.put(m_request).set<const double>("value",ttvec).exec();
        }
        //  Insert the results
        CreateData cd(xtc, bufEnd, m_det.namesLookup(), m_fexNamesId);
        cd.set_value(FexDef::ampl      , m_fex.amplitude());
        cd.set_value(FexDef::fltpos    , m_fex.filtered_position());
        cd.set_value(FexDef::fltpos_ps , m_fex.filtered_pos_ps());
        cd.set_value(FexDef::fltposfwhm, m_fex.filtered_fwhm());
        cd.set_value(FexDef::nxtampl   , m_fex.next_amplitude());
        cd.set_value(FexDef::refampl   , m_fex.ref_amplitude());

#define copy_projection(atype, src, index) {                            \
            unsigned shape[1];                                          \
            shape[0] = src.size();                                      \
            Array<atype> a = cdp.allocate<atype>(index,shape);          \
            memcpy(a.data(), src.data(), src.size()*sizeof(atype)); }

        if (m_fex.write_evt_projections()) {
#ifdef DBUG
            printf("writing projections sized %zu %zu\n",sig.size(),ref.size());
#endif
            CreateData cdp(xtc, bufEnd, m_det.namesLookup(), m_projNamesId);
            copy_projection(double, sig, ProjDef::proj_sig);
            copy_projection(double, ref, ProjDef::proj_ref);
        }
    }
    else if (result == OpalTTFex::NOBEAM) {
        m_background_sem.take();
        // Only do this once per SlowUpdate
        if (m_background_empty) {
            m_det.transitionXtc().extent = sizeof(Xtc);
            if (m_fex.write_ref_image() || m_fex.write_ref_projection()) {
                CreateData cd(m_det.transitionXtc(), m_det.trXtcBufEnd(), m_det.m_namesLookup, m_refNamesId);
                unsigned index=0;
                if (m_fex.write_ref_image()) {
                    unsigned shape[MaxRank];
                    shape[0] = m_det.m_rows;
                    shape[1] = m_det.m_columns;
                    Array<uint16_t> arrayT = cd.allocate<uint16_t>(index++, shape);
                    memcpy(arrayT.data(), subframes[2].data(), subframes[2].shape()[0]);
                }
                if (m_fex.write_ref_projection()) {
                    unsigned shape[MaxRank];
                    shape[0] = m_fex.ref_projection().size();
                    Array<double> arrayT = cd.allocate<double>(index++, shape);
                    memcpy(arrayT.data(), m_fex.ref_projection().data(), shape[0]*sizeof(double));
                }
            }
            m_background_empty = false;
        }
        m_background_sem.give();
    }

    return m_fex.write_image();
}


OpalTTSimL1::OpalTTSimL1(const char* evtxtc, Opal& d, Parameters* para) :
    m_det         (d),
    m_para        (para),
    m_simNamesId  (0,0),
    m_framebuffer  (2*1024*1024),
    m_evtindex     (0)

{
    _load_xtc(m_evtbuffer, evtxtc);
}

OpalTTSimL1::~OpalTTSimL1()
{
}

unsigned OpalTTSimL1::configure(XtcData::Xtc& xtc, const void* bufEnd, XtcData::ConfigIter& cfg)
{
    //  Add results into the dgram
    m_simNamesId = NamesId(m_det.nodeId, EventNamesIndex+4);
    Alg alg("simfex", 2, 1, 0);
    Names& fexNames = *new(xtc, bufEnd) Names(bufEnd,
                                              m_para->detName.c_str(), alg,
                                              m_para->detType.c_str(), m_para->serNo.c_str(), m_simNamesId, m_para->detSegment);

    FexDef fexDef;
    fexNames.add(xtc, bufEnd, fexDef);
    m_det.namesLookup()[m_simNamesId] = NameIndex(fexNames);

    return 0;
}

void OpalTTSimL1::event(XtcData::Xtc& xtc, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >& subframes)
{
#define L1PAYLOAD(ptype,f)                                              \
    ptype& f = *reinterpret_cast<ptype*>( reinterpret_cast<PdsL1::Xtc*>(&m_evtbuffer[m_evtindex])->payload() ); \
    m_evtindex += reinterpret_cast<PdsL1::Xtc*>(&m_evtbuffer[m_evtindex])->extent

    L1PAYLOAD(PdsL1::FrameV1       ,f);
    L1PAYLOAD(PdsL1::EvrDataV4     ,e);
    L1PAYLOAD(PdsL1::TimeToolDataV1,t);
    if (m_evtindex >= m_evtbuffer.size()) {
        printf("Resetting input events\n");
        m_evtindex=0;
    }

    CreateData cd(xtc, bufEnd, m_det.namesLookup(), m_simNamesId);

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

OpalTTSimL2::OpalTTSimL2(const char* evtxtc, const char* timxtc, Opal& d, Parameters* para) :
    m_det         (d),
    m_para        (para),
    m_simNamesId  (0,0),
    m_framebuffer (2*1024*1024),
    m_filesem     (Pds::Semaphore::FULL)
{
    int fd = open(evtxtc, O_RDONLY);
    if (fd < 0) {
        perror("Error opening file");
        exit(1);
    }

    m_iter = new XtcData::XtcFileIterator(fd, 0x400000);

    fd = open(timxtc, O_RDONLY);
    if (fd < 0) {
        perror("Error opening file");
        exit(1);
    }

    m_timiter = new XtcData::XtcFileIterator(fd, 0x40000);
}

OpalTTSimL2::~OpalTTSimL2()
{
}

unsigned OpalTTSimL2::configure(XtcData::Xtc& xtc, const void* bufEnd, XtcData::ConfigIter& cfg)
{
    //  Add results into the dgram
    m_simNamesId = NamesId(m_det.nodeId, EventNamesIndex+4);
    Alg alg("simfex", 2, 1, 0);
    Names& fexNames = *new(xtc, bufEnd) Names(bufEnd,
                                              m_para->detName.c_str(), alg,
                                              m_para->detType.c_str(), m_para->serNo.c_str(), m_simNamesId, m_para->detSegment);

    FexDef fexDef;
    fexNames.add(xtc, bufEnd, fexDef);
    m_det.namesLookup()[m_simNamesId] = NameIndex(fexNames);

    //  Retrieve the offline configuration for parsing the event data
    Dgram* dg;
    void* end;
#define FIND_CONFIG(iter,input)                                         \
    dg=iter->next();                                                    \
    end=(char*)dg + iter->size();                                       \
    while(dg) {                                                         \
        if (dg->service()==TransitionId::Configure) {                   \
            input.process(&dg->xtc, end);                               \
            break;                                                      \
        }                                                               \
        dg=iter->next();                                                \
    }

    FIND_CONFIG(m_iter,m_input);
    FIND_CONFIG(m_timiter,m_timinput);

#define DUMP_NAMES(input)                                               \
    for(std::unordered_map<unsigned,NameIndex>::iterator it=input.namesLookup.begin(); \
        it!=input.namesLookup.end(); it++) {                            \
        printf("namesid 0x%x\n",it->first);                             \
        Names& names = it->second.names();                              \
        for (unsigned i = 0; i < names.num(); i++) {                    \
            Name& name = names.get(i);                                  \
            printf("  %s\n",name.name());                               \
        }                                                               \
    }

    DUMP_NAMES(m_input);
    DUMP_NAMES(m_timinput);

    return 0;
}

void OpalTTSimL2::event(XtcData::Xtc& xtc, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >& subframes)
{
    m_filesem.take();
    Dgram* dg;
    void* end;
#define FIND_L1A(iter,input)                                            \
    while(1) {                                                          \
        dg=iter->next();                                                \
        end=(char*)dg + iter->size();                                   \
        while(dg) {                                                     \
            if (dg->service()==TransitionId::L1Accept) {                \
                input.process(&dg->xtc, end);                           \
                break;                                                  \
            }                                                           \
            dg=iter->next();                                            \
        }                                                               \
        if (dg)                                                         \
            break;                                                      \
        else {                                                          \
            logging::info("Rewind input xtc\n");                        \
            iter->rewind();                                             \
        }                                                               \
    }

    FIND_L1A(m_iter,m_input);
    FIND_L1A(m_timiter,m_timinput);

    /*
    CreateData cd(xtc, bufEnd, m_det.namesLookup(), m_simNamesId);

    //  Insert the results
    cd.set_value(FexDef::ampl      , t._amplitude);
    cd.set_value(FexDef::fltpos    , t._position_pixel);
    cd.set_value(FexDef::fltpos_ps , t._position_time);
    cd.set_value(FexDef::fltposfwhm, t._position_fwhm);
    cd.set_value(FexDef::nxtampl   , t._nxt_amplitude);
    cd.set_value(FexDef::refampl   , t._ref_amplitude);
    */

            //Data& data = it->second->data();

    //  Copy the ROI into a full image
    {
        unsigned namesId = m_input.namesLookup.begin()->first;
        namesId &= ~0xff;
        namesId |= 0x0a; // event names ID

        DescData descdata(*m_input.shapesdata[namesId],
                          m_input.namesLookup[namesId]);
        memcpy(subframes[2].data(), descdata.get_array<uint8_t>(0).data(), subframes[2].num_elem());
    }

    // transfer event codes into EventInfo
    {
        unsigned namesId = m_timinput.namesLookup.begin()->first;
        namesId &= ~0xff;
        namesId |= 0x01; // event names ID
        NameIndex&  index  = m_timinput.namesLookup[namesId];
        ShapesData& shapes = *m_timinput.shapesdata[namesId];
        DescData descdata(shapes, index);
        EventInfo& info = *reinterpret_cast<EventInfo*>(subframes[3].data());
        memcpy(info._seqInfo, descdata.get_array<uint8_t>(index.nameMap()["sequenceValues"]).data(), 18*sizeof(uint16_t));
#ifdef DBUG
        const uint16_t* p = (const uint16_t*)(info._seqInfo);
        printf("seq:");
        for(unsigned i=0; i<16; i++)
            printf(" %04x",p[i]);
        printf("\n");
#endif
    }
    m_filesem.give();
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

int Drp::L2Iter::process(Xtc* xtc, const void* bufEnd)
{
    switch (xtc->contains.id()) {
    case (TypeId::Parent): {
        iterate(xtc, bufEnd);
        break;
    }
    case (TypeId::Names): {
        Names& names = *(Names*)xtc;
        namesLookup[names.namesId()] = NameIndex(names);
        break;
    }
    case (TypeId::ShapesData): {
        ShapesData& _shapesdata = *(ShapesData*)xtc;
        shapesdata[_shapesdata.namesId()] = &_shapesdata;
        break;
    }
    default:
        break;
    }
    return Continue;
}
