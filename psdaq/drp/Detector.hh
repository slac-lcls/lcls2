/**
 * @file
 * @brief Detector, the abstract base of the DRP detector classes, and Factory, which creates detectors by type name.
 */
#pragma once

#include "drp.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "psdaq/service/EbDgram.hh"

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>

/** File-static, default-constructed NameIndex (each including translation unit gets its own copy); no use was found in psdaq. */
static XtcData::NameIndex _noName;

namespace Pds {
  namespace Eb {
    class ResultDgram;
  }
}

namespace Drp {
    /** Namespace of the GPU DRP code (psdaq/drpGpu); only Gpu::Detector is forward-declared here. */
    namespace Gpu {
        class Detector;
    }

struct Parameters;

/** Abstract base of the DRP detector classes: transition hooks (most default to doing nothing and returning 0), event handlers, cube binning hooks and a buffer for building transition XTC. */
class Detector
{
public:
    /** Set nodeId to -1u and virtChan to 0, keep para and pool, and allocate a transition buffer of para->maxTrSize bytes. */
    Detector(Parameters* para, MemPool* pool) :
        nodeId(-1u), virtChan(0), m_para(para), m_pool(pool), m_xtcbuf(para->maxTrSize) {}
    /** Does nothing; empty body. */
    virtual ~Detector() {}

    /** Return nullptr; Gpu::Detector overrides it to return itself. */
    virtual Gpu::Detector* gpuDetector() { return nullptr; }

    /** Return an empty JSON object unless overridden; msg is not used. */
    virtual nlohmann::json connectionInfo(const nlohmann::json& msg) {return nlohmann::json({});}
    /** Does nothing unless overridden. */
    virtual void connectionShutdown() {}
    /** Does nothing unless overridden. */
    virtual void connect(const nlohmann::json&, const std::string& collectionId) {};
    /** Pure virtual: add the Configure data for config_alias to xtc; implementations return 0 on success. */
    virtual unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) = 0;
    /** Return 0 without doing anything unless overridden. */
    virtual unsigned beginrun (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) {return 0;}
    /** Return 0 without doing anything unless overridden. */
    virtual unsigned beginstep(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& stepInfo) {return 0;};
    /** Return 0 without doing anything unless overridden. */
    virtual unsigned enable   (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) {return 0;};
    /** Return 0 without doing anything unless overridden. */
    virtual unsigned disable  (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) {return 0;};
    /** Reset xtc to an empty Parent Xtc whose source is nodeId. */
    virtual void slowupdate(XtcData::Xtc& xtc, const void* bufEnd) { xtc = {{XtcData::TypeId::Parent, 0}, {nodeId}}; };
    /** Pure virtual: add the L1Accept data of event (the DMA buffers of its lanes) to dgram. */
    virtual void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count) = 0;
    /** Does nothing unless overridden; called with the trigger result for an event. */
    virtual void event(XtcData::Dgram& dgram, const void* bufEnd, const Pds::Eb::ResultDgram& result) {};
    // For binning into the cube
    /** Return the shape in rawData of entry valueIndex of raw definition rawDefIndex (from rawDef()). */
    virtual XtcData::Shape shapeCube(unsigned rawDefIndex, unsigned valueIndex, XtcData::DescData& rawData);
    /** Generic cube accumulation without calibration: add the value or array entry valueIndex of rawData, converted to double, into bin bin of dst (entries of any XTC numeric type). subIndex is not used. Returns the size of one bin entry in bytes. */
    virtual unsigned addToCube(unsigned rawDefIndex, unsigned valueIndex, unsigned subIndex, 
                               double* dst, unsigned bin, XtcData::DescData& rawData);
    /** Return 0 unless overridden. */
    virtual unsigned subIndices    () { return 0; }
    /** Return 0 unless overridden. */
    virtual unsigned rawNamesIndex () { return 0; }
    /** Return 0 unless overridden. */
    virtual unsigned cubeNamesIndex() { return 0; }
    /** Return ten times the pebble buffer size unless overridden. */
    virtual unsigned cubeBinBytes  () { return m_pool->bufferSize()*10; }
    /** Log an error and abort; detectors used for cube binning must override it. */
    virtual std::vector<XtcData::VarDef>& rawDef();
    /** Return the pebble buffer size, times 10 when Parameters::nCubeWorkers is non-zero. */
    virtual unsigned maxMonBufSize () { return (m_para->nCubeWorkers==0 ? 1:10) * m_pool->pebble.bufferSize(); }
    //
    /** Does nothing unless overridden. */
    virtual void shutdown() {};

    // Scan methods.  Default is to fail.
    /** Return 1 (failure) unless overridden, per the code comment. */
    virtual unsigned configureScan(const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) {return 1;};
    /** Return 1 (failure) unless overridden. */
    virtual unsigned stepScan     (const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) {return 1;};

    /** Return the start of DMA buffer index as a TimingHeader pointer. */
    virtual Pds::TimingHeader* getTimingHeader(uint32_t index) const
    {
        return static_cast<Pds::TimingHeader*>(m_pool->dmaBuffers[index]);
    }
    /** Return false unless overridden. */
    virtual bool scanEnabled() {return false;}
    /** Return the start of the transition buffer as an Xtc reference. */
    XtcData::Xtc& transitionXtc() {return *reinterpret_cast<XtcData::Xtc*>(m_xtcbuf.data());}
    /** Return the end of the transition buffer. */
    const void*   trXtcBufEnd()   {return m_xtcbuf.data() + m_xtcbuf.size();}
    /** Return the names lookup table of this detector. */
    XtcData::NamesLookup& namesLookup() {return m_namesLookup;}
    unsigned nodeId;  ///< Node ID used as the XTC source; -1u until set.
    unsigned virtChan;  ///< Virtual channel number; 0 unless a subclass sets it.
protected:
    Parameters* m_para;
    MemPool* m_pool;
    XtcData::NamesLookup m_namesLookup;
    std::vector<char> m_xtcbuf;
};

template <typename T>
/** Creates objects derived from T by detector type name (Parameters::detType). */
class Factory
{
public:
    template <typename TDerived>
    /** Register TDerived (which must derive from T, checked by a static_assert) under name. */
    void register_type(const std::string& name)
    {
        static_assert(std::is_base_of<T, TDerived>::value,
                      "Factory::register_type doesn't accept this type because doesn't derive from base class");
        m_create_funcs[name] = &createFunc<TDerived>;
    }

    /** Create a new object of the type registered for para->detType with (para, pool), or return nullptr if none is registered. */
    T* create(Parameters* para, MemPool* pool)
    {
        std::string name = para->detType;
        auto it = m_create_funcs.find(name);
        if (it != m_create_funcs.end()) {
            return it->second(para, pool);
        }
        return nullptr;
    }

private:
    template <typename TDerived>
    static T* createFunc(Parameters* para, MemPool* pool)
    {
        return new TDerived(para, pool);
    }

    typedef T* (*PCreateFunc)(Parameters* para, MemPool* pool);
    std::unordered_map<std::string, PCreateFunc> m_create_funcs;
};

}
