#include "PvaDetector.hh"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <getopt.h>
#include <cassert>
#include <bitset>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <map>
#include <algorithm>
#include <limits>
#include <thread>
#include <Python.h>
#include "DataDriver.h"
#include "RunInfoDef.hh"
#include "xtcdata/xtc/Damage.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psalg/utils/SysLog.hh"

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using json = nlohmann::json;
using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;

static const XtcData::TimeStamp TimeMax(std::numeric_limits<unsigned>::max(),
                                        std::numeric_limits<unsigned>::max());
static unsigned tsMatchDegree = 2;

//
//  Put all the ugliness of non-global timestamps here
//
static int _compare(const XtcData::TimeStamp& ts1,
                    const XtcData::TimeStamp& ts2) {
  int result = 0;

  if ((tsMatchDegree == 0) && !(ts2 == TimeMax))
      return result;

  if (tsMatchDegree == 1) {
    /*
    **  Mask out the fiducial
    */
    const uint64_t mask = 0xfffffffffffe0000ULL;
    uint64_t ts1m = ts1.value()&mask;
    uint64_t ts2m = ts2.value()&mask;

    const uint64_t delta = 10000000; // 10 ms!
    if      (ts1m > ts2m)  result = ts1m - ts2m > delta ?  1 : 0;
    else if (ts2m > ts1m)  result = ts2m - ts1m > delta ? -1 : 0;

    return result;
  }

  if      (ts1 > ts2) result = 1;
  else if (ts2 > ts1) result = -1;
  return result;
}

namespace Drp {

static const XtcData::Name::DataType xtype[] = {
  XtcData::Name::UINT8 , // pvBoolean
  XtcData::Name::INT8  , // pvByte
  XtcData::Name::INT16 , // pvShort
  XtcData::Name::INT32 , // pvInt
  XtcData::Name::INT64 , // pvLong
  XtcData::Name::UINT8 , // pvUByte
  XtcData::Name::UINT16, // pvUShort
  XtcData::Name::UINT32, // pvUInt
  XtcData::Name::UINT64, // pvULong
  XtcData::Name::FLOAT , // pvFloat
  XtcData::Name::DOUBLE, // pvDouble
  XtcData::Name::CHARSTR, // pvString
};

PvaMonitor::PvaMonitor(const Parameters&  para,
                       const std::string& pvName,
                       const std::string& provider,
                       const std::string& request,
                       const std::string& field) :
  Pds_Epics::PvMonitorBase(pvName, provider, request, field),
  m_para                  (para),
  m_state                 (NotReady),
  m_pvaDetector           (nullptr)
{
}

void PvaMonitor::clear()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    disconnect();
    m_pvaDetector = nullptr;
    m_state = NotReady;
    reconnect();
}

int PvaMonitor::getVarDef(PvaDetector*     pvaDetector,
                          XtcData::VarDef& varDef,
                          size_t&          payloadSize,
                          size_t           rankHack)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    const std::chrono::seconds tmo(3);
    m_condition.wait_for(lock, tmo, [this] { return m_state == Ready; });
    if (m_state != Ready)  return 1;

    m_pvaDetector = pvaDetector;

    size_t rank = m_rank;
    if (rankHack != size_t(-1))
    {
      rank = rankHack; // Revisit: Hack!
      logging::warning("%s rank overridden from %zu to %zu\n",
                       name().c_str(), m_rank, rank);
    }

    auto xtcType = xtype[m_type];
    varDef.NameVec.push_back(XtcData::Name(m_fieldName.c_str(), xtcType, rank));

    payloadSize = m_nelem * XtcData::Name::get_element_size(xtcType);

    return 0;
}

void PvaMonitor::onConnect()
{
    logging::info("%s connected", name().c_str());

    if (m_para.verbose) {
        if (printStructure())
            logging::error("onConnect: printStructure() failed");
    }
}

void PvaMonitor::onDisconnect()
{
    logging::info("%s disconnected", name().c_str());

    // Try to bring the connection up again
    printf("*** Calling disconnect()\n");
    disconnect();
    m_state = NotReady;
    printf("*** Calling reconnect()\n");
    reconnect();
}

void PvaMonitor::updated()
{
    if (m_state == Ready) {
        int64_t seconds;
        int32_t nanoseconds;
        getTimestamp(seconds, nanoseconds);
        XtcData::TimeStamp timestamp(seconds, nanoseconds);
        //static XtcData::TimeStamp ts_prv(0, 0);
        //
        //if (timestamp > ts_prv) {
        if (m_pvaDetector)  m_pvaDetector->process(timestamp);
        //}
        //else {
        //  printf("Updated: ts didn't advance: new %016lx  prv %016lx  d %ld\n",
        //         timestamp.value(), ts_prv.value(), timestamp.to_ns() - ts_prv.to_ns());
        //}
        //ts_prv = timestamp;
        //
        //if (nanoseconds > 1000000000) {
        //  printf("Updated: nsec > 1 second: %016lx  s %ld  ns %d\n",
        //         timestamp.value(), seconds, nanoseconds);
        //}
        //
        //if ((timestamp.to_ns() > ts_prv.to_ns()) &&
        //    !(timestamp.value() > ts_prv.value())) {
        //  printf("Updated: > disagreement: ts to_ns %016lx  val %016lx\n"
        //         "                        prv to_ns %016lx  val %016lx\n",
        //         timestamp.to_ns(), timestamp.value(),
        //         ts_prv.to_ns(), ts_prv.value());
        //}
    }
    else {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (getParams(m_type, m_nelem, m_rank))  {
            logging::error("updated: getParams() failed");
        }
        else {
            m_state = Ready;
        }
        m_condition.notify_one();
    }
}


class Pgp
{
public:
    Pgp(const Parameters& para, DrpBase& drp, const bool& running) :
        m_para(para), m_pool(drp.pool), m_tebContributor(drp.tebContributor()), m_running(running),
        m_available(0), m_current(0), m_lastComplete(0), m_latency(0), m_nDmaRet(0)
    {
        m_nodeId = drp.nodeId();
        if (drp.pool.setMaskBytes(para.laneMask, 0)) {
            logging::error("Failed to allocate lane/vc");
        }
    }

    Pds::EbDgram* next(uint32_t& evtIndex, uint64_t& bytes);
    const int64_t latency() { return m_latency; }
    const uint64_t nDmaRet() { return m_nDmaRet; }
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex, uint64_t& bytes);
    const Parameters& m_para;
    MemPool& m_pool;
    Pds::Eb::TebContributor& m_tebContributor;
    static const int MAX_RET_CNT_C = 100;
    int32_t dmaRet[MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dest[MAX_RET_CNT_C];
    const bool& m_running;
    int32_t m_available;
    int32_t m_current;
    uint32_t m_lastComplete;
    XtcData::TransitionId::Value m_lastTid;
    uint32_t m_lastData[6];
    unsigned m_nodeId;
    int64_t m_latency;
    uint64_t m_nDmaRet;
};

Pds::EbDgram* Pgp::_handle(uint32_t& current, uint64_t& bytes)
{
    uint32_t size = dmaRet[m_current];
    uint32_t index = dmaIndex[m_current];
    uint32_t lane = (dest[m_current] >> 8) & 7;
    bytes += size;
    if (size > m_pool.dmaSize()) {
        logging::critical("DMA overflowed buffer: %d vs %d", size, m_pool.dmaSize());
        throw "DMA overflowed buffer";
    }

    const uint32_t* data = (uint32_t*)m_pool.dmaBuffers[index];
    uint32_t evtCounter = data[5] & 0xffffff;
    const unsigned bufferMask = m_pool.nbuffers() - 1;
    current = evtCounter & bufferMask;
    PGPEvent* event = &m_pool.pgpEvents[current];
    // Revisit: Doesn't always work?  assert(event->mask == 0);

    DmaBuffer* buffer = &event->buffers[lane];
    buffer->size = size;
    buffer->index = index;
    event->mask |= (1 << lane);

    logging::debug("PGPReader  lane %u  size %u  hdr %016lx.%016lx.%08x",
                   lane, size,
                   reinterpret_cast<const uint64_t*>(data)[0],
                   reinterpret_cast<const uint64_t*>(data)[1],
                   reinterpret_cast<const uint32_t*>(data)[4]);

    const Pds::TimingHeader* timingHeader = reinterpret_cast<const Pds::TimingHeader*>(data);
    if (timingHeader->error()) {
        logging::error("Timing header error bit is set");
    }
    XtcData::TransitionId::Value transitionId = timingHeader->service();
    if (transitionId != XtcData::TransitionId::L1Accept) {
        if ( transitionId != XtcData::TransitionId::SlowUpdate) {
            logging::info("PGPReader  saw %s @ %u.%09u (%014lx)",
                          XtcData::TransitionId::name(transitionId),
                          timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                          timingHeader->pulseId());
        }
        else {
            logging::debug("PGPReader  saw %s @ %u.%09u (%014lx)",
                           XtcData::TransitionId::name(transitionId),
                           timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                           timingHeader->pulseId());
        }
        if (transitionId == XtcData::TransitionId::BeginRun) {
            m_lastComplete = 0;  // EvtCounter reset
        }
    }
    if (evtCounter != ((m_lastComplete + 1) & 0xffffff)) {
        logging::critical("%sPGPReader: Jump in complete l1Count %u -> %u | difference %d, tid %s%s",
                          RED_ON, m_lastComplete, evtCounter, evtCounter - m_lastComplete, XtcData::TransitionId::name(transitionId), RED_OFF);
        logging::critical("data: %08x %08x %08x %08x %08x %08x",
                          data[0], data[1], data[2], data[3], data[4], data[5]);

        logging::critical("lastTid %s", XtcData::TransitionId::name(m_lastTid));
        logging::critical("lastData: %08x %08x %08x %08x %08x %08x",
                          m_lastData[0], m_lastData[1], m_lastData[2], m_lastData[3], m_lastData[4], m_lastData[5]);

        throw "Jump in event counter";

        for (unsigned e=m_lastComplete+1; e<evtCounter; e++) {
            PGPEvent* brokenEvent = &m_pool.pgpEvents[e & bufferMask];
            logging::error("broken event:  %08x", brokenEvent->mask);
            brokenEvent->mask = 0;

        }
    }
    m_lastComplete = evtCounter;
    m_lastTid = transitionId;
    memcpy(m_lastData, data, 24);

    auto now = std::chrono::system_clock::now();
    auto dgt = std::chrono::seconds{timingHeader->time.seconds() + POSIX_TIME_AT_EPICS_EPOCH}
             + std::chrono::nanoseconds{timingHeader->time.nanoseconds()};
    std::chrono::system_clock::time_point tp{std::chrono::duration_cast<std::chrono::system_clock::duration>(dgt)};
    m_latency = std::chrono::duration_cast<ms_t>(now - tp).count();

    // make new dgram in the pebble
    // It must be an EbDgram in order to be able to send it to the MEB
    Pds::EbDgram* dgram = new(m_pool.pebble[current]) Pds::EbDgram(*timingHeader, XtcData::Src(m_nodeId), m_para.rogMask);

    return dgram;
}

Pds::EbDgram* Pgp::next(uint32_t& evtIndex, uint64_t& bytes)
{
    // get new buffers
    if (m_current == m_available) {
        m_current = 0;
        auto start = std::chrono::steady_clock::now();
        while (true) {
            m_available = dmaReadBulkIndex(m_pool.fd(), MAX_RET_CNT_C, dmaRet, dmaIndex, NULL, NULL, dest);
            m_nDmaRet = m_available;
            if (m_available > 0) {
                m_pool.allocate(m_available);
                break;
            }

            // wait for a total of 10 ms otherwise timeout
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed > 10) {
                //if (m_running)  logging::debug("pgp timeout");
                return nullptr;
            }
        }
    }

    Pds::EbDgram* dgram = _handle(evtIndex, bytes);
    m_current++;
    return dgram;
}


PvaDetector::PvaDetector(Parameters& para, std::shared_ptr<PvaMonitor>& pvaMonitor, DrpBase& drp) :
    XpmDetector     (&para, &drp.pool),
    m_drp           (drp),
    m_pvaMonitor    (pvaMonitor),
    m_pgpQueue      (drp.pool.nbuffers()),
    m_pvQueue       (drp.pool.nbuffers()),
    m_bufferFreelist(m_pvQueue.size()),
    m_terminate     (false),
    m_running       (false),
    m_firstDimKw    (0)
{
    auto firstDimKw = para.kwargs["firstdim"];
    if (!firstDimKw.empty())
        m_firstDimKw = std::stoul(firstDimKw);
}

//std::string PvaDetector::sconfigure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd)
unsigned PvaDetector::configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd)
{
    logging::info("PVA configure");

    if (XpmDetector::configure(config_alias, xtc, bufEnd))
        return 1;

    if (m_exporter)  m_exporter.reset();
    m_exporter = std::make_shared<Pds::MetricExporter>();
    if (m_drp.exposer()) {
        m_drp.exposer()->RegisterCollectable(m_exporter);
    }

    XtcData::Alg     rawAlg("raw", 1, 0, 0);
    XtcData::NamesId rawNamesId(nodeId, RawNamesIndex);
    XtcData::Names&  rawNames = *new(xtc, bufEnd) XtcData::Names(bufEnd,
                                                                 m_para->detName.c_str(), rawAlg,
                                                                 m_para->detType.c_str(), m_para->serNo.c_str(), rawNamesId);
    size_t           payloadSize;
    XtcData::VarDef  rawVarDef;
    size_t           rankHack = m_firstDimKw != 0 ? 2 : -1; // Revisit: Hack!
    if (m_pvaMonitor->getVarDef(this, rawVarDef, payloadSize, rankHack)) {
        logging::error("Failed to connect with %s", m_pvaMonitor->name().c_str());
        return 1;
    }
    payloadSize += (sizeof(Pds::EbDgram)    + // An EbDgram is needed by the MEB
                    24                      + // Space needed by DescribedData
                    sizeof(XtcData::Shapes) + // Needed by DescribedData
                    sizeof(XtcData::Shape));  // Also need 1 of these per PV
    if (payloadSize > m_pool->pebble.bufferSize()) {
        logging::warning("Increase Pebble buffer size from %zd to %zd to avoid truncation of %s data",
                         m_pool->pebble.bufferSize(), payloadSize, m_pvaMonitor->name().c_str());
    }
    rawNames.add(xtc, bufEnd, rawVarDef);
    m_namesLookup[rawNamesId] = XtcData::NameIndex(rawNames);

    XtcData::Alg     infoAlg("epicsinfo", 1, 0, 0);
    XtcData::NamesId infoNamesId(nodeId, InfoNamesIndex);
    XtcData::Names&  infoNames = *new(xtc, bufEnd) XtcData::Names(bufEnd,
                                                                  "epicsinfo", infoAlg,
                                                                  "epicsinfo", "detnum1234", infoNamesId);
    XtcData::VarDef  infoVarDef;
    infoVarDef.NameVec.push_back({"keys", XtcData::Name::CHARSTR, 1});
    infoVarDef.NameVec.push_back({m_para->detName.c_str(), XtcData::Name::CHARSTR, 1});
    infoNames.add(xtc, bufEnd, infoVarDef);
    m_namesLookup[infoNamesId] = XtcData::NameIndex(infoNames);

    // add dictionary of information for each epics detname above.
    // first name is required to be "keys".  keys and values
    // are delimited by ",".
    XtcData::CreateData epicsInfo(xtc, bufEnd, m_namesLookup, infoNamesId);
    epicsInfo.set_string(0, "epicsname"); //  "," "provider");
    epicsInfo.set_string(1, (m_pvaMonitor->name()).c_str()); // + "," + provider).c_str());

    // (Re)initialize the queues
    m_pvQueue.startup();
    m_pgpQueue.startup();
    m_bufferFreelist.startup();
    size_t bufSize = m_pool->pebble.bufferSize();
    m_buffer.resize(m_pvQueue.size() * bufSize);
    for(unsigned i = 0; i < m_pvQueue.size(); ++i) {
        m_bufferFreelist.push(reinterpret_cast<XtcData::Dgram*>(&m_buffer[i * bufSize]));
    }

    m_terminate.store(false, std::memory_order_release);

    m_workerThread = std::thread{&PvaDetector::_worker, this};

    //    return std::string();
    return 0;
}

unsigned PvaDetector::unconfigure()
{
    m_terminate.store(true, std::memory_order_release);
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
    m_pvQueue.shutdown();
    m_pgpQueue.shutdown();
    m_bufferFreelist.shutdown();
    m_pvaMonitor->clear();   // Start afresh
    m_namesLookup.clear();   // erase all elements

    return 0;
}

void PvaDetector::event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* pgpEvent)
{
    XtcData::NamesId namesId(nodeId, RawNamesIndex);
    XtcData::DescribedData desc(dgram.xtc, bufEnd, m_namesLookup, namesId);
    auto ohSize      = (sizeof(Pds::EbDgram)      +
                        dgram.xtc.sizeofPayload() + // = the '24' in configure()
                        sizeof(XtcData::Shapes)   +
                        sizeof(XtcData::Shape));
    auto payloadSize = m_pool->pebble.bufferSize() - ohSize; // Subtract overhead
    uint32_t shape[XtcData::MaxRank];
    auto     size    = m_pvaMonitor->getData(desc.data(), payloadSize, shape);
    if (size > payloadSize) {           // Check actual size vs available size
        logging::debug("Truncated: Pebble buffer of size %zu is too small for payload of size %zu for %s\n",
                       m_pool->pebble.bufferSize(), size + ohSize, m_pvaMonitor->name().c_str());
        dgram.xtc.damage.increase(XtcData::Damage::Truncated);
        size = payloadSize;
    }

    desc.set_data_length(size);

    if (m_pvaMonitor->rank() > 0) {
        if (m_firstDimKw != 0) {            // Revisit: Hack!
            shape[1] = shape[0] / m_firstDimKw;
            shape[0] = m_firstDimKw;
        }
        desc.set_array_shape(0, shape);
    }

    //size_t sz = (sizeof(dgram) + dgram.xtc.sizeofPayload()) >> 2;
    //uint32_t* payload = (uint32_t*)dgram.xtc.payload();
    //printf("sz = %zd, size = %zd, extent = %d, szofPyld = %d, pyldIdx = %ld\n", sz, size, dgram.xtc.extent, dgram.xtc.sizeofPayload(), payload - (uint32_t*)&dgram);
    //uint32_t* buf = (uint32_t*)&dgram;
    //for (unsigned i = 0; i < sz; ++i) {
    //  if (&buf[i] == (uint32_t*)&dgram)       printf(  "dgram:   ");
    //  if (&buf[i] == (uint32_t*)payload)      printf("\npayload: ");
    //  if (&buf[i] == (uint32_t*)desc.data())  printf("\ndata:    ");
    //  printf("%08x ", buf[i]);
    //}
    //printf("\n");
}

void PvaDetector::_worker()
{
    // setup monitoring
    std::map<std::string, std::string> labels{{"instrument", m_para->instrument},
                                              {"partition", std::to_string(m_para->partition)},
                                              {"detname", m_para->detName},
                                              {"detseg", std::to_string(m_para->detSegment)},
                                              {"alias", m_para->alias}};
    m_nEvents = 0;
    m_exporter->add("drp_event_rate", labels, Pds::MetricType::Rate,
                    [&](){return m_nEvents;});
    uint64_t bytes = 0L;
    m_exporter->add("drp_pgp_byte_rate", labels, Pds::MetricType::Rate,
                    [&](){return bytes;});
    m_nUpdates = 0;
    m_exporter->add("pva_update_rate", labels, Pds::MetricType::Rate,
                    [&](){return m_nUpdates;});
    m_nMatch = 0;
    m_exporter->add("pva_match_count", labels, Pds::MetricType::Counter,
                    [&](){return m_nMatch;});
    m_nEmpty = 0;
    m_exporter->add("pva_empty_count", labels, Pds::MetricType::Counter,
                    [&](){return m_nEmpty;});
    m_nMissed = 0;
    m_exporter->add("pva_miss_count", labels, Pds::MetricType::Counter,
                    [&](){return m_nMissed;});
    m_nTooOld = 0;
    m_exporter->add("pva_tooOld_count", labels, Pds::MetricType::Counter,
                    [&](){return m_nTooOld;});
    m_nTimedOut = 0;
    m_exporter->add("pva_timeout_count", labels, Pds::MetricType::Counter,
                    [&](){return m_nTimedOut;});
    m_timeDiff = 0;
    m_exporter->add("pva_time_diff", labels, Pds::MetricType::Gauge,
                    [&](){return m_timeDiff;});

    m_exporter->add("drp_worker_input_queue", labels, Pds::MetricType::Gauge,
                    [&](){return m_pgpQueue.guess_size();});
    m_exporter->constant("drp_worker_queue_depth", labels, m_pgpQueue.size());

    // Borrow this for awhile
    m_exporter->add("drp_worker_output_queue", labels, Pds::MetricType::Gauge,
                    [&](){return m_pvQueue.guess_size();});

    Pgp pgp(*m_para, m_drp, m_running);

    m_exporter->add("drp_th_latency", labels, Pds::MetricType::Gauge,
                    [&](){return pgp.latency();});
    m_exporter->add("drp_num_dma_ret", labels, Pds::MetricType::Gauge,
                    [&](){return pgp.nDmaRet();});

    const uint64_t msTmo = m_para->kwargs.find("match_tmo_ms") != m_para->kwargs.end()
                         ? std::stoul(m_para->kwargs["match_tmo_ms"])
                         : 1333; // Avoid event rate multiples and factors

    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }

        uint32_t index;
        Pds::EbDgram* dgram = pgp.next(index, bytes);
        if (dgram) {
            m_nEvents++;

            XtcData::TransitionId::Value service = dgram->service();
            // Also queue SlowUpdates to keep things in time order
            if ((service == XtcData::TransitionId::L1Accept) ||
                (service == XtcData::TransitionId::SlowUpdate)) {
                m_pgpQueue.push(index);

                //printf("                         PGP: %u.%09u\n",
                //       dgram->time.seconds(), dgram->time.nanoseconds());

                _matchUp();

                // Prevent PGP events from stacking up by by timing them out.
                // The maximum timeout is < the TEB event build timeout to keep
                // prompt contributions from timing out before latent ones arrive.
                // If the PV is updating, _timeout() never finds anything to do.
                XtcData::TimeStamp timestamp;
                //const uint64_t msTmo = tsMatchDegree==2 ? 100 : 1000; //4400;
                //const uint64_t msTmo = tsMatchDegree!=1 ? 100 : 1141; // 1s + fid. time for fuzzy ts matching
                //const uint64_t ebTmo = 6000; // This overflows PGP (?) buffers: Pds::Eb::EB_TMO_MS/2 - 100;
                //const uint64_t ebTmo = Pds::Eb::EB_TMO_MS/2 - 100; // This overflows PGP (?) buffers
                //const uint64_t msTmo = tsMatchDegree==2 ? 100 : ebTmo;
                const uint64_t nsTmo = msTmo * 1000000;
                _timeout(timestamp.from_ns(dgram->time.to_ns() - nsTmo));
            }
            else {
                // Allocate a transition dgram from the pool and initialize its header
                Pds::EbDgram* trDgram = m_pool->allocateTr();
                const void*   bufEnd  = (char*)trDgram + m_para->maxTrSize;
                *trDgram = *dgram;
                // copy the temporary xtc created on phase 1 of the transition
                // into the real location
                XtcData::Xtc& trXtc = transitionXtc();
                trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
                auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
                memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());
                PGPEvent* pgpEvent = &m_pool->pgpEvents[index];
                pgpEvent->transitionDgram = trDgram;

                if (service == XtcData::TransitionId::Enable) {
                    m_running = true;
                }
                else if (service == XtcData::TransitionId::Disable) { // Sweep out L1As
                    m_running = false;
                    logging::debug("Sweeping out L1Accepts and SlowUpdates");
                    _timeout(TimeMax);
                }

                _sendToTeb(*dgram, index);
            }
        }
    }
    logging::info("Worker thread finished");
}

void PvaDetector::process(const XtcData::TimeStamp& timestamp)
{
    // Protect against namesLookup not being stable before Enable
    if (m_running.load(std::memory_order_relaxed)) {
        ++m_nUpdates;
        logging::debug("%s updated @ %u.%09u", m_pvaMonitor->name().c_str(), timestamp.seconds(), timestamp.nanoseconds());

        XtcData::Dgram* dgram;
        if (m_bufferFreelist.try_pop(dgram)) { // If a buffer is available...
            //static uint64_t last_ts = 0;
            //uint64_t ts = timestamp.to_ns();
            //int64_t  dT = ts - last_ts;
            //printf("  PV:  %u.%09u, dT %9ld, ts %18lu, last %18lu\n", timestamp.seconds(), timestamp.nanoseconds(), dT, ts, last_ts);
            //if (dT > 0)  last_ts = ts;

            dgram->time = timestamp;           //   Save the PV's timestamp
            dgram->xtc = {{XtcData::TypeId::Parent, 0}, {nodeId}};

            const void* bufEnd = (char*)dgram + m_pool->pebble.bufferSize();
            event(*dgram, bufEnd, nullptr);    // PGPEvent not needed in this case

            m_pvQueue.push(dgram);
        }
        else {
            ++m_nMissed;                       // Else count it as missed
        }
    }
}

void PvaDetector::_matchUp()
{
    while (true) {
        XtcData::Dgram* pvDg;
        if (!m_pvQueue.peek(pvDg))  break;

        uint32_t pgpIdx;
        if (!m_pgpQueue.peek(pgpIdx))  break;

        // Try to drain all but one or two when PV timestamps are being ignored
        // If an additional entry appears, it is left in the queue for next time
        if (tsMatchDegree == 0) {
            auto sz = m_pvQueue.guess_size(); // Size may grow during this loop
            while (--sz) {
                m_pvQueue.try_pop(pvDg);      // Pop and drop oldest
                m_bufferFreelist.push(pvDg);  // Return buffer to freelist
            }
            m_pvQueue.peek(pvDg);             // Proceed with most recent entry
        }

        Pds::EbDgram* pgpDg = reinterpret_cast<Pds::EbDgram*>(m_pool->pebble[pgpIdx]);

        m_timeDiff = pgpDg->time.to_ns() - pvDg->time.to_ns();

        logging::debug("PV: %u.%09d, PGP: %u.%09d, PGP - PV: %10ld ns, svc %2d",
      //printf        ("PV: %u.%09d, PGP: %u.%09d, PGP - PV: %10ld ns, svc %2d",
                       pvDg->time.seconds(), pvDg->time.nanoseconds(),
                       pgpDg->time.seconds(), pgpDg->time.nanoseconds(),
                       m_timeDiff, pgpDg->service());

        //  Mask out fiducial until it's understood
        //        if      (pvDg->time == pgpDg->time)  _handleMatch  (*pvDg, *pgpDg);

        int result = _compare(pvDg->time,pgpDg->time);
        //printf("pv %016lx, pgp %016lx, diff %ld, compare %d\n", pvDg->time.value(), pgpDg->time.value(), pvDg->time.value() - pgpDg->time.value(), _compare(pvDg->time, pgpDg->time));
        if      (result==0) { _handleMatch  (*pvDg, *pgpDg); /*printf("  Matched\n");*/ }
        else if (result >0) { _handleYounger(*pvDg, *pgpDg); /*printf("  Younger\n");*/ }
        else                { _handleOlder  (*pvDg, *pgpDg); /*printf("  Older\n");  */ }

        //_handleMatch  (*pvDg, *pgpDg);
    }
    //printf("\n");
}

void PvaDetector::_handleMatch(const XtcData::Dgram& pvDg, Pds::EbDgram& pgpDg)
{
    logging::debug("PV matches PGP!!  "
                   "TimeStamps: PV %u.%09u == PGP %u.%09u",
                   pvDg.time.seconds(), pvDg.time.nanoseconds(),
                   pgpDg.time.seconds(), pgpDg.time.nanoseconds());

    uint32_t pgpIdx;
    m_pgpQueue.try_pop(pgpIdx);         // Actually consume the element

    XtcData::Dgram* dgram;
    if (pgpDg.service() == XtcData::TransitionId::L1Accept) {
        pgpDg.xtc.damage.increase(pvDg.xtc.damage.value());
        auto bufEnd  = (char*)&pgpDg + m_pool->pebble.bufferSize();
        auto payload = pgpDg.xtc.alloc(pvDg.xtc.sizeofPayload(), bufEnd);
        memcpy(payload, (const void*)pvDg.xtc.payload(), pvDg.xtc.sizeofPayload());

        m_pvQueue.try_pop(dgram);       // Actually consume the element
        m_bufferFreelist.push(dgram);   // Return buffer to freelist

        ++m_nMatch;
    }
    else { // SlowUpdate
        // Allocate a transition dgram from the pool and initialize its header
        Pds::EbDgram* trDg = m_pool->allocateTr();
        *trDg = pgpDg;                  // Initialized Xtc, possibly w/ damage
        PGPEvent* pgpEvent = &m_pool->pgpEvents[pgpIdx];
        pgpEvent->transitionDgram = trDg;

        if (tsMatchDegree == 2) {         // Keep PV for the next L1A
            m_pvQueue.try_pop(dgram);     // Actually consume the element
            m_bufferFreelist.push(dgram); // Return buffer to freelist
        }
    }

    _sendToTeb(pgpDg, pgpIdx);
}

void PvaDetector::_handleYounger(const XtcData::Dgram& pvDg, Pds::EbDgram& pgpDg)
{
    uint32_t pgpIdx;
    m_pgpQueue.try_pop(pgpIdx);       // Actually consume the element

    if (pgpDg.service() == XtcData::TransitionId::L1Accept) {
        // No corresponding PV data so mark event damaged
        pgpDg.xtc.damage.increase(XtcData::Damage::MissingData);

        ++m_nEmpty;

        logging::debug("PV too young!!    "
                       "TimeStamps: PV %u.%09u > PGP %u.%09u",
                       pvDg.time.seconds(), pvDg.time.nanoseconds(),
                       pgpDg.time.seconds(), pgpDg.time.nanoseconds());
    }
    else { // SlowUpdate
        // Allocate a transition dgram from the pool and initialize its header
        Pds::EbDgram* trDg = m_pool->allocateTr();
        *trDg = pgpDg;                  // Initialized Xtc, possibly w/ damage
        PGPEvent* pgpEvent = &m_pool->pgpEvents[pgpIdx];
        pgpEvent->transitionDgram = trDg;
    }

    _sendToTeb(pgpDg, pgpIdx);
}

void PvaDetector::_handleOlder(const XtcData::Dgram& pvDg, Pds::EbDgram& pgpDg)
{
    if (pgpDg.service() == XtcData::TransitionId::L1Accept) {
        ++m_nTooOld;
        logging::debug("PV too old!!      "
                       "TimeStamps: PV %u.%09u < PGP %u.%09u [0x%08x%04x.%05x < 0x%08x%04x.%05x]",
                       pvDg.time.seconds(), pvDg.time.nanoseconds(),
                       pgpDg.time.seconds(), pgpDg.time.nanoseconds(),
                       pvDg.time.seconds(), (pvDg.time.nanoseconds()>>16)&0xfffe, pvDg.time.nanoseconds()&0x1ffff,
                       pgpDg.time.seconds(), (pgpDg.time.nanoseconds()>>16)&0xfffe, pgpDg.time.nanoseconds()&0x1ffff);
    }

    XtcData::Dgram* dgram;
    m_pvQueue.try_pop(dgram);           // Actually consume the element
    m_bufferFreelist.push(dgram);       // Return buffer to freelist
}

void PvaDetector::_timeout(const XtcData::TimeStamp& timestamp)
{
    while (true) {
        uint32_t index;
        if (!m_pgpQueue.peek(index)) {
            break;
        }

        Pds::EbDgram& dgram = *reinterpret_cast<Pds::EbDgram*>(m_pool->pebble[index]);
        if (dgram.time > timestamp) {
            break;                  // dgram is newer than the timeout timestamp
        }

        uint32_t idx;
        m_pgpQueue.try_pop(idx);        // Actually consume the element
        assert(idx == index);

        if (dgram.service() == XtcData::TransitionId::L1Accept) {
          // No PVA data so mark event as damaged
          dgram.xtc.damage.increase(XtcData::Damage::TimedOut);
          ++m_nTimedOut;
          //printf("TO: %u.%09u, PGP: %u.%09u, PGP - TO: %10ld ns, svc %2d  Timeout\n",
          //       timestamp.seconds(), timestamp.nanoseconds(),
          //       dgram.time.seconds(), dgram.time.nanoseconds(),
          //       dgram.time.to_ns() - timestamp.to_ns(),
          //       dgram.service());
          logging::debug("Event timed out!! "
                         "TimeStamps: timeout %u.%09u > PGP %u.%09u [0x%08x%04x.%05x > 0x%08x%04x.%05x]",
                         timestamp.seconds(), timestamp.nanoseconds(),
                         dgram.time.seconds(), dgram.time.nanoseconds(),
                         timestamp.seconds(), (timestamp.nanoseconds()>>16)&0xfffe, timestamp.nanoseconds()&0x1ffff,
                         dgram.time.seconds(), (dgram.time.nanoseconds()>>16)&0xfffe, dgram.time.nanoseconds()&0x1ffff);
        }
        else { // SlowUpdate
            // Allocate a transition dgram from the pool and initialize its header
            Pds::EbDgram* trDg = m_pool->allocateTr();
            *trDg = dgram;              // Initialized Xtc, possibly w/ damage
            PGPEvent* pgpEvent = &m_pool->pgpEvents[index];
            pgpEvent->transitionDgram = trDg;
        }

        _sendToTeb(dgram, index);
    }
}

void PvaDetector::_sendToTeb(const Pds::EbDgram& dgram, uint32_t index)
{
    // Make sure the datagram didn't get too big
    const size_t size = sizeof(dgram) + dgram.xtc.sizeofPayload();
    const size_t maxSize = (dgram.service() == XtcData::TransitionId::L1Accept)
                         ? m_pool->pebble.bufferSize()
                         : m_para->maxTrSize;
    if (size > maxSize) {
        logging::critical("%s Dgram of size %zd overflowed buffer of size %zd", XtcData::TransitionId::name(dgram.service()), size, maxSize);
        throw "Dgram overflowed buffer";
    }

    auto l3InpBuf = m_drp.tebContributor().fetch(index);
    Pds::EbDgram* l3InpDg = new(l3InpBuf) Pds::EbDgram(dgram);
    if (l3InpDg->isEvent()) {
        auto triggerPrimitive = m_drp.triggerPrimitive();
        if (triggerPrimitive) { // else this DRP doesn't provide input
            const void* bufEnd = (char*)l3InpDg + sizeof(*l3InpDg) + triggerPrimitive->size();
            triggerPrimitive->event(*m_pool, index, dgram.xtc, l3InpDg->xtc, bufEnd); // Produce
        }
    }
    m_drp.tebContributor().process(l3InpDg);
}


PvaApp::PvaApp(Parameters& para, std::shared_ptr<PvaMonitor> pvaMonitor) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_drp(para, context()),
    m_para(para),
    m_pvaDetector(std::make_unique<PvaDetector>(m_para, pvaMonitor, m_drp)),
    m_det(m_pvaDetector.get()),
    m_unconfigure(false)
{
    Py_Initialize();                    // for use by configuration

    if (m_det == nullptr) {
        logging::critical("Error !! Could not create Detector object for %s", m_para.detType.c_str());
        throw "Could not create Detector object for " + m_para.detType;
    }
    logging::info("Ready for transitions");
}

PvaApp::~PvaApp()
{
    // Try to take things down gracefully when an exception takes us off the
    // normal path so that the most chance is given for prints to show up
    handleReset(json({}));

    Py_Finalize();                      // for use by configuration
}

void PvaApp::_disconnect()
{
    m_drp.disconnect();
    m_det->shutdown();
}

void PvaApp::_unconfigure()
{
    m_drp.pool.shutdown();  // Release Tr buffer pool
    m_drp.unconfigure();    // TebContributor must be shut down before the worker
    m_pvaDetector->unconfigure();
    m_unconfigure = false;
}

json PvaApp::connectionInfo()
{
    std::string ip = m_para.kwargs.find("ep_domain") != m_para.kwargs.end()
                   ? getNicIp(m_para.kwargs["ep_domain"])
                   : getNicIp(m_para.kwargs["forceEnet"] == "yes");
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json info = m_det->connectionInfo();
    body["connect_info"].update(info);
    json bufInfo = m_drp.connectionInfo(ip);
    body["connect_info"].update(bufInfo);
    return body;
}

void PvaApp::connectionShutdown()
{
    m_drp.shutdown();
}

void PvaApp::_error(const std::string& which, const nlohmann::json& msg, const std::string& errorMsg)
{
    json body = json({});
    body["err_info"] = errorMsg;
    json answer = createMsg(which, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PvaApp::handleConnect(const nlohmann::json& msg)
{
    m_det->nodeId = msg["body"]["drp"][std::to_string(getId())]["drp_id"];
    m_det->connect(msg, std::to_string(getId()));

    std::string errorMsg = m_drp.connect(msg, getId());
    if (!errorMsg.empty()) {
        logging::error("Error in DrpBase::connect");
        logging::error("%s", errorMsg.c_str());
        _error("connect", msg, errorMsg);
        return;
    }

    json body = json({});
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PvaApp::handleDisconnect(const json& msg)
{
    // Carry out the queued Unconfigure, if there was one
    if (m_unconfigure) {
        _unconfigure();
    }

    _disconnect();

    json body = json({});
    reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void PvaApp::handlePhase1(const json& msg)
{
    std::string key = msg["header"]["key"];
    logging::debug("handlePhase1 for %s in PvaDetectorApp", key.c_str());

    XtcData::Xtc& xtc = m_det->transitionXtc();
    xtc = {{XtcData::TypeId::Parent, 0}, {m_det->nodeId}};
    auto bufEnd = m_det->trXtcBufEnd();

    json phase1Info{ "" };
    if (msg.find("body") != msg.end()) {
        if (msg["body"].find("phase1Info") != msg["body"].end()) {
            phase1Info = msg["body"]["phase1Info"];
        }
    }

    json body = json({});

    if (key == "configure") {
        if (m_unconfigure) {
            _unconfigure();
        }

        std::string errorMsg = m_drp.configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        std::string config_alias = msg["body"]["config_alias"];
        unsigned error = m_det->configure(config_alias, xtc, bufEnd);
        if (error) {
            std::string errorMsg = "Failed transition phase 1";
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        m_drp.runInfoSupport(xtc, bufEnd, m_det->namesLookup());
    }
    else if (key == "unconfigure") {
        // "Queue" unconfiguration until after phase 2 has completed
        m_unconfigure = true;
    }
    else if (key == "beginrun") {
        RunInfo runInfo;
        std::string errorMsg = m_drp.beginrun(phase1Info, runInfo);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        else {
            m_drp.runInfoData(xtc, bufEnd, m_det->namesLookup(), runInfo);
        }
    }
    else if (key == "endrun") {
        std::string errorMsg = m_drp.endrun(phase1Info);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
    }
    else if (key == "enable") {
        bool chunkRequest;
        ChunkInfo chunkInfo;
        std::string errorMsg = m_drp.enable(phase1Info, chunkRequest, chunkInfo);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        } else if (chunkRequest) {
            logging::debug("handlePhase1 enable found chunkRequest");
            m_drp.chunkInfoData(xtc, bufEnd, m_det->namesLookup(), chunkInfo);
        }
        logging::debug("handlePhase1 enable complete");
    }

    json answer = createMsg(key, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PvaApp::handleReset(const nlohmann::json& msg)
{
    unsubscribePartition();    // ZMQ_UNSUBSCRIBE
    _unconfigure();
    _disconnect();
    connectionShutdown();
}

} // namespace Drp


int main(int argc, char* argv[])
{
    Drp::Parameters para;
    std::string kwargs_str;
    int c;
    while((c = getopt(argc, argv, "p:o:l:D:S:C:d:u:k:P:T::M:01v")) != EOF) {
        switch(c) {
            case 'p':
                para.partition = std::stoi(optarg);
                break;
            case 'o':
                para.outputDir = optarg;
                break;
            case 'l':
                para.laneMask = std::stoul(optarg, nullptr, 16);
                break;
            case 'D':
                para.detType = optarg;  // Defaults to 'pv'
                break;
            case 'S':
                para.serNo = optarg;
                break;
            case 'u':
                para.alias = optarg;
                break;
            case 'C':
                para.collectionHost = optarg;
                break;
            case 'd':
                para.device = optarg;
                break;
            case 'k':
                kwargs_str = kwargs_str.empty()
                           ? optarg
                           : kwargs_str + ", " + optarg;
                break;
            case 'P':
                para.instrument = optarg;
                break;
            case 'M':
                para.prometheusDir = optarg;
                break;
            //  Indicate level of timestamp matching (ugh)
            case '0':
                tsMatchDegree = 0;
                break;
            case '1':
                fprintf(stderr, "Option -1 is disabled\n");  exit(EXIT_FAILURE);
                tsMatchDegree = 1;
                break;
            case 'v':
                ++para.verbose;
                break;
            default:
                return 1;
        }
    }

    switch (para.verbose) {
        case 0:  logging::init(para.instrument.c_str(), LOG_INFO);   break;
        default: logging::init(para.instrument.c_str(), LOG_DEBUG);  break;
    }
    logging::info("logging configured");
    if (para.instrument.empty()) {
        logging::warning("-P: instrument name is missing");
    }
    // Check required parameters
    if (para.partition == unsigned(-1)) {
        logging::critical("-p: partition is mandatory");
        return 1;
    }
    if (para.device.empty()) {
        logging::critical("-d: device is mandatory");
        return 1;
    }
    if (para.alias.empty()) {
        logging::critical("-u: alias is mandatory");
        return 1;
    }

    // Only one lane is supported by this DRP
    if (std::bitset<PGP_MAX_LANES>(para.laneMask).count() != 1) {
        logging::critical("-l: lane mask must have only 1 bit set");
        return 1;
    }

    // Allow detType to be overridden, but generally, psana will expect 'pv'
    if (para.detType.empty()) {
      para.detType = "pv";
    }

    // Alias must be of form <detName>_<detSegment>
    size_t found = para.alias.rfind('_');
    if ((found == std::string::npos) || !isdigit(para.alias.back())) {
        logging::critical("-u: alias must have _N suffix");
        return 1;
    }
    para.detName = para.alias.substr(0, found);
    para.detSegment = std::stoi(para.alias.substr(found+1, para.alias.size()));

    // Provider is "pva" (default) or "ca"
    std::string pv;                     // [<provider>/]<PV name>[.<field>]
    if (optind < argc)
    {
        pv = argv[optind++];

        if (optind < argc)
        {
            logging::error("Unrecognized argument:");
            while (optind < argc)
                logging::error("  %s ", argv[optind++]);
            return 1;
        }
    }
    else {
        logging::critical("A PV ([<provider>/]<PV name>[.<field>]) is mandatory");
        return 1;
    }

    para.maxTrSize = 256 * 1024;
    try {
        get_kwargs(kwargs_str, para.kwargs);
        for (const auto& kwargs : para.kwargs) {
            if (kwargs.first == "forceEnet")     continue;
            if (kwargs.first == "ep_fabric")     continue;
            if (kwargs.first == "ep_domain")     continue;
            if (kwargs.first == "ep_provider")   continue;
            if (kwargs.first == "sim_length")    continue;  // XpmDetector
            if (kwargs.first == "timebase")      continue;  // XpmDetector
            if (kwargs.first == "pebbleBufSize") continue;  // DrpBase
            if (kwargs.first == "batching")      continue;  // DrpBase
            if (kwargs.first == "directIO")      continue;  // DrpBase
            if (kwargs.first == "firstdim")      continue;
            if (kwargs.first == "match_tmo_ms")  continue;
            logging::critical("Unrecognized kwarg '%s=%s'\n",
                              kwargs.first.c_str(), kwargs.second.c_str());
            return 1;
        }

        std::string provider = "pva";
        std::string field    = "value";
        auto pos = pv.find("/", 0);
        if (pos != std::string::npos) { // Parse provider
            provider = pv.substr(0, pos);
            pv       = pv.substr(pos+1);
        }
        pos = pv.find(".", 0);
        if (pos != std::string::npos) { // Parse field
            field = pv.substr(pos+1);
            pv    = pv.substr(0, pos);
        }
        auto request(provider == "pva" ? "field(value,timeStamp,dimension)"
                                       : "field(value,timeStamp)");
        auto pvaMonitor(std::make_shared<Drp::PvaMonitor>(para, pv, provider, request, field));
        Drp::PvaApp(para, pvaMonitor).run();
        return 0;
    }
    catch (std::exception& e)  { logging::critical("%s", e.what()); }
    catch (std::string& e)     { logging::critical("%s", e.c_str()); }
    catch (char const* e)      { logging::critical("%s", e); }
    catch (...)                { logging::critical("Default exception"); }
    return EXIT_FAILURE;
}
