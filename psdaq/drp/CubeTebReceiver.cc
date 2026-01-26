#include "CubeTebReceiver.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/Smd.hh"
#include "psdaq/eb/CubeResultDgram.hh"

using namespace XtcData;
using namespace Drp;
using namespace Pds;
using namespace Pds::Eb;
using logging = psalg::SysLog;
using us_t = std::chrono::microseconds;


CubeTebReceiver::CubeTebReceiver(const Parameters& para, DrpBase& drp) :
    TebReceiverBase(para, drp),
    m_mon       (drp.mebContributor()),
#ifdef USE_TEB_COLLECTOR
    m_para      (para),
    m_collector (para, m_pool)
#else
    m_para      (para),
    m_fileWriter(std::max(m_pool.pebble.bufferSize(), para.maxTrSize),
                 const_cast<Parameters&>(para).kwargs["directIO"] != "no"),
    m_smdWriter (std::max(m_pool.pebble.bufferSize(), para.maxTrSize), para.maxTrSize)
#endif
{
}

int CubeTebReceiver::setupMetrics(const std::shared_ptr<Pds::MetricExporter> exporter,
                                  std::map<std::string, std::string>&        labels)
{
#ifdef USE_TEB_COLLECTOR
#define fileWriterRef  m_collector.fileWriter()
#define smdWriterRef   m_collector.smdWriter()
#else
#define fileWriterRef  m_fileWriter
#define smdWriterRef   m_smdWriter
#endif

    exporter->constant("DRP_RecordDepthMax", labels, fileWriterRef.size());
    exporter->add("DRP_RecordDepth", labels, Pds::MetricType::Gauge, [&](){ return fileWriterRef.depth(); });
    exporter->add("DRP_smdWriting",  labels, Pds::MetricType::Gauge, [&](){ return smdWriterRef.writing(); });
    exporter->add("DRP_fileWriting", labels, Pds::MetricType::Gauge, [&](){ return fileWriterRef.writing(); });
    exporter->add("DRP_bufFreeBlk",  labels, Pds::MetricType::Gauge, [&](){ return fileWriterRef.freeBlocked(); });
    exporter->add("DRP_bufPendBlk",  labels, Pds::MetricType::Gauge, [&](){ return fileWriterRef.pendBlocked(); });

    return 0;
}

//
//  Handles result
//    Queues binning summation to workers, if indicated
//    Queue event to CubeCollector
//    Post to monitoring
//
void CubeTebReceiver::complete(unsigned index, const ResultDgram& res)
{
    // This function is called by the base class's process() method to complete
    // processing and dispose of the event.  It presumes that the caller has
    // already vetted index and result
    const Pds::Eb::CubeResultDgram& result = reinterpret_cast<const Pds::Eb::CubeResultDgram&>(res);
    logging::debug("CubeTebReceiver::complete result data(%x) persist(%c) monitor(%x) bin(%u) worker(%u) record(%c)", result.data(), result.persist() ? 'Y':'N', result.monitor(), result.bin(), result.worker(), result.record() ? 'Y':'N');

    TransitionId::Value transitionId = result.service();
    auto dgram = transitionId == TransitionId::L1Accept ? (EbDgram*)m_pool.pebble[index]
                                                        : m_pool.transitionDgrams[index];

    m_evtSize = sizeof(Dgram) + dgram->xtc.sizeofPayload();

    // Measure latency before sending dgram for monitoring
    if (dgram->pulseId() - m_latPid > 1300000/14) { // 10 Hz
        m_latency = Pds::Eb::latency<us_t>(dgram->time);
        m_latPid = dgram->pulseId();
    }

    if (m_mon.enabled()) {
        // L1Accept
        if (result.isEvent()) {
            if (result.monitor()) {
                m_mon.post(dgram, result.monBufNo());
            }
        }
        // Other Transition
        else {
            m_mon.post(dgram);
        }
    }

#ifdef USE_TEB_COLLECTOR
    _collector->queueDgram(index, result); // copies the result
}
#else
    if (writing()) {                    // Won't ever be true for Configure
        // write event to file if it passes event builder or if it's a transition
        if (transitionId != TransitionId::L1Accept) {
            if (transitionId == TransitionId::BeginRun) {
                offsetReset(); // reset offset when writing out a new file
                _writeDgram(reinterpret_cast<Dgram*>(m_configureBuffer.data()));
            }
            _writeDgram(dgram);
            if ((transitionId == TransitionId::Enable) && m_chunkRequest) {
                logging::debug("%s calling reopenFiles()", __PRETTY_FUNCTION__);
                reopenFiles();
            } else if (transitionId == TransitionId::EndRun) {
                logging::debug("%s calling closeFiles()", __PRETTY_FUNCTION__);
                closeFiles();
            }
        }
    }

    // Free the transition datagram buffer
    if (!dgram->isEvent()) {
        m_pool.freeTr(dgram);
    }
        
    // Free the pebble datagram buffer
    m_pool.freePebble(index);
}

void CubeTebReceiver::_writeDgram(Dgram* dgram)
{
    size_t size = sizeof(*dgram) + dgram->xtc.sizeofPayload();
    m_fileWriter.writeEvent(dgram, size, dgram->time);

    // small data writing
    Smd smd;
    const void* bufEnd = m_smdWriter.buffer.data() + m_smdWriter.buffer.size();
    NamesId namesId(dgram->xtc.src.value(), NamesIndex::OFFSETINFO);
    Dgram* smdDgram = smd.generate(dgram, m_smdWriter.buffer.data(), bufEnd, chunkSize(), size,
                                   m_smdWriter.namesLookup, namesId);
    m_smdWriter.writeEvent(smdDgram, sizeof(Dgram) + smdDgram->xtc.sizeofPayload(), smdDgram->time);
    offsetAppend(size);
}
#endif

#ifdef USE_TEB_COLLECTOR
CubeCollector::CubeCollector(const Params& params, MemPool& memPool) :
    m_para(params),
    m_pool(memPool),
    m_last(-1),
    m_current(-1),
    m_fileWriter(std::max(m_pool.pebble.bufferSize(), para.maxTrSize),
                 const_cast<Parameters&>(para).kwargs["directIO"] != "no"), // Default to "yes"
    m_smdWriter (std::max(m_pool.pebble.bufferSize(), para.maxTrSize), para.maxTrSize),
{
}

//
//  Recording intermediate summations requires some serialization
//
void CubeCollector::_writeDgram(Dgram* dgram)
{
    size_t size = sizeof(*dgram) + dgram->xtc.sizeofPayload();
    m_fileWriter.writeEvent(dgram, size, dgram->time);

    // small data writing
    Smd smd;
    const void* bufEnd = m_smdWriter.buffer.data() + m_smdWriter.buffer.size();
    NamesId namesId(dgram->xtc.src.value(), NamesIndex::OFFSETINFO);
    Dgram* smdDgram = smd.generate(dgram, m_smdWriter.buffer.data(), bufEnd, chunkSize(), size,
                                   m_smdWriter.namesLookup, namesId);
    m_smdWriter.writeEvent(smdDgram, sizeof(Dgram) + smdDgram->xtc.sizeofPayload(), smdDgram->time);
    offsetAppend(size);
}

void CubeCollector::queueDgram(unsigned index, const Pds::Eb::CubeResultDgram& result)
{
    m_result[index] = result; // must copy
    m_last = index;

    unsigned worker = result.worker();
    m_workerQueues[worker].push(index, result); // must copy
}

const char* CubeCollector::_wait_for_complete(const Pds::Eb::CubeResultDgram& result)
{
    if (result.persist()) {
        unsigned worker = result.worker();
        while (m_last_pid[worker] < result.pulseId())
            ;
        if (result.record_bin()) {
            return m_worker_payload[result.worker()].pop();
        }
    }
    return 0;
}

void CubeCollector::_process()
{
    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) [[unlikely]] {
            break;
        }

        if (m_current == m_last)
            continue;

        m_current = (m_current + 1) % (m_pool.nbuffers() - 1);

        unsigned index = m_current;
        //  Need to make sure the worker is done with this buffer
        const ResultDgram& result = m_result[index];
        const char* payload = _wait_for_complete(result);

        TransitionId::Value transitionId = result.service();
        auto dgram = transitionId == TransitionId::L1Accept ? (EbDgram*)m_pool.pebble[index]
            : m_pool.transitionDgrams[index];

        if (writing()) {                    // Won't ever be true for Configure
            // write event to file if it passes event builder or if it's a transition
            if (result.persist() || result.prescale()) {
                _writeDgram(result,payload);
                if (payload)
                    m_worker_sem[result.worker()].give();
            }
            else if (transitionId != TransitionId::L1Accept) {
                if (transitionId == TransitionId::BeginRun) {
                    offsetReset(); // reset offset when writing out a new file
                    _writeDgram(reinterpret_cast<Dgram*>(m_configureBuffer.data()));
                }
                _writeDgram(result);
                if ((transitionId == TransitionId::Enable) && m_chunkRequest) {
                    logging::debug("%s calling reopenFiles()", __PRETTY_FUNCTION__);
                    reopenFiles();
                } else if (transitionId == TransitionId::EndRun) {
                    //  This might be the place to write the cube

                    logging::debug("%s calling closeFiles()", __PRETTY_FUNCTION__);
                    closeFiles();
                }
            }
        }

        // Free the transition datagram buffer
        if (!dgram->isEvent()) {
            m_pool.freeTr(dgram);
        }
        
        // Free the pebble datagram buffer
        m_pool.freePebble(index);
    }
}
#endif
