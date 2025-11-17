#include "TebReceiver.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/Smd.hh"
#include "psdaq/eb/ResultDgram.hh"

using namespace XtcData;
using namespace Drp;
using namespace Pds;
using namespace Pds::Eb;
using logging = psalg::SysLog;
using us_t = std::chrono::microseconds;


TebReceiver::TebReceiver(const Parameters& para, DrpBase& drp) :
    TebReceiverBase(para, drp),
    m_mon       (drp.mebContributor()),
    m_fileWriter(std::max(m_pool.pebble.bufferSize(), para.maxTrSize),
                 const_cast<Parameters&>(para).kwargs["directIO"] != "no"), // Default to "yes"
    m_smdWriter (std::max(m_pool.pebble.bufferSize(), para.maxTrSize), para.maxTrSize),
    m_para      (para)
{
}

int TebReceiver::setupMetrics(const std::shared_ptr<Pds::MetricExporter> exporter,
                             std::map<std::string, std::string>&        labels)
{
    exporter->constant("DRP_RecordDepthMax", labels, m_fileWriter.size());
    exporter->add("DRP_RecordDepth", labels, Pds::MetricType::Gauge, [&](){ return m_fileWriter.depth(); });
    exporter->add("DRP_smdWriting",  labels, Pds::MetricType::Gauge, [&](){ return m_smdWriter.writing(); });
    exporter->add("DRP_fileWriting", labels, Pds::MetricType::Gauge, [&](){ return m_fileWriter.writing(); });
    exporter->add("DRP_bufFreeBlk",  labels, Pds::MetricType::Gauge, [&](){ return m_fileWriter.freeBlocked(); });
    exporter->add("DRP_bufPendBlk",  labels, Pds::MetricType::Gauge, [&](){ return m_fileWriter.pendBlocked(); });

    return 0;
}

void TebReceiver::_writeDgram(Dgram* dgram)
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

void TebReceiver::complete(unsigned index, const ResultDgram& result)
{
    // This function is called by the base class's process() method to complete
    // processing and dispose of the event.  It presumes that the caller has
    // already vetted index and result
    TransitionId::Value transitionId = result.service();
    auto dgram = transitionId == TransitionId::L1Accept ? (EbDgram*)m_pool.pebble[index]
                                                        : m_pool.transitionDgrams[index];

    if (writing()) {                    // Won't ever be true for Configure
        // write event to file if it passes event builder or if it's a transition
        if (result.persist() || result.prescale()) {
            _writeDgram(dgram);
        }
        else if (transitionId != TransitionId::L1Accept) {
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

#if 0  // For "Pause/Resume" deadtime test:
    // For this test, SlowUpdates either need to obey deadtime or be turned off.
    // Also, the TEB and MEB must not time out events.
    if (dgram->xtc.src.value() == 0) {  // Do this on only one DRP
        static auto _t0(tp);
        static bool _enabled(false);
        if (transitionId == TransitionId::Enable) {
            _t0 = tp;
            _enabled = true;
        }
        if (_enabled && (tp - _t0 > std::chrono::seconds(1 * 60))) { // Delay a bit before sleeping
            printf("*** TebReceiver: Inducing deadtime by sleeping for 30s at PID %014lx, ts %9u.%09u\n",
                   pulseId, dgram->time.seconds(), dgram->time.nanoseconds());
            std::this_thread::sleep_for(std::chrono::seconds(30));
            _t0 = tp;
            _enabled = false;
            printf("*** TebReceiver: Continuing after sleeping for 30s\n");
        }
    }
#endif

    // Free the transition datagram buffer
    if (!dgram->isEvent()) {
        m_pool.freeTr(dgram);
    }

    // Free the pebble datagram buffer
    m_pool.freePebble(index);
}
