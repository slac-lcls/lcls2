#include <unistd.h>                     // gethostname(), sysconf()
#include <stdlib.h>                     // posix_memalign()
#include <iostream>
#include <fstream>
#include <iomanip>
#include <bitset>
#include <climits>                      // HOST_NAME_MAX
#include <chrono>
#include <sys/types.h>
#include <sys/stat.h>                   // stat()
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/EbDgram.hh"
#include <DmaDriver.h>
#include "DrpBase.hh"
#include "RunInfoDef.hh"
#include "psalg/utils/SysLog.hh"
#include "xtcdata/xtc/Smd.hh"
#include "DataDriver.h"

#include "rapidjson/document.h"

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using namespace XtcData;
using json = nlohmann::json;
using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;

static void local_mkdir (const char * path);
static json createFileReportMsg(std::string path, std::string absolute_path,
                                timespec create_time, timespec modify_time,
                                unsigned run_num, std::string hostname);
static json createPulseIdMsg(uint64_t pulseId);
static json createChunkRequestMsg();

namespace Drp {

static long readInfinibandCounter(const std::string& counter)
{
    std::string path{"/sys/class/infiniband/mlx5_0/ports/1/counters/" + counter};
    std::ifstream in(path);
    if (in.is_open()) {
        std::string line;
        std::getline(in, line);
        return stol(line);
    }
    else {
        return 0;
    }
}

static unsigned nextPowerOf2(unsigned n)
{
    unsigned count = 0;

    if (n && !(n & (n - 1))) {
        return n;
    }

    while( n != 0) {
        n >>= 1;
        count += 1;
    }

    return 1 << count;
}


void Pebble::create(unsigned nL1Buffers, size_t l1BufSize, unsigned nTrBuffers, size_t trBufSize)
{
    size_t algnSz = 16;                    // For cache boundaries
    m_bufferSize  = algnSz * ((l1BufSize + algnSz - 1) / algnSz);

    size_t pgSz   = sysconf(_SC_PAGESIZE); // For shmem/MMU
    m_size        = nL1Buffers*m_bufferSize + nTrBuffers*trBufSize;
    m_size        = pgSz * ((m_size + pgSz - 1) / pgSz);
    m_buffer      = nullptr;
    int    ret    = posix_memalign((void**)&m_buffer, pgSz, m_size);
    if (ret) {
        logging::critical("Pebble creation of size %zu failed: %s\n", m_size, strerror(ret));
        throw "Pebble creation failed";
    }
}

MemPool::MemPool(Parameters& para) :
    m_transitionBuffers(nextPowerOf2(Pds::Eb::TEB_TR_BUFFERS)), // See eb.hh
    m_inUse(0)
{
    m_fd = open(para.device.c_str(), O_RDWR);
    if (m_fd < 0) {
        logging::critical("Error opening %s: %s", para.device.c_str(), strerror(errno));
        throw "Error opening kcu1500!!";
    }

    uint32_t dmaCount;
    dmaBuffers = dmaMapDma(m_fd, &dmaCount, &m_dmaSize);
    if (dmaBuffers == NULL ) {
        logging::critical("Failed to map dma buffers: %s", strerror(errno));
        throw "Error calling dmaMapDma!!";
    }
    logging::info("dmaCount %u,  dmaSize %u", dmaCount, m_dmaSize);

    // make sure there are more buffers in the pebble than in the pgp driver
    // otherwise the pebble buffers will be overwritten by the pgp event builder
    m_nbuffers = nextPowerOf2(dmaCount);

    // make the size of the pebble buffer that will contain the datagram equal
    // to the dmaSize times the number of lanes
    // Also include space in the pebble for a pool of transition buffers of
    // worst case size so that they will be part of the memory region that can
    // be RDMAed from to the MEB
    size_t maxL1ASize = para.kwargs.find("pebbleBufSize") == para.kwargs.end() // Allow overriding the Pebble size
                      ? __builtin_popcount(para.laneMask) * m_dmaSize
                      : std::stoul(para.kwargs["pebbleBufSize"]);
    auto nTrBuffers = m_transitionBuffers.size();
    pebble.create(m_nbuffers, maxL1ASize, nTrBuffers, para.maxTrSize);
    logging::info("nL1Buffers %u,  pebble buffer size %zu", m_nbuffers, pebble.bufferSize());
    logging::info("nTrBuffers %u,  transition buffer size %zu", nTrBuffers, para.maxTrSize);

    pgpEvents.resize(m_nbuffers);

    // Put the transition buffer pool at the end of the pebble buffers
    uint8_t* buffer = pebble[m_nbuffers];
    for (size_t i = 0; i < m_transitionBuffers.size(); i++) {
        m_transitionBuffers.push(&buffer[i * para.maxTrSize]);
    }
    m_setMaskBytesDone = false;
}

MemPool::~MemPool()
{
   logging::info("%s: closing file descriptor", __PRETTY_FUNCTION__);
   close(m_fd);
}

Pds::EbDgram* MemPool::allocateTr()
{
    void* dgram = nullptr;
    if (!m_transitionBuffers.pop(dgram)) {
        // See comments for setting the number of transition buffers in eb.hh
        return nullptr;
    }
    return static_cast<Pds::EbDgram*>(dgram);
}

void MemPool::shutdown()
{
    m_transitionBuffers.shutdown();
}

int MemPool::setMaskBytes(uint8_t laneMask, unsigned virtChan)
{
    int retval = 0;
    if (m_setMaskBytesDone) {
        logging::info("%s: earlier setting in effect", __PRETTY_FUNCTION__);
    } else {
        uint8_t mask[DMA_MASK_SIZE];
        dmaInitMaskBytes(mask);
        for (unsigned i=0; i<PGP_MAX_LANES; i++) {
            if (laneMask & (1 << i)) {
                uint32_t channel = i;
                uint32_t dest = dmaDest(channel, virtChan);
                logging::info("setting lane  %u, dest 0x%x", i, dest);
                dmaAddMaskBytes(mask, dest);
            }
        }
        if (dmaSetMaskBytes(m_fd, mask)) {
            retval = 1; // error
        } else {
            m_setMaskBytesDone = true;
        }
    }
    return retval;
}

std::string Drp::FileParameters::runName()
{
    std::ostringstream ss;
    ss << m_experimentName <<
          "-r" << std::setfill('0') << std::setw(4) << m_runNumber <<
          "-s" << std::setw(3) << m_nodeId <<
          "-c" << std::setw(3) << m_chunkId;
    return ss.str();
}

EbReceiver::EbReceiver(Parameters& para, Pds::Eb::TebCtrbParams& tPrms,
                       MemPool& pool, ZmqSocket& inprocSend, Pds::Eb::MebContributor& mon,
                       const std::shared_ptr<Pds::MetricExporter>& exporter) :
  EbCtrbInBase(tPrms, exporter),
  m_pool(pool),
  m_det(nullptr),
  m_tsId(-1u),
  m_mon(mon),
  m_fileWriter(std::max(pool.pebble.bufferSize(), para.maxTrSize), para.kwargs["directIO"] == "yes"),
  m_smdWriter(std::max(pool.pebble.bufferSize(), para.maxTrSize)),
  m_writing(false),
  m_inprocSend(inprocSend),
  m_count(0),
  m_offset(0),
  m_chunkOffset(0),
  m_chunkRequest(false),
  m_chunkPending(false),
  m_configureBuffer(para.maxTrSize),
  m_damage(0),
  m_evtSize(0),
  m_latency(0)
{
    std::map<std::string, std::string> labels
        {{"instrument", para.instrument},
         {"partition", std::to_string(para.partition)},
         {"detname", para.detName},
         {"alias", para.alias}};
    exporter->add("DRP_Damage"    ,   labels, Pds::MetricType::Gauge,   [&](){ return m_damage; });
    exporter->add("DRP_RecordSize",   labels, Pds::MetricType::Counter, [&](){ return m_offset; });
    exporter->add("DRP_RecordDepth",  labels, Pds::MetricType::Gauge,   [&](){ return m_fileWriter.depth(); });
    exporter->constant("DRP_RecordDepthMax", labels, m_fileWriter.size());
    m_dmgType = exporter->histogram("DRP_DamageType", labels, 16);
    exporter->add("DRP_smdWriting",   labels, Pds::MetricType::Gauge,   [&](){ return m_smdWriter.writing(); });
    exporter->add("DRP_fileWriting",  labels, Pds::MetricType::Gauge,   [&](){ return m_fileWriter.writing(); });
    exporter->add("DRP_bufFreeBlk",   labels, Pds::MetricType::Gauge,   [&](){ return m_fileWriter.freeBlocked(); });
    exporter->add("DRP_bufPendBlk",   labels, Pds::MetricType::Gauge,   [&](){ return m_fileWriter.pendBlocked(); });
    exporter->add("DRP_evtSize",      labels, Pds::MetricType::Gauge,   [&](){ return m_evtSize; });
    exporter->add("DRP_evtLatency",   labels, Pds::MetricType::Gauge,   [&](){ return m_latency; });
}

std::string EbReceiver::openFiles(const Parameters& para, const RunInfo& runInfo, std::string hostname, unsigned nodeId)
{
    std::string retVal = std::string{};     // return empty string on success
    if (runInfo.runNumber) {
        m_chunkOffset = m_offset = 0;
        std::ostringstream ss;
        ss << runInfo.experimentName <<
              "-r" << std::setfill('0') << std::setw(4) << runInfo.runNumber <<
              "-s" << std::setw(3) << nodeId <<
              "-c000";
        std::string runName = ss.str();
        // data
        std::string exptDir = {para.outputDir + "/" + para.instrument + "/" + runInfo.experimentName};
        local_mkdir(exptDir.c_str());
        std::string dataDir = {exptDir + "/xtc"};
        local_mkdir(dataDir.c_str());
        std::string path = {"/" + para.instrument + "/" + runInfo.experimentName + "/xtc/" + runName + ".xtc2"};
        std::string absolute_path = {para.outputDir + path};
        // cpo suggests leaving this print statement in because
        // filesystems can hang in ways we can't timeout/detect
        // and this print statement may speed up debugging significantly.
        std::cout << "Opening file " << absolute_path << std::endl;
        logging::info("Opening file '%s'", absolute_path.c_str());
        if (m_fileWriter.open(absolute_path) == 0) {
            timespec tt; clock_gettime(CLOCK_REALTIME,&tt);
            json msg = createFileReportMsg(path, absolute_path, tt, tt, runInfo.runNumber, hostname);
            m_inprocSend.send(msg.dump());
        } else if (retVal.empty()) {
            retVal = {"Failed to open file '" + absolute_path + "'"};
        }
        // smalldata
        std::string smalldataDir = {para.outputDir + "/" + para.instrument + "/" + runInfo.experimentName + "/xtc/smalldata"};
        local_mkdir(smalldataDir.c_str());
        std::string smalldata_path = {"/" + para.instrument + "/" + runInfo.experimentName + "/xtc/smalldata/" + runName + ".smd.xtc2"};
        std::string smalldata_absolute_path = {para.outputDir + smalldata_path};
        logging::info("Opening file '%s'", smalldata_absolute_path.c_str());
        if (m_smdWriter.open(smalldata_absolute_path) == 0) {
            timespec tt; clock_gettime(CLOCK_REALTIME,&tt);
            json msg = createFileReportMsg(smalldata_path, smalldata_absolute_path, tt, tt, runInfo.runNumber, hostname);
            m_inprocSend.send(msg.dump());
        } else if (retVal.empty()) {
            retVal = {"Failed to open file '" + smalldata_absolute_path + "'"};
        }
        if (retVal.empty()) {
            m_writing = true;
            // cache file parameters for use by reopenFiles() (data file chunking)
            logging::debug("initializing m_fileParameters...");
            new((void *)&m_fileParameters) FileParameters(para, runInfo, hostname, nodeId);
        }
    }
    return retVal;
}

// return true if incremented chunkId
bool EbReceiver::advanceChunkId()
{
    bool status = false;
//  m_chunkPending_sem.take();
    if (!m_chunkPending) {
        logging::debug("%s: m_fileParameters.advanceChunkId()", __PRETTY_FUNCTION__);
        m_fileParameters.advanceChunkId();
        logging::debug("%s: m_chunkPending = true  chunkId = %u", __PRETTY_FUNCTION__, m_fileParameters.chunkId());
        m_chunkPending = true;
        status = true;
    }
//  m_chunkPending_sem.give();
    return status;
}

std::string EbReceiver::reopenFiles()
{
    logging::debug("entered %s", __PRETTY_FUNCTION__);
    if (m_writing == false) {
        logging::error("%s: m_writing is false", __PRETTY_FUNCTION__);
        return std::string("reopenFiles: m_writing is false");
    }
    std::string outputDir = m_fileParameters.outputDir();
    std::string instrument = m_fileParameters.instrument();
    std::string experimentName = m_fileParameters.experimentName();
    unsigned runNumber = m_fileParameters.runNumber();
    std::string hostname = m_fileParameters.hostname();

    std::string retVal = std::string{};     // return empty string on success
    m_chunkRequest = false;
    m_chunkOffset = m_offset;

    // close data file (for old chunk)
    logging::debug("%s: calling m_fileWriter.close()...", __PRETTY_FUNCTION__);
    m_fileWriter.close();

    // open data file (for new chunk)
    std::string runName = m_fileParameters.runName();
    std::string exptDir = {outputDir + "/" + instrument + "/" + experimentName};
    local_mkdir(exptDir.c_str());
    std::string dataDir = {exptDir + "/xtc"};
    local_mkdir(dataDir.c_str());
    std::string path = {"/" + instrument + "/" + experimentName + "/xtc/" + runName + ".xtc2"};
    std::string absolute_path = {outputDir + path};
    // cpo suggests leaving this print statement in because
    // filesystems can hang in ways we can't timeout/detect
    // and this print statement may speed up debugging significantly.
    std::cout << "Opening file " << absolute_path << std::endl;
    logging::info("%s: Opening file '%s'", __PRETTY_FUNCTION__, absolute_path.c_str());
    if (m_fileWriter.open(absolute_path) == 0) {
        timespec tt; clock_gettime(CLOCK_REALTIME,&tt);
        json msg = createFileReportMsg(path, absolute_path, tt, tt, runNumber, hostname);
        m_inprocSend.send(msg.dump());
        logging::debug("%s: m_chunkPending = false", __PRETTY_FUNCTION__);
        m_chunkPending = false;
    } else if (retVal.empty()) {
        retVal = {"Failed to open file '" + absolute_path + "'"};
    }

    return retVal;
}

std::string EbReceiver::closeFiles()
{
    logging::debug("%s: m_writing is %s", __PRETTY_FUNCTION__, m_writing ? "true" : "false");
    if (m_writing) {
        m_writing = false;
        logging::debug("calling m_smdWriter.close()...");
        m_smdWriter.close();
        logging::debug("calling m_fileWriter.close()...");
        m_fileWriter.close();
    }
    return std::string{};
}

uint64_t EbReceiver::chunkSize()
{
    return m_offset - m_chunkOffset;
}

bool EbReceiver::chunkPending()
{
    return m_chunkPending;
}

void EbReceiver::chunkRequestSet()
{
    m_chunkRequest = true;
}

void EbReceiver::chunkReset()
{
    // clean up the state left behind by a previous run
    m_chunkOffset = 0;
    m_chunkRequest = false;
//  m_chunkPending_sem = Pds::Semaphore::FULL;
    m_chunkPending = false;
}

bool EbReceiver::writing()
{
    return m_writing;
}

void EbReceiver::resetCounters()
{
    EbCtrbInBase::resetCounters();

    m_lastIndex = 0;
    m_damage = 0;
    m_dmgType->clear();
    m_latency = 0;
}

void EbReceiver::_writeDgram(XtcData::Dgram* dgram)
{
    size_t size = sizeof(*dgram) + dgram->xtc.sizeofPayload();
    m_fileWriter.writeEvent(dgram, size, dgram->time);

    // small data writing
    Smd smd;
    const void* bufEnd = m_smdWriter.buffer + sizeof(m_smdWriter.buffer);
    XtcData::NamesId namesId(dgram->xtc.src.value(), NamesIndex::OFFSETINFO);
    XtcData::Dgram* smdDgram = smd.generate(dgram, m_smdWriter.buffer, bufEnd, chunkSize(), size,
                                            m_smdWriter.namesLookup, namesId);
    m_smdWriter.writeEvent(smdDgram, sizeof(XtcData::Dgram) + smdDgram->xtc.sizeofPayload(), smdDgram->time);
    m_offset += size;
}

void EbReceiver::process(const Pds::Eb::ResultDgram& result, unsigned index)
{
    bool error = false;
    if (index != ((m_lastIndex + 1) & (m_pool.nbuffers() - 1))) {
        logging::critical("%sEbReceiver: jumping index %u  previous index %u  diff %d%s", RED_ON, index, m_lastIndex, index - m_lastIndex, RED_OFF);
        error = true;
    }

    Pds::EbDgram* dgram = (Pds::EbDgram*)m_pool.pebble[index];
    uint64_t pulseId = dgram->pulseId();
    XtcData::TransitionId::Value transitionId = dgram->service();
    if (transitionId != XtcData::TransitionId::L1Accept) {
        if (transitionId == 0) {
            logging::warning("transitionId == 0 in %s", __PRETTY_FUNCTION__);
        }
        dgram = m_pool.pgpEvents[index].transitionDgram;
        if (pulseId != dgram->pulseId()) {
            logging::error("pulseId mismatch: pebble %014lx, trDgram %014lx, xor %014lx, diff %ld",
                           pulseId, dgram->pulseId(), pulseId ^ dgram->pulseId(), pulseId - dgram->pulseId());
            error = true;
        }
        if (transitionId != dgram->service()) {
            logging::error("tid mismatch: pebble %u, trDgram %u", transitionId, dgram->service());
            error = true;
        }
    }
    if (pulseId == 0) {
        logging::critical("%spulseId %14lx, ts %u.%09u, tid %d, env %08x%s",
                          RED_ON, pulseId, dgram->time.seconds(), dgram->time.nanoseconds(), dgram->service(), dgram->env, RED_OFF);
        error = true;
    }
    if (pulseId != result.pulseId()) {
        logging::critical("pulseId mismatch: pebble %014lx, result %014lx, xor %014lx, diff %ld",
                          pulseId, result.pulseId(), pulseId ^ result.pulseId(), pulseId - result.pulseId());
        error = true;
    }
    if (transitionId != result.service()) {
        logging::error("tid mismatch: pebble %u, result %u", transitionId, result.service());
        error = true;
    }

    if (error) {
        logging::critical("pid     %014lx, tid     %s, env %08x", pulseId, XtcData::TransitionId::name(transitionId), dgram->env);
        logging::critical("lastPid %014lx, lastTid %s", m_lastPid, XtcData::TransitionId::name(m_lastTid));
        logging::critical("index %u  previous index %u", index, m_lastIndex);
        abort();
    }

    m_lastIndex = index;
    m_lastPid = pulseId;
    m_lastTid = transitionId;

    // Transfer Result damage to the datagram
    dgram->xtc.damage.increase(result.xtc.damage.value());
    uint16_t damage = dgram->xtc.damage.value();
    if (damage) {
        m_damage++;
        while (damage) {
            unsigned dmgType = __builtin_ffsl(damage) - 1;
            damage &= ~(1 << dmgType);
            m_dmgType->observe(dmgType);
        }
    }

    // pass everything except L1 accepts and slow updates to control level
    if ((transitionId != XtcData::TransitionId::L1Accept)) {
        if (transitionId != XtcData::TransitionId::SlowUpdate) {
            if (transitionId == XtcData::TransitionId::Configure) {
                // Cache Configure Dgram for writing out after files are opened
                XtcData::Dgram* configDgram = dgram;
                size_t size = sizeof(*configDgram) + configDgram->xtc.sizeofPayload();
                memcpy(m_configureBuffer.data(), configDgram, size);
            }
            if (transitionId == XtcData::TransitionId::BeginRun)
              m_offset = 0;// reset for monitoring (and not recording)
            // send pulseId to inproc so it gets forwarded to the collection
            json msg = createPulseIdMsg(pulseId);
            m_inprocSend.send(msg.dump());

            logging::info("EbReceiver saw %s @ %u.%09u (%014lx)",
                           XtcData::TransitionId::name(transitionId),
                          dgram->time.seconds(), dgram->time.nanoseconds(), pulseId);
        }
        else {
            logging::debug("EbReceiver saw %s @ %u.%09u (%014lx)",
                           XtcData::TransitionId::name(transitionId),
                           dgram->time.seconds(), dgram->time.nanoseconds(), pulseId);
        }
    }
    else { // L1Accept
        // On just the timing system DRP, save the trigger information
        if (m_det && (m_det->nodeId == m_tsId)) {
            const void* bufEnd = (char*)dgram + m_pool.bufferSize();
            m_det->event(*dgram, bufEnd, result);
        }
    }

    if (m_writing && !m_chunkRequest && (transitionId == XtcData::TransitionId::L1Accept)) {
        if (chunkSize() > DefaultChunkThresh) {
            // request chunking opportunity
            chunkRequestSet();
            logging::debug("%s: sending chunk request (chunkSize() > DefaultChunkThresh)", __PRETTY_FUNCTION__);
            json msg = createChunkRequestMsg();
            m_inprocSend.send(msg.dump());
        }
    }

    if (m_writing) {                    // Won't ever be true for Configure
        // write event to file if it passes event builder or if it's a transition
        if (result.persist() || result.prescale()) {
            _writeDgram(dgram);
        }
        else if (transitionId != XtcData::TransitionId::L1Accept) {
            if (transitionId == XtcData::TransitionId::BeginRun) {
                m_offset = 0; // reset offset when writing out a new file
                _writeDgram(reinterpret_cast<XtcData::Dgram*>(m_configureBuffer.data()));
            }
            _writeDgram(dgram);
            if ((transitionId == XtcData::TransitionId::Enable) && m_chunkRequest) {
                logging::debug("%s calling reopenFiles()", __PRETTY_FUNCTION__);
                reopenFiles();
            } else if (transitionId == XtcData::TransitionId::EndRun) {
                logging::debug("%s calling closeFiles()", __PRETTY_FUNCTION__);
                closeFiles();
            }
        }
    }

    m_evtSize = sizeof(*dgram) + dgram->xtc.sizeofPayload();

    // Measure latency before sending dgram for monitoring
    auto now = std::chrono::system_clock::now();
    auto dgt = std::chrono::seconds{dgram->time.seconds() + POSIX_TIME_AT_EPICS_EPOCH}
             + std::chrono::nanoseconds{dgram->time.nanoseconds()};
    std::chrono::system_clock::time_point tp{std::chrono::duration_cast<std::chrono::system_clock::duration>(dgt)};
    m_latency = std::chrono::duration_cast<ms_t>(now - tp).count();

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
        if (transitionId == XtcData::TransitionId::Enable) {
            _t0 = tp;
            _enabled = true;
        }
        if (_enabled && (tp - _t0 > std::chrono::seconds(1 * 60))) { // Delay a bit before sleeping
            printf("*** EbReceiver: Inducing deadtime by sleeping for 30s at PID %014lx, ts %9u.%09u\n",
                   pulseId, dgram->time.seconds(), dgram->time.nanoseconds());
            std::this_thread::sleep_for(std::chrono::seconds(30));
            _t0 = tp;
            _enabled = false;
            printf("*** EbReceiver: Continuing after sleeping for 30s\n");
        }
    }
#endif

    // Free the transition datagram buffer
    if (!dgram->isEvent()) {
        m_pool.freeTr(dgram);
    }

    // Return buffers and reset event.  Careful with order here!
    // index could be reused as soon as dmaRetIndexes() completes
    PGPEvent* event = &m_pool.pgpEvents[index];
    for (int i=0; i<PGP_MAX_LANES; i++) {
        if (event->mask &  (1 << i)) {
            event->mask ^= (1 << i);    // Zero out mask before dmaRetIndexes()
            m_indices[m_count] = event->buffers[i].index;
            m_count++;
            if (m_count == m_size) {
                dmaRetIndexes(m_pool.fd(), m_count, m_indices);
                // std::cout<<"return dma buffers to driver\n";
                m_pool.release(m_count);
                m_count = 0;
            }
        }
    }
}


DrpBase::DrpBase(Parameters& para, ZmqContext& context) :
    pool(para), m_para(para), m_inprocSend(&context, ZMQ_PAIR)
{
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    m_hostname = std::string(hostname);

    m_exposer = Pds::createExposer(para.prometheusDir, m_hostname);
    m_exporter = std::make_shared<Pds::MetricExporter>();

    if (m_exposer) {
        m_exposer->RegisterCollectable(m_exporter);
    }

    std::map<std::string, std::string> labels{{"instrument", para.instrument},
                                              {"partition", std::to_string(para.partition)},
                                              {"detname", para.detName},
                                              {"detseg", std::to_string(para.detSegment)},
                                              {"alias", para.alias}};
    m_exporter->add("drp_port_rcv_rate", labels, Pds::MetricType::Rate,
                    [](){return 4*readInfinibandCounter("port_rcv_data");});

    m_exporter->add("drp_port_xmit_rate", labels, Pds::MetricType::Rate,
                    [](){return 4*readInfinibandCounter("port_xmit_data");});

    m_exporter->add("drp_dma_in_use", labels, Pds::MetricType::Gauge,
                    [&](){return pool.inUse();});
    m_exporter->constant("drp_dma_in_use_max", labels, pool.nbuffers());

    m_tPrms.instrument = para.instrument;
    m_tPrms.partition  = para.partition;
    m_tPrms.alias      = para.alias;
    m_tPrms.detName    = para.detName;
    m_tPrms.detSegment = para.detSegment;
    m_tPrms.maxEntries = m_para.kwargs["batching"] == "yes" ? Pds::Eb::MAX_ENTRIES : 1; // Default to "no"
    m_tPrms.core[0]    = -1;
    m_tPrms.core[1]    = -1;
    m_tPrms.verbose    = para.verbose;
    m_tPrms.kwargs     = para.kwargs;
    m_tebContributor = std::make_unique<Pds::Eb::TebContributor>(m_tPrms, pool.nbuffers(), m_exporter);

    m_mPrms.instrument = para.instrument;
    m_mPrms.partition  = para.partition;
    m_mPrms.alias      = para.alias;
    m_tPrms.detName    = para.detName;
    m_tPrms.detSegment = para.detSegment;
    m_mPrms.maxEvSize  = pool.bufferSize();
    m_mPrms.maxTrSize  = para.maxTrSize;
    m_mPrms.verbose    = para.verbose;
    m_mPrms.kwargs     = para.kwargs;
    m_mebContributor = std::make_unique<Pds::Eb::MebContributor>(m_mPrms, m_exporter);

    m_ebRecv = std::make_unique<EbReceiver>(m_para, m_tPrms, pool, m_inprocSend, *m_mebContributor, m_exporter);

    m_inprocSend.connect("inproc://drp");

    if (para.outputDir.empty()) {
        logging::info("output dir: n/a");
    } else {
        // Induce the automounter to mount in case user enables recording
        struct stat statBuf;
        std::string statPth = para.outputDir + "/" + para.instrument;
        if (::stat(statPth.c_str(), &statBuf) < 0) {
            logging::error("%s: stat(%s) error: %m", __PRETTY_FUNCTION__, statPth.c_str());
        }
        logging::info("output dir: %s", statPth.c_str());
    }
}

void DrpBase::shutdown()
{
    m_tebContributor->shutdown();
    m_mebContributor->shutdown();
    m_ebRecv->shutdown();
}

json DrpBase::connectionInfo(const std::string& ip)
{
    m_tPrms.ifAddr = ip;
    m_tPrms.port.clear();               // Use an ephemeral port

    int rc = m_ebRecv->startConnection(m_tPrms.port);
    if (rc)  throw "Error starting connection";

    json info = {{"drp_port", m_tPrms.port},
                 {"num_buffers", pool.nbuffers()},
                 {"max_ev_size", pool.bufferSize()},
                 {"max_tr_size", m_para.maxTrSize}};
    return info;
}

std::string DrpBase::connect(const json& msg, size_t id)
{
    // Save a copy of the json so we can use it to connect to the config database on configure
    m_connectMsg = msg;
    m_collectionId = id;

    int rc = parseConnectionParams(msg["body"], id);
    if (rc) {
        return std::string{"Connection parameters error - see log"};
    }

    // Make a guess at the size of the Input entries
    size_t inpSizeGuess = sizeof(Pds::EbDgram) + 2 * sizeof(uint32_t);

    rc = m_tebContributor->connect(inpSizeGuess);
    if (rc) {
        return std::string{"TebContributor connect failed"};
    }
    if (m_mPrms.addrs.size() != 0) {
        void* poolBase = (void*)pool.pebble[0];
        size_t poolSize = pool.pebble.size();
        rc = m_mebContributor->connect(m_mPrms, poolBase, poolSize);
        if (rc) {
            return std::string{"MebContributor connect failed"};
        }
    }

    // Make a guess at the size of the Result entries
    size_t resSizeGuess = sizeof(Pds::EbDgram) + 2  * sizeof(uint32_t);

    rc = m_ebRecv->connect(resSizeGuess, m_numTebBuffers);
    if (rc) {
        return std::string{"EbReceiver connect failed"};
    }

    // On the timing system DRP, EbReceiver needs to know its node ID
    if (m_para.detType == "ts")  m_ebRecv->tsId(m_nodeId);

    return std::string{};
}

std::string DrpBase::configure(const json& msg)
{
    if (setupTriggerPrimitives(msg["body"])) {
        return std::string("Failed to set up TriggerPrimitive(s)");
    }

    int rc = m_tebContributor->configure();
    if (rc) {
        return std::string{"TebContributor configure failed"};
    }

    if (m_mPrms.addrs.size() != 0) {
        void* poolBase = (void*)pool.pebble[0];
        size_t poolSize = pool.pebble.size();
        rc = m_mebContributor->configure(poolBase, poolSize);
        if (rc) {
            return std::string{"MebContributor configure failed"};
        }
    }

    rc = m_ebRecv->configure(m_numTebBuffers);
    if (rc) {
        return std::string{"EbReceiver configure failed"};
    }

    printParams();

    // start eb receiver thread
    m_tebContributor->startup(*m_ebRecv);

    // Same time as the TEBs and MEBs
    m_tebContributor->resetCounters();
    m_mebContributor->resetCounters();
    m_ebRecv->resetCounters();
    return std::string{};
}

std::string DrpBase::beginrun(const json& phase1Info, RunInfo& runInfo)
{
    std::string msg;
    std::string experiment_name;
    unsigned int run_number = 0;
    if (phase1Info.find("run_info") != phase1Info.end()) {
        if (phase1Info["run_info"].find("experiment_name") != phase1Info["run_info"].end()) {
            experiment_name = phase1Info["run_info"]["experiment_name"];
        }
        if (phase1Info["run_info"].find("run_number") != phase1Info["run_info"].end()) {
            run_number = phase1Info["run_info"]["run_number"];
        }
    }
    runInfo.experimentName = experiment_name;
    runInfo.runNumber = run_number;

    logging::debug("expt=\"%s\" runnum=%u",
                   experiment_name.c_str(), run_number);

    // if run_number is nonzero then we are recording
    if (run_number) {
        logging::debug("Recording to directory '%s'", m_para.outputDir.c_str());
        if (m_para.outputDir.empty()) {
            msg = "Cannot record due to missing output directory";
        } else {
            msg = m_ebRecv->openFiles(m_para, runInfo, m_hostname, m_nodeId);
        }
    }

    // Same time as the TEBs and MEBs
    m_tebContributor->resetCounters();
    m_mebContributor->resetCounters();
    m_ebRecv->resetCounters();
    m_ebRecv->chunkReset();
    return msg;
}

void DrpBase::runInfoSupport(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup)
{
    logging::debug("entered %s", __PRETTY_FUNCTION__);
    XtcData::Alg runInfoAlg("runinfo", 0, 0, 1);
    XtcData::NamesId runInfoNamesId(xtc.src.value(), NamesIndex::RUNINFO);
    XtcData::Names& runInfoNames = *new(xtc, bufEnd) XtcData::Names(bufEnd,
                                                                    "runinfo", runInfoAlg,
                                                                    "runinfo", "", runInfoNamesId);
    RunInfoDef myRunInfoDef;
    runInfoNames.add(xtc, bufEnd, myRunInfoDef);
    namesLookup[runInfoNamesId] = XtcData::NameIndex(runInfoNames);
}

void DrpBase::runInfoData(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, const RunInfo& runInfo)
{
    XtcData::NamesId runInfoNamesId(xtc.src.value(), NamesIndex::RUNINFO);
    XtcData::CreateData runinfo(xtc, bufEnd, namesLookup, runInfoNamesId);
    runinfo.set_string(RunInfoDef::EXPT, runInfo.experimentName.c_str());
    runinfo.set_value(RunInfoDef::RUNNUM, runInfo.runNumber);
}

void DrpBase::chunkInfoSupport(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup)
{
    logging::debug("entered %s", __PRETTY_FUNCTION__);
    XtcData::Alg chunkInfoAlg("chunkinfo", 0, 0, 1);
    XtcData::NamesId chunkInfoNamesId(xtc.src.value(), NamesIndex::CHUNKINFO);
    XtcData::Names& chunkInfoNames = *new(xtc, bufEnd) XtcData::Names(bufEnd,
                                                                      "chunkinfo", chunkInfoAlg,
                                                                      "chunkinfo", "", chunkInfoNamesId);
    ChunkInfoDef myChunkInfoDef;
    chunkInfoNames.add(xtc, bufEnd, myChunkInfoDef);
    namesLookup[chunkInfoNamesId] = XtcData::NameIndex(chunkInfoNames);
}

void DrpBase::chunkInfoData(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, const ChunkInfo& chunkInfo)
{
    XtcData::NamesId chunkInfoNamesId(xtc.src.value(), NamesIndex::CHUNKINFO);
    XtcData::CreateData chunkinfo(xtc, bufEnd, namesLookup, chunkInfoNamesId);
    chunkinfo.set_string(ChunkInfoDef::FILENAME, chunkInfo.filename.c_str());
    chunkinfo.set_value(ChunkInfoDef::CHUNKID, chunkInfo.chunkId);
}

std::string DrpBase::endrun(const json& phase1Info)
{
    return std::string{};
}

std::string DrpBase::enable(const json& phase1Info, bool& chunkRequest, ChunkInfo& chunkInfo)
{
    std::string retval = std::string{};

    logging::debug("%s: writing() is %s", __PRETTY_FUNCTION__, m_ebRecv->writing() ? "true" : "false");
    chunkRequest = false;
    if (m_ebRecv->writing()) {
        logging::debug("%s: chunkSize() = %lu", __PRETTY_FUNCTION__, m_ebRecv->chunkSize());
        if (m_ebRecv->chunkSize() > EbReceiver::DefaultChunkThresh / 2ull) {
            if (m_ebRecv->advanceChunkId()) {
                logging::debug("%s: advanceChunkId() returned true", __PRETTY_FUNCTION__);
                // request new chunk after this Enable dgram is written
                chunkRequest = true;
                m_ebRecv->chunkRequestSet();
                chunkInfo.filename = {m_ebRecv->fileParameters()->runName() + ".xtc2"};
                chunkInfo.chunkId = m_ebRecv->fileParameters()->chunkId();
                logging::debug("%s: chunkInfo.filename = %s", __PRETTY_FUNCTION__, chunkInfo.filename.c_str());
                logging::debug("%s: chunkInfo.chunkId  = %u", __PRETTY_FUNCTION__, chunkInfo.chunkId);
            }
        }
    }
    return retval;
}

void DrpBase::unconfigure()
{
    m_tebContributor->unconfigure();
    if (m_mPrms.addrs.size() != 0) {
        m_mebContributor->unconfigure();
    }
    m_ebRecv->unconfigure();
}

void DrpBase::disconnect()
{
    m_tebContributor->disconnect();
    if (m_mPrms.addrs.size() != 0) {
        m_mebContributor->disconnect();
    }
    m_ebRecv->disconnect();
}

int DrpBase::setupTriggerPrimitives(const json& body)
{
    using namespace rapidjson;

    Document top;
    const std::string configAlias   = body["config_alias"];
    const std::string triggerConfig = body["trigger_config"];

    // In the following, _0 is added in prints to show the default segment number
    logging::info("Fetching trigger info from ConfigDb/%s/%s_0",
                  configAlias.c_str(), triggerConfig.c_str());

    if (Pds::Trg::fetchDocument(m_connectMsg.dump(), configAlias, triggerConfig, top))
    {
        logging::error("%s:\n  Document '%s_0' not found in ConfigDb",
                       __PRETTY_FUNCTION__, triggerConfig.c_str());
        return -1;
    }

    bool buildAll = top.HasMember("buildAll") && top["buildAll"].GetInt()==1;
    if (!buildAll && !top.HasMember(m_para.detName.c_str())) {
        logging::warning("This DRP is not contributing trigger input data: "
                         "'%s' not found in ConfigDb for %s",
                         m_para.detName.c_str(), triggerConfig.c_str());
        m_tPrms.contractor = 0;    // This DRP won't provide trigger input data
        m_triggerPrimitive = nullptr;
        m_tPrms.maxInputSize = sizeof(Pds::EbDgram); // Revisit: 0 is bad
        return 0;
    }
    m_tPrms.contractor = m_tPrms.readoutGroup;

    std::string symbol("create_producer");
    if (!buildAll)  symbol +=  "_" + m_para.detName;
    m_triggerPrimitive = m_trigPrimFactory.create(top, triggerConfig, symbol);
    if (!m_triggerPrimitive) {
        logging::error("%s:\n  Failed to create TriggerPrimitive",
                       __PRETTY_FUNCTION__);
        return -1;
    }
    m_tPrms.maxInputSize = sizeof(Pds::EbDgram) + m_triggerPrimitive->size();

    if (m_triggerPrimitive->configure(top, m_connectMsg, m_collectionId)) {
        logging::error("%s:\n  Failed to configure TriggerPrimitive",
                       __PRETTY_FUNCTION__);
        return -1;
    }

    return 0;
}

int DrpBase::parseConnectionParams(const json& body, size_t id)
{
    int rc = 0;
    std::string stringId = std::to_string(id);
    logging::debug("id %zu", id);
    m_tPrms.id = body["drp"][stringId]["drp_id"];
    m_mPrms.id = m_tPrms.id;
    m_nodeId = body["drp"][stringId]["drp_id"];

    uint64_t builders = 0;
    m_tPrms.addrs.clear();
    m_tPrms.ports.clear();
    for (auto it : body["teb"].items()) {
        unsigned tebId = it.value()["teb_id"];
        std::string address = it.value()["connect_info"]["nic_ip"];
        std::string port = it.value()["connect_info"]["teb_port"];
        logging::debug("TEB: %u  %s:%s", tebId, address.c_str(), port.c_str());
        builders |= 1ul << tebId;
        m_tPrms.addrs.push_back(address);
        m_tPrms.ports.push_back(port);
    }
    m_tPrms.builders = builders;

    // Store our readout group as a mask to make comparison with Dgram::readoutGroups() cheaper
    m_tPrms.readoutGroup = 1 << unsigned(body["drp"][stringId]["det_info"]["readout"]);
    m_tPrms.contractor = 0;             // Overridden during Configure

    m_para.rogMask = 0x00ff0000;
    m_numTebBuffers = 0;                // Only need the largest value
    unsigned maxBuffers = 0;
    bool bufErr = false;
    for (auto it : body["drp"].items()) {
        unsigned drpId = it.value()["drp_id"];

        // Build readout group mask for ignoring other partitions' RoGs
        unsigned rog(it.value()["det_info"]["readout"]);
        if (rog < Pds::Eb::NUM_READOUT_GROUPS) {
            m_para.rogMask |= 1 << rog;
        }
        else {
            logging::warning("Ignoring Readout Group %d > max (%d)", rog, Pds::Eb::NUM_READOUT_GROUPS - 1);
        }

        // The Common RoG governs the index into the Results region.
        // Its range must be >= that of any subsidary RoG.
        auto numBuffers = it.value()["connect_info"]["num_buffers"];
        if (numBuffers > m_numTebBuffers) {
            if (rog == m_tPrms.partition) {
                m_numTebBuffers = numBuffers;
            }
            else if (numBuffers > maxBuffers) {
                maxBuffers = numBuffers;
                bufErr = drpId == m_nodeId; // Report only for the current DRP
            }
        }
    }

    // Disallow non-common RoG DRPs from having more buffers than the common one
    // because the buffer index based on the common RoG DRPs won't be able to
    // reach the higher buffer numbers.  Can't use an index based on the largest
    // non-common RoG DRP because it would overrun the common RoG DRPs' region.
    if ((maxBuffers > m_numTebBuffers) && bufErr) {
        logging::error("DMA buffer count (%u) must be <= %u\n",
                       pool.nbuffers(), m_numTebBuffers);
        rc = 1;
    }

    m_mPrms.addrs.clear();
    m_mPrms.ports.clear();
    m_mPrms.maxEvents = 0;
    if (body.find("meb") != body.end()) {
        for (auto it : body["meb"].items()) {
            unsigned mebId = it.value()["meb_id"];
            std::string address = it.value()["connect_info"]["nic_ip"];
            std::string port = it.value()["connect_info"]["meb_port"];
            logging::debug("MEB: %u  %s:%s", mebId, address.c_str(), port.c_str());
            m_mPrms.addrs.push_back(address);
            m_mPrms.ports.push_back(port);
            unsigned count = it.value()["connect_info"]["max_ev_count"];
            if (!m_mPrms.maxEvents)  m_mPrms.maxEvents = count;
            if (count != m_mPrms.maxEvents) {
                logging::error("maxEvents (%u) must be the same for all MEBs, got %u from ID %u",
                               m_mPrms.maxEvents, count, mebId);
                rc = 1;
            }
        }
    }

    return rc;
}

void DrpBase::printParams() const
{
    using namespace Pds::Eb;

    printf("\nParameters of Contributor ID %d (%s:%s):\n",         m_tPrms.id,
                                                                   m_tPrms.ifAddr.c_str(), m_tPrms.port.c_str());
    printf("  Thread core numbers:          %d, %d\n",             m_tPrms.core[0], m_tPrms.core[1]);
    printf("  Partition:                    %u\n",                 m_tPrms.partition);
    printf("  Readout group receipient:     0x%02x\n",             m_tPrms.readoutGroup);
    printf("  Readout group contractor:     0x%02x\n",             m_tPrms.contractor);
    printf("  Bit list of TEBs:             0x%016lx, cnt: %zu\n", m_tPrms.builders,
                                                                   std::bitset<64>(m_tPrms.builders).count());
    printf("  Number of MEBs:               %zu\n",                m_mPrms.addrs.size());
    printf("  Batching state:               %s\n",                 m_tPrms.maxEntries > 1 ? "Enabled" : "Disabled");
    printf("  Batch duration:               0x%014x = %u uS\n",    m_tPrms.maxEntries, m_tPrms.maxEntries); // Revisit: * 14/13
    printf("  Batch pool depth:             0x%08x = %u\n",        pool.nbuffers() / m_tPrms.maxEntries, pool.nbuffers() / m_tPrms.maxEntries);
    printf("  Max # of entries / batch:     0x%08x = %u\n",        m_tPrms.maxEntries, m_tPrms.maxEntries);
    printf("  # of TEB contrib.   buffers:  0x%08x = %u\n",        pool.nbuffers(), pool.nbuffers());
    printf("  # of TEB transition buffers:  0x%08x = %u\n",        TEB_TR_BUFFERS, TEB_TR_BUFFERS);
    printf("  Max  TEB contribution  size:  0x%08zx = %zu\n",      m_tPrms.maxInputSize, m_tPrms.maxInputSize);
    printf("  Max  MEB L1Accept      size:  0x%08zx = %zu\n",      m_mPrms.maxEvSize, m_mPrms.maxEvSize);
    printf("  Max  MEB transition    size:  0x%08zx = %zu\n",      m_mPrms.maxTrSize, m_mPrms.maxTrSize);
    printf("  # of MEB contrib.   buffers:  0x%08x = %u\n",        m_mPrms.maxEvents, m_mPrms.maxEvents);
    printf("  # of MEB transition buffers:  0x%08x = %u\n",        MEB_TR_BUFFERS, MEB_TR_BUFFERS);
    printf("\n");
}

}

#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>

static void local_mkdir (const char * path)
{
    struct stat64 buf;

    if (path && (stat64(path, &buf) != 0)) {
        if (mkdir(path, 0777)) {
            logging::critical("mkdir %s: %m", path);
        }
    }
}

static json createFileReportMsg(std::string path, std::string absolute_path,
                                timespec create_time, timespec modify_time,
                                unsigned run_num, std::string hostname)
{
    char buf[100];
    json msg, body;

    msg["key"] = "fileReport";
    body["path"] = path;
    body["absolute_path"] = absolute_path;
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&create_time.tv_sec));
    body["create_timestamp"] = buf;
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&modify_time.tv_sec));
    body["modify_timestamp"] = buf;
    body["hostname"] = hostname;
    body["gen"] = 2;                // 2 == LCLS-II
    body["run_num"] = run_num;
    msg["body"] = body;
    return msg;
}

static json createPulseIdMsg(uint64_t pulseId)
{
    json msg, body;
    msg["key"] = "pulseId";
    body["pulseId"] = pulseId;
    msg["body"] = body;
    return msg;
}

static json createChunkRequestMsg()
{
    json msg, body;
    msg["key"] = "chunkRequest";
    msg["body"] = body;
    return msg;
}

