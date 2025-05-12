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
#include "psdaq/aes-stream-drivers/DmaDriver.h"
#include "DrpBase.hh"
#include "RunInfoDef.hh"
#include "psalg/utils/SysLog.hh"
#include "xtcdata/xtc/Smd.hh"
#include "psdaq/aes-stream-drivers/DataDriver.h"
#include "psdaq/aes-stream-drivers/DmaDest.h"
#include "psdaq/epicstools/PVBase.hh"

#include "rapidjson/document.h"

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using namespace XtcData;
using json = nlohmann::json;
using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;
using us_t = std::chrono::microseconds;

static void local_mkdir (const char * path);
static json createFileReportMsg(std::string path, std::string absolute_path,
                                timespec create_time, timespec modify_time,
                                unsigned run_num, std::string hostname);
static json createPulseIdMsg(uint64_t pulseId);
static json createChunkRequestMsg();

namespace Drp {

static std::string _getHostName()
{
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    return std::string(hostname);
}

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

MemPool::MemPool(const Parameters& para) :
    m_transitionBuffers(nextPowerOf2(Pds::Eb::TEB_TR_BUFFERS)), // See eb.hh
    m_dmaAllocs(0),
    m_dmaFrees(0),
    m_allocs(0),
    m_frees(0)
{
}

void MemPool::_initialize(const Parameters& para)
{
    logging::info("dmaCount %u,  dmaSize %u", m_dmaCount, m_dmaSize);

    // make sure there are more buffers in the pebble than in the pgp driver
    // otherwise the pebble buffers will be overwritten by the pgp event builder
    m_nDmaBuffers = nextPowerOf2(m_dmaCount);
    if (m_nDmaBuffers > 0xffffff) {     // Mask for evtCounter
        logging::critical("nDmaBuffers (%u) can't exceed evtCounter range (%u)",
                          m_nDmaBuffers, 0xffffff);
        abort();
    }

    // make the size of the pebble buffer that will contain the datagram equal
    // to the dmaSize times the number of lanes
    // Also include space in the pebble for a pool of transition buffers of
    // worst case size so that they will be part of the memory region that can
    // be RDMAed from to the MEB
    size_t maxL1ASize = para.kwargs.find("pebbleBufSize") == para.kwargs.end() // Allow overriding the Pebble size
                      ? __builtin_popcount(para.laneMask) * m_dmaSize
                      : std::stoul(const_cast<Parameters&>(para).kwargs["pebbleBufSize"]);
    m_nbuffers        = para.kwargs.find("pebbleBufCount") == para.kwargs.end() // Allow overriding the Pebble count
                      ? m_nDmaBuffers
                      : std::stoul(const_cast<Parameters&>(para).kwargs["pebbleBufCount"]);
    if (m_nbuffers < m_nDmaBuffers) {
      logging::critical("nPebbleBuffers (%u) must be > nDmaBuffers (%u)",
                        m_nbuffers, m_nDmaBuffers);
      abort();
    }
    auto nTrBuffers = m_transitionBuffers.size();
    pebble.create(m_nbuffers, maxL1ASize, nTrBuffers, para.maxTrSize);
    logging::info("nL1Buffers %u,  pebble buffer size %zu", m_nbuffers, pebble.bufferSize());
    logging::info("nTrBuffers %u,  transition buffer size %zu", nTrBuffers, para.maxTrSize);

    pgpEvents.resize(m_nDmaBuffers);
    transitionDgrams.resize(m_nbuffers);

    // Put the transition buffer pool at the end of the pebble buffers
    uint8_t* buffer = pebble[m_nbuffers];
    for (size_t i = 0; i < m_transitionBuffers.size(); i++) {
        m_transitionBuffers.push(&buffer[i * para.maxTrSize]);
    }
}

unsigned MemPool::allocateDma()
{
    // Actually, the DMA buffer is allocated by the f/w and we only account for it here

    auto allocs = m_dmaAllocs.fetch_add(1, std::memory_order_acq_rel);

    return allocs;
}

unsigned MemPool::allocate()
{
    auto allocs = m_allocs.fetch_add(1, std::memory_order_acq_rel);
    asm volatile("mfence" ::: "memory");
    auto frees  = m_frees.load(std::memory_order_acquire);

    // Block when there are no available pebble buffers
    if (allocs - frees == m_nbuffers - 1) {
        std::unique_lock<std::mutex> lock(m_lock);
        m_condition.wait(lock, [this] {
            return (m_allocs.load(std::memory_order_acquire) -
                    m_frees.load(std::memory_order_acquire)) != m_nbuffers;
        });
    }

    return allocs & (m_nbuffers - 1);
}

void MemPool::freeDma(unsigned count, uint32_t* indices)
{
    _freeDma(count, indices);

    m_dmaFrees.fetch_add(count, std::memory_order_acq_rel);
}

void MemPool::freePebble()
{
    auto frees  = m_frees.fetch_add(1, std::memory_order_acq_rel);
    asm volatile("mfence" ::: "memory");
    auto allocs = m_allocs.load(std::memory_order_acquire);

    // Release when all pebble buffers were in use but now one is free
    if (allocs - frees == m_nbuffers) {
        std::lock_guard<std::mutex> lock(m_lock);
        m_condition.notify_one();
    }
}

void MemPool::flushPebble()
{
    while (inUse()) {
        freePebble();
    }
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

void MemPool::resetCounters()
{
    if (dmaInUse() == 0) {
        m_dmaAllocs.store(0);
        m_dmaFrees .store(0);
    } else {
        // This means DMA buffers were lost and we don't have their indices to free them
        // (return them to the PGP FPGA).  Can run with fewer buffers but crash instead?
        logging::warning("DMA counters cannot be reset while buffers are still in use: "
                         "Allocs %lu, Frees %lu, inUse %ld",
                         m_dmaAllocs.load(), m_dmaFrees.load(), dmaInUse());
    }

    if (inUse()) {
        logging::warning("Pebble counters reset although buffers are still in use: "
                         "Allocs %lu, Frees %lu, inUse %ld",
                         m_allocs.load(), m_frees.load(), inUse());
    }
    m_allocs.store(0);
    m_frees .store(0);
}

void MemPool::shutdown()
{
    m_transitionBuffers.shutdown();
}

MemPoolCpu::MemPoolCpu(const Parameters& para) :
    MemPool(para),
    m_setMaskBytesDone(false)
{
    m_fd = open(para.device.c_str(), O_RDWR);
    if (m_fd < 0) {
        logging::critical("Error opening %s: %m", para.device.c_str());
        abort();
    }
    logging::info("PGP device '%s' opened", para.device.c_str());

    dmaBuffers = dmaMapDma(m_fd, &m_dmaCount, &m_dmaSize);
    if (dmaBuffers == NULL ) {
        logging::critical("Failed to map dma buffers: %m");
        abort();
    }

    // Continue with initialization of the base class
    _initialize(para);
}

MemPoolCpu::~MemPoolCpu()
{
   logging::debug("%s: Closing PGP device file descriptor", __PRETTY_FUNCTION__);
    close(m_fd);
}

void MemPoolCpu::_freeDma(unsigned count, uint32_t* indices)
{
    dmaRetIndexes(m_fd, count, indices);
}

int MemPoolCpu::setMaskBytes(uint8_t laneMask, unsigned virtChan)
{
    int retval = 0;
    if (m_setMaskBytesDone) {
        logging::debug("%s: earlier setting in effect", __PRETTY_FUNCTION__);
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

PgpReader::PgpReader(const Parameters& para, MemPool& pool, unsigned maxRetCnt, unsigned dmaFreeCnt) :
    m_para        (para),
    m_pool        (pool),
    m_tmo         {0},
    dmaRet        (maxRetCnt),
    dmaIndex      (maxRetCnt),
    dest          (maxRetCnt),
    dmaFlags      (maxRetCnt),
    dmaErrors     (maxRetCnt),
    m_lastComplete(0),
    m_lastTid     (TransitionId::Reset),
    m_dmaIndices  (dmaFreeCnt),
    m_count       (0),
    m_dmaBytes    (0),
    m_dmaSize     (0),
    m_latPid      (0),
    m_latency     (0),
    m_nDmaErrors  (0),
    m_nNoComRoG   (0),
    m_nMissingRoGs(0),
    m_nTmgHdrError(0),
    m_nPgpJumps   (0),
    m_nNoTrDgrams (0)
{
    // Ensure there are more DMA buffers than the size of the batch used to free them
    if (pool.dmaCount() < m_dmaIndices.size()) {
        logging::critical("nDmaIndices (%zu) must be >= dmaCount (%u)",
                          m_dmaIndices.size(), pool.dmaCount());
        abort();
    }

    m_pfd.fd = pool.fd();
    m_pfd.events = POLLIN;

    pool.resetCounters();
}

PgpReader::~PgpReader()
{
    flush();
}

int32_t PgpReader::read()
{
    if (m_tmo) {                        // If in interrupt mode...
        // Wait for DMAed data to become available
        if (poll(&m_pfd, 1, m_tmo) < 0) {
            logging::error("%s: poll() error: %m", __PRETTY_FUNCTION__);
            return 0;
        }
        if (m_pfd.revents == 0) {
            return 0;                   // Timed out
        }
        if (m_pfd.revents == POLLIN) {  // When DMAed data is available...
            m_tmo = 0;                  // switch to polling mode
            m_t0  = Pds::fast_monotonic_clock::now();
        }
    }

    int rc = dmaReadBulkIndex(m_pool.fd(), dmaRet.size(), dmaRet.data(), dmaIndex.data(), dmaFlags.data(), dmaErrors.data(), dest.data());

    if ((rc == 0) && (m_tmo == 0)) {    // If no DMAed data and in polling mode...
        auto t1 { Pds::fast_monotonic_clock::now() };

        if (t1 - m_t0 >= ms_t{1}) {
            m_tmo = 10;                 // switch to interrupt mode after 1 ms
        }
    }

    return rc;
}

void PgpReader::flush()
{
    // Return DMA buffers queued for freeing
    if (m_count)  m_pool.freeDma(m_count, m_dmaIndices.data());
    m_count = 0;

    // Also return DMA buffers queued for reading, without adjusting counters
    int32_t ret;
    while ( (ret = read()) ) {
        dmaRetIndexes(m_pool.fd(), ret, dmaIndex.data());
    }

    // Free any in-use pebble buffers
    m_pool.flushPebble();
}

const Pds::TimingHeader* PgpReader::handle(Detector* det, unsigned current)
{
    uint32_t size = dmaRet[current];
    uint32_t index = dmaIndex[current];
    uint32_t lane = (dest[current] >> 8) & 7;
    m_dmaSize = size;
    m_dmaBytes += size;
    // dmaReadBulkIndex() returns a maximum size of m_pool.dmaSize(), never larger.
    // If the DMA overflowed the buffer, the excess is returned in a 2nd DMA buffer,
    // which thus won't have the expected header.  Take the exact match as an overflow indicator.
    if (size == m_pool.dmaSize()) {
        logging::critical("DMA overflowed buffer: %d vs %d", size, m_pool.dmaSize());
        abort();
    }

    const Pds::TimingHeader* timingHeader = det->getTimingHeader(index);
    if (timingHeader->pulseId() - m_latPid > 1300000/14) { // 10 Hz
        m_latency = std::chrono::duration_cast<us_t>(age(timingHeader->time)).count();
        m_latPid = timingHeader->pulseId();
    }
    uint32_t evtCounter = timingHeader->evtCounter & 0xffffff;
    uint32_t pgpIndex = evtCounter & (m_pool.nDmaBuffers() - 1);
    PGPEvent* event = &m_pool.pgpEvents[pgpIndex];
    DmaBuffer* buffer = &event->buffers[lane];
    buffer->size = size;
    buffer->index = index;
    event->mask |= (1 << lane);

    m_pool.allocateDma(); // DMA buffer was allocated when f/w incremented evtCounter

    uint32_t flag = dmaFlags[current];
    uint32_t err  = dmaErrors[current];
    if (err) {
        logging::error("DMA with error 0x%x  flag 0x%x",err,flag);
        //  How do I return this buffer?
        ++m_lastComplete;
        handleBrokenEvent(*event);
        freeDma(event);                 // Leaves event mask = 0
        ++m_nDmaErrors;
        return nullptr;
    }

    if (timingHeader->error()) {
        logging::error("Timing header error bit is set");
        ++m_nTmgHdrError;
    }
    TransitionId::Value transitionId = timingHeader->service();
    //{
    //  uint32_t lane = dest[current] >> 8;
    //  uint32_t vc   = dest[current] & 0xff;
    //  auto event_header = timingHeader;
    //  printf("Size %u B | Dest %u.%u | Transition id %d | pulse id %014lx | env %08x | event counter %u | index %u\n",
    //         size, lane, vc, transitionId, event_header->pulseId(), event_header->env, event_header->evtCounter, index);
    //  //for(unsigned i=0; i<32; i++) //i<((size+3)>>2); i++)
    //  //  printf("%08x%c",reinterpret_cast<uint32_t*>(m_pool.dmaBuffers[index])[i], (i&7)==7 ? '\n':' ');
    //}
    auto rogs = timingHeader->readoutGroups();
    if ((rogs & (1 << m_para.partition)) == 0) {
        logging::debug("%s @ %u.%09u (%014lx) without common readout group (%u) in env 0x%08x",
                       TransitionId::name(transitionId),
                       timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                       timingHeader->pulseId(), m_para.partition, timingHeader->env);
        ++m_lastComplete;
        handleBrokenEvent(*event);
        freeDma(event);                 // Leaves event mask = 0
        ++m_nNoComRoG;
        return nullptr;
    }
    if (transitionId == TransitionId::SlowUpdate) {
        uint16_t missingRogs = m_para.rogMask & ~rogs;
        if (missingRogs) {
            logging::debug("%s @ %u.%09u (%014lx) missing readout group(s) (0x%04x) in env 0x%08x",
                           TransitionId::name(transitionId),
                           timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                           timingHeader->pulseId(), missingRogs, timingHeader->env);
            ++m_lastComplete;
            handleBrokenEvent(*event);
            freeDma(event);             // Leaves event mask = 0
            ++m_nMissingRoGs;
            return nullptr;
        }
    }

    const uint32_t* data = reinterpret_cast<const uint32_t*>(timingHeader);
    logging::debug("PGPReader  lane %u  size %u  hdr %016lx.%016lx.%08x  flag 0x%x  err 0x%x",
                   lane, size,
                   reinterpret_cast<const uint64_t*>(data)[0], // PulseId
                   reinterpret_cast<const uint64_t*>(data)[1], // Timestamp
                   reinterpret_cast<const uint32_t*>(data)[4], // env
                   flag, err);

    if (event->mask == m_para.laneMask) {
        // Allocate a pebble buffer once the event is built
        event->pebbleIndex = m_pool.allocate(); // This can block

        if (transitionId != TransitionId::L1Accept) {
            if (transitionId != TransitionId::SlowUpdate) {
                logging::info("PGPReader  saw %s @ %u.%09u (%014lx)",
                              TransitionId::name(transitionId),
                              timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                              timingHeader->pulseId());
            }
            else {
                logging::debug("PGPReader  saw %s @ %u.%09u (%014lx)",
                               TransitionId::name(transitionId),
                               timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                               timingHeader->pulseId());
            }
            if (transitionId == TransitionId::BeginRun) {
                resetEventCounter();
            }
        }
        if (evtCounter != ((m_lastComplete + 1) & 0xffffff)) {
            auto evtCntDiff = evtCounter - m_lastComplete;
            logging::error("%sPGPReader: Jump in TimingHeader evtCounter %u -> %u | difference %d, DMA size %u%s",
                           RED_ON, m_lastComplete, evtCounter, evtCntDiff, size, RED_OFF);
            logging::error("new data: %08x %08x %08x %08x %08x %08x  (%s)",
                           data[0], data[1], data[2], data[3], data[4], data[5], TransitionId::name(transitionId));
            logging::error("lastData: %08x %08x %08x %08x %08x %08x  (%s)",
                           m_lastData[0], m_lastData[1], m_lastData[2], m_lastData[3], m_lastData[4], m_lastData[5], TransitionId::name(m_lastTid));
            handleBrokenEvent(*event);
            freeDma(event);             // Leaves event mask = 0
            m_pool.freePebble();        // Avoid leaking pebbles on errors
            ++m_nPgpJumps;
            return nullptr;             // Throw away out-of-sequence events
        }
        m_lastComplete = evtCounter;
        m_lastTid = transitionId;
        memcpy(m_lastData, data, 24);

        // Allocate a transition datagram from the pool.  Since a
        // SPSCQueue is used (not an SPMC queue), this can be done here,
        // but not in the workers or there will be concurrency issues.
        if (transitionId != TransitionId::L1Accept) {
            uint32_t evtIndex = event->pebbleIndex;
            m_pool.transitionDgrams[evtIndex] = m_pool.allocateTr();
            if (!m_pool.transitionDgrams[evtIndex]) {
                ++m_nNoTrDgrams;
                freeDma(event);         // Leaves event mask = 0
                m_pool.freePebble();    // Avoid leaking pebbles on errors
                return nullptr;         // Can happen during shutdown
            }
        }
    }
    else {
        return nullptr;                 // Event is still incomplete
    }

    return timingHeader;
}

std::chrono::nanoseconds PgpReader::age(const TimeStamp& time) const {
//    auto now = Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC).time_since_epoch();
//    auto tTh = (std::chrono::seconds    { time.seconds()     } +
//                std::chrono::nanoseconds{ time.nanoseconds() });
//    return now - tTh - m_tOffset;
    using ns_t = std::chrono::nanoseconds;
    return ns_t{ Pds::Eb::latency<ns_t>(time) };
}

void PgpReader::freeDma(PGPEvent* event)
{
    // DMA buffers must be freeable from multiple threads
    std::lock_guard<std::mutex> lock(m_lock);

    // Return buffers and reset event.  Careful with order here!
    // index could be reused as soon as dmaRetIndexes() completes
    for (int i=0; i<PGP_MAX_LANES; i++) {
        if (event->mask &  (1 << i)) {
            event->mask ^= (1 << i);    // Zero out mask before dmaRetIndexes()
            m_dmaIndices[m_count++] = event->buffers[i].index;
            if (m_count == m_dmaIndices.size()) {
                // Return buffers.  An index could be reused as soon as dmaRetIndexes() completes
                m_pool.freeDma(m_count, m_dmaIndices.data());
                m_count = 0;
            }
        }
    }
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
                       MemPool& pool, ZmqSocket& inprocSend, DrpBase& drp) :
  EbCtrbInBase(tPrms),
  m_pool(pool),
  m_drp(drp),
  m_tsId(-1u),
  m_fileWriter(std::max(pool.pebble.bufferSize(), para.maxTrSize), para.kwargs["directIO"] != "no"), // Default to "yes"
  m_smdWriter(std::max(pool.pebble.bufferSize(), para.maxTrSize)),
  m_writing(false),
  m_inprocSend(inprocSend),
  m_offset(0),
  m_chunkOffset(0),
  m_chunkRequest(false),
  m_chunkPending(false),
  m_configureBuffer(para.maxTrSize),
  m_damage(0),
  m_evtSize(0),
  m_latPid(0),
  m_latency(0),
  m_para(para)
{
}

int EbReceiver::_setupMetrics(const std::shared_ptr<Pds::MetricExporter> exporter)
{
    std::map<std::string, std::string> labels
        {{"instrument", m_para.instrument},
         {"partition", std::to_string(m_para.partition)},
         {"detname", m_para.detName},
         {"alias", m_para.alias}};
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
    exporter->add("DRP_transitionId", labels, Pds::MetricType::Gauge,   [&](){ return m_lastTid; });

    return 0;
}

int EbReceiver::connect(const std::shared_ptr<Pds::MetricExporter> exporter)
{
    m_lastTid = TransitionId::Unconfigure;

    if (exporter) {
      int rc = _setupMetrics(exporter);
      if (rc)  return rc;
    }

    // On the timing system DRP, EbReceiver needs to know its node ID
    if (m_para.detType == "ts")  m_tsId = m_drp.nodeId();

    int rc = this->EbCtrbInBase::connect(exporter);
    if (rc)  return rc;

    return 0;
}

void EbReceiver::unconfigure()
{
    closeFiles();                       // Close files when BeginRun has failed
    this->EbCtrbInBase::unconfigure();
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

void EbReceiver::resetCounters(bool all = false)
{
    EbCtrbInBase::resetCounters();

    if (all)  m_lastIndex = -1u;
    m_damage = 0;
    if (m_dmgType)  m_dmgType->clear();
    m_latency = 0;
}

void EbReceiver::_writeDgram(Dgram* dgram)
{
    size_t size = sizeof(*dgram) + dgram->xtc.sizeofPayload();
    m_fileWriter.writeEvent(dgram, size, dgram->time);

    // small data writing
    Smd smd;
    const void* bufEnd = m_smdWriter.buffer + sizeof(m_smdWriter.buffer);
    NamesId namesId(dgram->xtc.src.value(), NamesIndex::OFFSETINFO);
    Dgram* smdDgram = smd.generate(dgram, m_smdWriter.buffer, bufEnd, chunkSize(), size,
                                   m_smdWriter.namesLookup, namesId);
    m_smdWriter.writeEvent(smdDgram, sizeof(Dgram) + smdDgram->xtc.sizeofPayload(), smdDgram->time);
    m_offset += size;
}

void EbReceiver::process(const Pds::Eb::ResultDgram& result, unsigned index)
{
    bool error = false;
    if (index != ((m_lastIndex + 1) & (m_pool.nbuffers() - 1))) {
        logging::critical("%sEbReceiver: jumping index %u  previous index %u  diff %d%s",
                          RED_ON, index, m_lastIndex, index - m_lastIndex, RED_OFF);
        error = true;
    }

    Pds::EbDgram* dgram = (Pds::EbDgram*)m_pool.pebble[index];
    uint64_t pulseId = dgram->pulseId();
    TransitionId::Value transitionId = dgram->service();
    if (transitionId != TransitionId::L1Accept) {
        if (transitionId == 0) {
            logging::warning("transitionId == 0 in %s", __PRETTY_FUNCTION__);
        }
        dgram = m_pool.transitionDgrams[index];
        if (pulseId != dgram->pulseId()) {
            logging::critical("pulseId mismatch: pebble %014lx, trDgram %014lx, xor %014lx, diff %ld",
                              pulseId, dgram->pulseId(), pulseId ^ dgram->pulseId(), pulseId - dgram->pulseId());
            error = true;
        }
        if (transitionId != dgram->service()) {
            logging::critical("tid mismatch: pebble %u, trDgram %u", transitionId, dgram->service());
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
        logging::critical("tid mismatch: pebble %u, result %u", transitionId, result.service());
        error = true;
    }

    if (error) {
        logging::critical("idx     %8u, pid     %014lx, tid     %s, env     %08x", index, pulseId, TransitionId::name(transitionId), dgram->env);
        logging::critical("lastIdx %8u, lastPid %014lx, lastTid %s, lastEnv %08x", m_lastIndex, m_lastPid, TransitionId::name(m_lastTid));
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
    if ((transitionId != TransitionId::L1Accept)) {
        if (transitionId != TransitionId::SlowUpdate) {
            if (transitionId == TransitionId::Configure) {
                // Cache Configure Dgram for writing out after files are opened
                Dgram* configDgram = dgram;
                size_t size = sizeof(*configDgram) + configDgram->xtc.sizeofPayload();
                memcpy(m_configureBuffer.data(), configDgram, size);
            }
            if (transitionId == TransitionId::BeginRun)
              m_offset = 0;// reset for monitoring (and not recording)
            // send pulseId to inproc so it gets forwarded to the collection
            json msg = createPulseIdMsg(pulseId);
            m_inprocSend.send(msg.dump());

            logging::info("EbReceiver saw %s @ %u.%09u (%014lx)",
                           TransitionId::name(transitionId),
                          dgram->time.seconds(), dgram->time.nanoseconds(), pulseId);
        }
        else {
            logging::debug("EbReceiver saw %s @ %u.%09u (%014lx)",
                           TransitionId::name(transitionId),
                           dgram->time.seconds(), dgram->time.nanoseconds(), pulseId);
        }
    }
    else { // L1Accept
        // On just the timing system DRP, save the trigger information
        if (m_drp.nodeId() == m_tsId) {
            const void* bufEnd = (char*)dgram + m_pool.bufferSize();
            m_drp.detector().event(*dgram, bufEnd, result);
        }
    }

    if (m_writing && !m_chunkRequest && (transitionId == TransitionId::L1Accept)) {
        if (chunkSize() > DefaultChunkThresh) {
            // request chunking opportunity
            chunkRequestSet();
            logging::debug("%s: sending chunk request (chunkSize() > DefaultChunkThresh)", __PRETTY_FUNCTION__);
            json msg = createChunkRequestMsg();
            m_inprocSend.send(msg.dump());
        }
    }

    // To write/monitor event, require the commmon readout group to have triggered
    // Events for which the common RoG didn't trigger are counted as NoComRoG errors
    if (dgram->readoutGroups() & (1 << m_para.partition)) {
        if (m_writing) {                    // Won't ever be true for Configure
            // write event to file if it passes event builder or if it's a transition
            if (result.persist() || result.prescale()) {
                _writeDgram(dgram);
            }
            else if (transitionId != TransitionId::L1Accept) {
                if (transitionId == TransitionId::BeginRun) {
                    m_offset = 0; // reset offset when writing out a new file
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

        m_evtSize = sizeof(*dgram) + dgram->xtc.sizeofPayload();

        // Measure latency before sending dgram for monitoring
        if (dgram->pulseId() - m_latPid > 1300000/14) { // 10 Hz
            m_latency = Pds::Eb::latency<us_t>(dgram->time);
            m_latPid = dgram->pulseId();
        }

        auto& mon = m_drp.mebContributor();
        if (mon.enabled()) {
            // L1Accept
            if (result.isEvent()) {
                if (result.monitor()) {
                    mon.post(dgram, result.monBufNo());
                }
            }
            // Other Transition
            else {
                mon.post(dgram);
            }
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

    // Free the pebble datagram buffer
    m_pool.freePebble();
}


class PV : public Pds_Epics::PVBase
{
public:
  PV(const char* pvName) : PVBase(pvName) {}
  virtual ~PV() {}
public:
  void updated() {}
  bool ready()   { return getComplete(); }
};

static bool _pvVectElem(const std::shared_ptr<PV> pv, unsigned element, double& value)
{
  if (!pv || !pv->ready())  return false;

  value = pv->getVectorElemAt<double>(element);

  return true;
}

DrpBase::DrpBase(Parameters& para, MemPool& pool_, Detector& det, ZmqContext& context) :
    pool(pool_), m_para(para), m_det(det), m_inprocSend(&context, ZMQ_PAIR)
{
    // Try to reduce clutter in grafana by picking the same port on each invocation.
    // Since DRPs on the same node have unique lane masks use the lowest bit set as an
    // offset into the port space.  If the port is in use, a search is done.
    unsigned portOffset = ffs(para.laneMask) - 1; // Assumes laneMask is never 0
    size_t found = para.device.rfind('_');
    if ((found != std::string::npos) && isdigit(para.device[found+1])) {
        portOffset += PGP_MAX_LANES * std::stoi(para.device.substr(found+1, para.device.size()));
    }
    m_exposer = Pds::createExposer(para.prometheusDir, _getHostName(), portOffset);

    m_tPrms.instrument = para.instrument;
    m_tPrms.partition  = para.partition;
    m_tPrms.alias      = para.alias;
    m_tPrms.detName    = para.detName;
    m_tPrms.detSegment = para.detSegment;
    m_tPrms.maxEntries = m_para.kwargs["batching"] == "no" ? 1 : Pds::Eb::MAX_ENTRIES; // Default to "yes"
    m_tPrms.core[0]    = -1;
    m_tPrms.core[1]    = -1;
    m_tPrms.verbose    = para.verbose;
    m_tPrms.kwargs     = para.kwargs;
    m_tebContributor = std::make_unique<Pds::Eb::TebContributor>(m_tPrms, pool.nbuffers());

    m_mPrms.instrument = para.instrument;
    m_mPrms.partition  = para.partition;
    m_mPrms.alias      = para.alias;
    m_tPrms.detName    = para.detName;
    m_tPrms.detSegment = para.detSegment;
    m_mPrms.maxEvSize  = pool.bufferSize();
    m_mPrms.maxTrSize  = para.maxTrSize;
    m_mPrms.verbose    = para.verbose;
    m_mPrms.kwargs     = para.kwargs;
    m_mebContributor = std::make_unique<Pds::Eb::MebContributor>(m_mPrms);

    m_ebRecv = std::make_unique<EbReceiver>(m_para, m_tPrms, pool, m_inprocSend, *this);

    m_inprocSend.connect("inproc://drp");

    if (para.outputDir.empty()) {
        logging::info("Output dir: n/a");
    } else {
        // Induce the automounter to mount in case user enables recording
        struct stat statBuf;
        std::string statPth = para.outputDir + "/" + para.instrument;
        logging::info("Output dir: %s", statPth.c_str());
        if (::stat(statPth.c_str(), &statBuf) < 0) {
            logging::error("stat(%s) error: %m", statPth.c_str());
        } else {
            logging::info("Output dir: ready");
        }
    }

    //  Add pva_addr to the environment
    if (para.kwargs.find("pva_addr")!=para.kwargs.end()) {
        const char* a = para.kwargs["pva_addr"].c_str();
        char* p = getenv("EPICS_PVA_ADDR_LIST");
        char envBuff[256];
        if (p)
            snprintf(envBuff,sizeof(envBuff), "%s %s", p, a);
        else
            snprintf(envBuff, sizeof(envBuff), "%s", a);
        logging::info("Setting env %s\n", envBuff);
        if (setenv("EPICS_PVA_ADDR_LIST",envBuff,1))
            perror("setenv pva_addr");
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
    if (rc)  {
        logging::critical("Error starting EbReceiver connection");
        abort();
    }

    json info = {{"drp_port", m_tPrms.port},
                 {"num_buffers", pool.nbuffers()},
                 {"max_ev_size", pool.bufferSize()},
                 {"max_tr_size", m_para.maxTrSize}};

    return info;
}

int DrpBase::setupMetrics(const std::shared_ptr<Pds::MetricExporter> exporter)
{
    std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                              {"partition", std::to_string(m_para.partition)},
                                              {"detname", m_para.detName},
                                              {"detseg", std::to_string(m_para.detSegment)},
                                              {"alias", m_para.alias}};
    exporter->add("drp_port_rcv_rate", labels, Pds::MetricType::Rate,
                  [](){return 4*readInfinibandCounter("port_rcv_data");});

    exporter->add("drp_port_xmit_rate", labels, Pds::MetricType::Rate,
                  [](){return 4*readInfinibandCounter("port_xmit_data");});

    exporter->add("drp_dma_in_use", labels, Pds::MetricType::Gauge,
                  [&](){return pool.dmaInUse();});
    exporter->constant("drp_dma_in_use_max", labels, pool.nDmaBuffers());

    exporter->add("drp_pebble_in_use", labels, Pds::MetricType::Gauge,
                  [&](){return pool.inUse();});
    exporter->constant("drp_pebble_in_use_max", labels, pool.nbuffers());

    exporter->addFloat("drp_deadtime", labels,
                       [&](double& value){return _pvVectElem(m_deadtimePv, m_xpmPort, value);});

    return 0;
}

std::string DrpBase::connect(const json& msg, size_t id)
{
    // If triggers had been left running, they will have been stopped during Allocate
    // Flush anything that accumulated
    pgpFlush();

    // Save a copy of the json so we can use it to connect to the config database on configure
    m_connectMsg = msg;
    m_collectionId = id;

    // If the exporter already exists, replace it so that previous metrics are deleted
    if (m_exposer) {
        m_exporter = std::make_shared<Pds::MetricExporter>();
        m_exposer->RegisterCollectable(m_exporter);
    }

    if (m_exporter) {
        if (setupMetrics(m_exporter)) {
            return std::string{"Failed to set up metrics"};
        }
    }

    int rc = parseConnectionParams(msg["body"], id);
    if (rc) {
        return std::string{"Connection parameters error - see log"};
    }

    rc = m_tebContributor->connect(m_exporter);
    if (rc) {
        return std::string{"TebContributor connect failed"};
    }
    if (m_mPrms.addrs.size() != 0) {
        rc = m_mebContributor->connect(m_exporter);
        if (rc) {
            return std::string{"MebContributor connect failed"};
        }
    }

    rc = m_ebRecv->connect(m_exporter);
    if (rc) {
        return std::string{"EbReceiver connect failed"};
    }

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
        rc = m_mebContributor->configure();
        if (rc) {
            return std::string{"MebContributor configure failed"};
        }
    }

    rc = m_ebRecv->EbCtrbInBase::configure(m_numTebBuffers);
    if (rc) {
        return std::string{"EbReceiver configure failed"};
    }

    printParams();

    // start eb receiver thread
    m_tebContributor->startup(*m_ebRecv);

    // Same time as the TEBs and MEBs
    m_tebContributor->resetCounters();
    m_mebContributor->resetCounters();
    m_ebRecv->resetCounters(true);
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
            msg = m_ebRecv->openFiles(m_para, runInfo, _getHostName(), m_nodeId);
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
    Alg runInfoAlg("runinfo", 0, 0, 1);
    NamesId runInfoNamesId(xtc.src.value(), NamesIndex::RUNINFO);
    Names& runInfoNames = *new(xtc, bufEnd) Names(bufEnd,
                                                  "runinfo", runInfoAlg,
                                                  "runinfo", "", runInfoNamesId);
    RunInfoDef myRunInfoDef;
    runInfoNames.add(xtc, bufEnd, myRunInfoDef);
    namesLookup[runInfoNamesId] = NameIndex(runInfoNames);
}

void DrpBase::runInfoData(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, const RunInfo& runInfo)
{
    NamesId runInfoNamesId(xtc.src.value(), NamesIndex::RUNINFO);
    CreateData runinfo(xtc, bufEnd, namesLookup, runInfoNamesId);
    runinfo.set_string(RunInfoDef::EXPT, runInfo.experimentName.c_str());
    runinfo.set_value(RunInfoDef::RUNNUM, runInfo.runNumber);
}

void DrpBase::chunkInfoSupport(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup)
{
    logging::debug("entered %s", __PRETTY_FUNCTION__);
    Alg chunkInfoAlg("chunkinfo", 0, 0, 1);
    NamesId chunkInfoNamesId(xtc.src.value(), NamesIndex::CHUNKINFO);
    Names& chunkInfoNames = *new(xtc, bufEnd) Names(bufEnd,
                                                    "chunkinfo", chunkInfoAlg,
                                                    "chunkinfo", "", chunkInfoNamesId);
    ChunkInfoDef myChunkInfoDef;
    chunkInfoNames.add(xtc, bufEnd, myChunkInfoDef);
    namesLookup[chunkInfoNamesId] = NameIndex(chunkInfoNames);
}

void DrpBase::chunkInfoData(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, const ChunkInfo& chunkInfo)
{
    NamesId chunkInfoNamesId(xtc.src.value(), NamesIndex::CHUNKINFO);
    CreateData chunkinfo(xtc, bufEnd, namesLookup, chunkInfoNamesId);
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

    if (m_exporter) {
        m_exporter.reset();
    }
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
        logging::error("Document '%s_0' not found in ConfigDb/%s",
                       triggerConfig.c_str(), configAlias.c_str());
        return -1;
    }
    bool buildAll = top.HasMember("buildAll") && top["buildAll"].GetInt()==1;

    std::string buildDets("---");
    if (top.HasMember("buildDets"))
        buildDets = top["buildDets"].GetString();

    if (!(buildAll || buildDets.find(m_para.detName))) {
        logging::info("This DRP is not contributing trigger input data: "
                      "buildAll is False and '%s' was not found in ConfigDb/%s/%s_0",
                      m_para.detName.c_str(), configAlias.c_str(), triggerConfig.c_str());
        m_tPrms.contractor = 0;    // This DRP won't provide trigger input data
        m_triggerPrimitive = nullptr;
        m_tPrms.maxInputSize = sizeof(Pds::EbDgram); // Revisit: 0 is bad
        return 0;
    }
    m_tPrms.contractor = m_tPrms.readoutGroup;

    if (!top.HasMember("soname")) {
        logging::error("Key 'soname' not found in Document %s", triggerConfig.c_str());
        return -1;
    }
    std::string soname(top["soname"].GetString());
    if (m_det.gpuDetector())  soname += "_gpu";

    //  Look for the detector-specific producer first
    std::string symbol("create_producer");
    symbol +=  "_" + m_para.detName;
    m_triggerPrimitive = m_trigPrimFactory.create(soname, symbol);
    if (m_triggerPrimitive) {
        logging::info("Created detector-specific TriggerPrimitive [%s]", symbol.c_str());
    }
    else {
        // Now try the generic producer
        symbol = std::string("create_producer");
        m_triggerPrimitive = m_trigPrimFactory.create(soname, symbol);
        if (!m_triggerPrimitive) {
            logging::error("Failed to create TriggerPrimitive; try '-v'");
            return -1;
        }
        logging::info("Created generic TriggerPrimitive [%s]", symbol.c_str());
    }
    m_tPrms.maxInputSize = sizeof(Pds::EbDgram) + m_triggerPrimitive->size();

    if (m_triggerPrimitive->configure(top, m_connectMsg, m_collectionId)) {
        logging::error("TriggerPrimitive::configure() failed");
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

    // Connect to an XPM PV to give Prometheus access to our deadtime
    std::string pv_base = body["control"]["0"]["control_info"]["pv_base"];
    unsigned    xpm_id  = body["drp"][stringId]["connect_info"]["xpm_id"];
    unsigned    rog     = body["drp"][stringId]["det_info"]["readout"];
    std::string pv(pv_base  +
                   ":XPM:"  + std::to_string(xpm_id) +
                   ":PART:" + std::to_string(rog) +
                   ":DeadFLnk");
    m_deadtimePv = std::make_shared<PV>(pv.c_str());
    m_xpmPort    = body["drp"][stringId]["connect_info"]["xpm_port"];

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
    m_supervisorIpPort.clear();
    m_isSupervisor = false;

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
        // Its range must be >= that of any secondary RoG.
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

        // For DrpPython's broadcasting of calibration constants
        std::string alias = it.value()["proc_info"]["alias"];
        if (m_supervisorIpPort.empty() && alias.find(m_para.detName) != std::string::npos) {
            enum {base_port = 32256};
            std::string ip = it.value()["connect_info"]["nic_ip"]; // Use IB, if available
            unsigned id = it.value()["connect_info"]["xpm_id"];    // Supervisor's xpm_id
            uint16_t port = base_port + id * 8 + m_para.partition;
            m_supervisorIpPort = ip + ":" + std::to_string(port);  // Supervisor's IP and port
            m_isSupervisor = alias == m_para.alias;                // True if we're the supervisor
        }
    }

    if (body.find("tpr") != body.end()) {
        for (auto it : body["tpr"].items()) {
            // Build readout group mask for ignoring other partitions' RoGs
            unsigned rog(it.value()["det_info"]["readout"]);
            if (rog < Pds::Eb::NUM_READOUT_GROUPS) {
                m_para.rogMask |= 1 << rog;
            }
            else {
                logging::warning("Ignoring Readout Group %d > max (%d)", rog, Pds::Eb::NUM_READOUT_GROUPS - 1);
            }
        }
    }

    // Disallow non-common RoG DRPs from having more buffers than the common one
    // because a buffer index based on the common RoG DRPs' won't be able to
    // reach the higher buffer numbers.  Can't use an index based on the largest
    // non-common RoG DRP because it would overrun the common RoG DRPs' region.
    if ((maxBuffers > m_numTebBuffers) && bufErr) {
        logging::error("Pebble buffer count (%u) must be <= the common RoG's (%u)\n",
                       pool.nbuffers(), m_numTebBuffers);
        rc = 1;
    }

    m_mPrms.addrs.clear();
    m_mPrms.ports.clear();
    m_mPrms.maxEvents.clear();
    if (body.find("meb") != body.end()) {
        for (auto it : body["meb"].items()) {
            unsigned mebId = it.value()["meb_id"];
            std::string address = it.value()["connect_info"]["nic_ip"];
            std::string port = it.value()["connect_info"]["meb_port"];
            logging::debug("MEB: %u  %s:%s", mebId, address.c_str(), port.c_str());
            m_mPrms.addrs.push_back(address);
            m_mPrms.ports.push_back(port);
        }
        m_mPrms.maxEvents.resize(m_mPrms.addrs.size());
        for (auto it : body["meb"].items()) {
            unsigned mebId = it.value()["meb_id"];
            unsigned count = it.value()["connect_info"]["max_ev_count"];
            m_mPrms.maxEvents[mebId] = count;
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
    printf("  Instrument:                   %s\n",                 m_tPrms.instrument.c_str());
    printf("  Partition:                    %u\n",                 m_tPrms.partition);
    printf("  Alias (detName, detSeg):      %s ('%s', %u)\n",      m_tPrms.alias.c_str(), m_tPrms.detName.c_str(), m_tPrms.detSegment);
    printf("  Readout group receipient:     0x%02x\n",             m_tPrms.readoutGroup);
    printf("  Readout group contractor:     0x%02x\n",             m_tPrms.contractor);
    printf("  Bit list of TEBs:             0x%016lx, cnt: %zu\n", m_tPrms.builders,
                                                                   std::bitset<64>(m_tPrms.builders).count());
    printf("  Number of MEBs:               %zu\n",                m_mPrms.addrs.size());
    printf("  Batching state:               %s\n",                 m_tPrms.maxEntries > 1 ? "Enabled" : "Disabled");
    printf("  Batch duration:               0x%014x = %u ticks\n", m_tPrms.maxEntries, m_tPrms.maxEntries);
    printf("  Batch pool depth:             0x%08x = %u\n",        pool.nbuffers() / m_tPrms.maxEntries, pool.nbuffers() / m_tPrms.maxEntries);
    printf("  Max # of entries / batch:     0x%08x = %u\n",        m_tPrms.maxEntries, m_tPrms.maxEntries);
    printf("  # of TEB contrib.   buffers:  0x%08x = %u\n",        pool.nbuffers(), pool.nbuffers());
    printf("  # of TEB transition buffers:  0x%08x = %u\n",        TEB_TR_BUFFERS, TEB_TR_BUFFERS);
    printf("  Max  TEB contribution  size:  0x%08zx = %zu\n",      m_tPrms.maxInputSize, m_tPrms.maxInputSize);
    printf("  Max  MEB L1Accept      size:  0x%08zx = %zu\n",      m_mPrms.maxEvSize, m_mPrms.maxEvSize);
    printf("  Max  MEB transition    size:  0x%08zx = %zu\n",      m_mPrms.maxTrSize, m_mPrms.maxTrSize);
    for (unsigned i = 0; i < m_mPrms.maxEvents.size(); ++i)
      printf("  # of MEB %u contrib. buffers:  0x%08x = %u\n",      i, m_mPrms.maxEvents[i], m_mPrms.maxEvents[i]);
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
