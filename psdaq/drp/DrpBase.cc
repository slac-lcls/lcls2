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
#include "DrpBase.hh"
#include "RunInfoDef.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/aes-stream-drivers/DataDriver.h"
#include "psdaq/aes-stream-drivers/DmaDest.h"
#include "psdaq/epicstools/PVBase.hh"

#include "rapidjson/document.h"

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using namespace XtcData;
using namespace Pds::Eb;
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

static const unsigned EvtCtrMask = 0xffffff;

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
    m_trBufSize   = algnSz * ((trBufSize + algnSz - 1) / algnSz);

    // Round up to an integer number of pages for both L1 and transition buffers
    size_t pgSz   = sysconf(_SC_PAGESIZE); // For shmem/MMU
    size_t l1Sz   = pgSz * (((nL1Buffers*m_bufferSize + pgSz - 1) / pgSz) + 1); // +1 for overrun detection
    size_t trSz   = pgSz * (((nTrBuffers*m_trBufSize  + pgSz - 1) / pgSz) + 1); // +1 for overrun detection
    m_size        = l1Sz + trSz;
    m_buffer      = nullptr;
    int    ret    = posix_memalign((void**)&m_buffer, pgSz, m_size);
    if (ret) {
        logging::critical("Failed to create pebble of size %zu for %u transitions of %zu B and %u L1Accepts of %zu B: %s",
                          m_size, nTrBuffers, trBufSize, nL1Buffers, l1BufSize, strerror(ret));
        throw "Pebble creation failed";
    }
    m_trBuffer = m_buffer + l1Sz;

    m_nTrBuffers = nTrBuffers;
    m_nL1Buffers = nL1Buffers;

    logging::info("Allocated %.1f GB pebble for %u transitions of %zu B and %u L1Accepts of %zu B",
                  double(m_size)/1e9, nTrBuffers, trBufSize, nL1Buffers, l1BufSize);

    // Store a sentinel value at the end of the buffers for debugging overruns
    auto buf = m_buffer + m_bufferSize - sizeof(uint32_t);
    for (unsigned i = 0; i < nL1Buffers; ++i) {
        *(uint32_t*)buf = 0xcdcdcdcd;
        buf += m_bufferSize;
    }
    *(uint32_t*)(buf - m_bufferSize + sizeof(uint32_t)) = 0xcdcdcdcd; // Also 1st word of page after L1 buffer pool
    buf = m_trBuffer + m_trBufSize - sizeof(uint32_t);
    for (unsigned i = 0; i < nTrBuffers; ++i) {
        *(uint32_t*)buf = 0xefefefef;
        buf += m_trBufSize;
    }
    *(uint32_t*)(buf - m_trBufSize + sizeof(uint32_t)) = 0xefefefef; // Also 1st word of page after transition pool
}

MemPool::MemPool(const Parameters& para) :
    m_transitionBuffers(nextPowerOf2(Pds::Eb::TEB_TR_BUFFERS)), // See eb.hh
    m_dmaAllocs(0),
    m_dmaFrees(0),
    m_allocs(0),
    m_frees(0),
    m_dmaOverrun(0),
    m_l1Overrun(0),
    m_trOverrun(0),
    m_nDmaReadErr(0),
    m_nDmaRetErr(0)
{
}

void MemPool::_initialize(const Parameters& para)
{
    logging::info("dmaCount %u,  dmaSize %u", m_dmaCount, m_dmaSize);

    // make sure there are more buffers in the pebble than in the pgp driver
    // otherwise the pebble buffers will be overwritten by the pgp event builder
    m_nDmaBuffers = nextPowerOf2(m_dmaCount);
    if (m_nDmaBuffers > EvtCtrMask+1) {
        logging::critical("nDmaBuffers (%u) can't exceed evtCounter range (0:%u)",
                          m_nDmaBuffers, EvtCtrMask);
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
    logging::info("nL1Buffers %u,  pebble buffer size %zu B", m_nbuffers, pebble.bufferSize());
    logging::info("nTrBuffers %u,  transition buffer size %zu B", nTrBuffers, pebble.trBufSize());

    pgpEvents.resize(m_nDmaBuffers);
    transitionDgrams.resize(m_nbuffers);

    // Put the transition buffer pool at the end of the pebble buffers
    uint8_t* buffer = pebble.trBuffer();
    for (size_t i = 0; i < m_transitionBuffers.size(); i++) {
        m_transitionBuffers.push(&buffer[i * pebble.trBufSize()]);
    }
}

unsigned MemPool::allocateDma()
{
    // Actually, the DMA buffer is allocated by the f/w and we only account for it here

    auto allocs = m_dmaAllocs.fetch_add(1, std::memory_order_acq_rel);

    return allocs;
}

ssize_t MemPool::readDma(uint32_t count, int32_t* ret, uint32_t* index, uint32_t* flags, uint32_t* errors, uint32_t* dest)
{
    auto rc = dmaReadBulkIndex(fd(), count, ret, index, flags, errors, dest);
    if (rc < 0) [[unlikely]] {
        if (m_nDmaReadErr++ == 0)  logging::error("dmaReadBulkIndex error %d: %m", rc);
        else                       logging::debug("dmaReadBulkIndex error %d: %m", rc);
    }
    return rc;
}

ssize_t MemPool::freeDma(unsigned count, uint32_t* indices)
{
    auto rc = _freeDma(count, indices);
    if (rc < 0) [[unlikely]] {
        if (m_nDmaRetErr++ == 0)  logging::error("dmaRetIndexes error %d: %m", rc);
        else                      logging::debug("dmaRetIndexes error %d: %m", rc);
    }

    if (!rc)  m_dmaFrees.fetch_add(count, std::memory_order_acq_rel);

    return rc;
}

/** Pebble buffers must be freed in the same order in which they were allocated */
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

/** Pebble buffers must be freed in the same order in which they were allocaed */
void MemPool::freePebble(unsigned index)
{
    auto frees  = m_frees.fetch_add(1, std::memory_order_acq_rel);
    asm volatile("mfence" ::: "memory");
    auto allocs = m_allocs.load(std::memory_order_acquire);

    // Sanity check of freeing order
    if (index != (frees & (m_nbuffers - 1))) [[unlikely]] {
        static unsigned errCnt = 0;
        if (errCnt++ < 5) {
            logging::error("Next pebble index to free (%u) is not the one expected (%u)",
                           index, frees & (m_nbuffers - 1));
        }
        //exit(EXIT_FAILURE);
    }

    // Release when all pebble buffers were in use but now one is free
    if (allocs - frees == m_nbuffers) {
        std::lock_guard<std::mutex> lock(m_lock);
        m_condition.notify_one();
    }

    // Check that the sentinel value at the end of the buffer is still there
    const auto dgram = (Pds::EbDgram*)pebble[frees & (m_nbuffers - 1)];
    const auto word = (uint32_t*)((uint8_t*)dgram + pebble.bufferSize() - sizeof(uint32_t));
    auto idx = ((uint8_t*)dgram - pebble.buffer()) / pebble.bufferSize();
    auto sz = sizeof(*dgram) + dgram->xtc.sizeofPayload();
    if (word[0] != 0xcdcdcdcd) [[unlikely]] {
        if (!(m_l1Overrun & 0x01)) {
            logging::error("(%014lx, %u.%09u, %s, %zu) L1 buffer[%zu] overrun: %08x vs %08x",
                           dgram->pulseId(), dgram->time.seconds(), dgram->time.nanoseconds(),
                           TransitionId::name(dgram->service()), sz, idx, word[0], 0xcdcdcdcd);
            m_l1Overrun |= 0x01;
        }
    }
    // Check that the sentinel value at start of the space after the buffer pool is still there
    if ((idx == pebble.nL1Buffers()-1) && (word[1] != 0xcdcdcdcd)) [[unlikely]] {
        if (!(m_l1Overrun & 0x02)) {
            logging::error("(%014lx, %u.%09u, %s, %zu) L1 buffer[%zu] pool overrun: %08x %08x vs %08x",
                           dgram->pulseId(), dgram->time.seconds(), dgram->time.nanoseconds(),
                           TransitionId::name(dgram->service()), sz, idx, word[0], word[1], 0xcdcdcdcd);
            m_l1Overrun |= 0x02;
        }
    }
}

void MemPool::flushPebble()
{
    while (inUse()) {
        freePebble(m_frees.load(std::memory_order_acquire) & (m_nbuffers - 1));
    }
}

Pds::EbDgram* MemPool::allocateTr()
{
    void* dgram = nullptr;
    if (!m_transitionBuffers.pop(dgram)) [[unlikely]] {
        // See comments for setting the number of transition buffers in eb.hh
        return nullptr;
    }
    return static_cast<Pds::EbDgram*>(dgram);
}

void MemPool::freeTr(Pds::EbDgram* dgram)
{
    // Do this check before freeing the dgram in case it is reallocated
    // Check that the sentinel value at the end of the buffer is still there
    const auto word = (uint32_t*)((uint8_t*)dgram + pebble.trBufSize() - sizeof(uint32_t));
    auto idx = ((uint8_t*)dgram - pebble.trBuffer()) / pebble.trBufSize();
    auto sz = sizeof(*dgram) + dgram->xtc.sizeofPayload();
    if (word[0] != 0xefefefef) [[unlikely]] {
        if (!(m_trOverrun & 0x01)) {
            logging::error("(%014lx, %u.%09u, %s, %zu) Tr buffer[%zu] overrun: %08x vs %08x",
                           dgram->pulseId(), dgram->time.seconds(), dgram->time.nanoseconds(),
                           TransitionId::name(dgram->service()), sz, idx, word[0], 0xefefefef);
            m_trOverrun |= 0x01;
        }
    }
    // Check that the sentinel value at start of the space after the buffer pool is still there
    if ((idx == pebble.nTrBuffers()-1) && (word[1] != 0xefefefef)) [[unlikely]] {
        if (!(m_trOverrun & 0x02)) {
            logging::error("(%014lx, %u.%09u, %s, %zu) Tr buffer[%zu] pool overrun: %08x %08x vs %08x",
                           dgram->pulseId(), dgram->time.seconds(), dgram->time.nanoseconds(),
                           TransitionId::name(dgram->service()), sz, idx, word[0], word[1], 0xefefefef);
            m_trOverrun |= 0x02;
        }
    }

    m_transitionBuffers.push(dgram);
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

    m_dmaOverrun = 0;
    m_l1Overrun = 0;
    m_trOverrun = 0;

    m_nDmaReadErr = 0;
    m_nDmaRetErr = 0;
}

void MemPool::shutdown()
{
    m_transitionBuffers.shutdown();

    if (m_nDmaReadErr)  logging::warning("dmaReadBulkIndex failed %u times", m_nDmaReadErr);
    if (m_nDmaRetErr)   logging::warning("dmaRetIndexes failed %u times", m_nDmaRetErr);
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
        logging::critical("Failed to map DMA buffers: %m");
        abort();
    }

    // Store a sentinel value at the end of the buffers for debugging overruns
    for (unsigned i = 0; i < m_dmaCount; ++i) {
        uint8_t* buf = (uint8_t*)(dmaBuffers[i]);
        *(uint32_t*)(buf + m_dmaSize - sizeof(uint32_t)) = 0xabababab;
    }

    // Continue with initialization of the base class
    _initialize(para);
}

MemPoolCpu::~MemPoolCpu()
{
   auto rc = dmaUnMapDma(m_fd, dmaBuffers);
   if (rc) {
     logging::error("Failed to unmap DMA buffers: %m");
   }

   logging::debug("%s: Closing PGP device file descriptor", __PRETTY_FUNCTION__);
    close(m_fd);
}

ssize_t MemPoolCpu::_freeDma(unsigned count, uint32_t* indices)
{
    // Check that the sentinel value at the end of the buffer is still there
    for (unsigned i = 0; i < count; ++i) {
        auto idx = indices[i];
        const auto buffer = (uint8_t*)dmaBuffers[idx];
        const auto word = (uint32_t*)(buffer + m_dmaSize - sizeof(uint32_t));
        if (word[0] != 0xabababab) [[unlikely]] {
            if (!(m_dmaOverrun & 0x01)) {
                const auto th = (const Pds::TimingHeader*)buffer;
                logging::error("(%014lx, %u.%09u, %s) DMA buffer[%zu] overrun: %08x vs %08x",
                               th->pulseId(), th->time.seconds(), th->time.nanoseconds(),
                               TransitionId::name(th->service()), idx, word[0], 0xabababab);
                m_dmaOverrun |= 0x01;
            }
        }
        // The driver allocates the DMA pool, so we have no control over what comes after it
        // Unclear how to recognize overruns, so commenting this out for now
        //if ((idx == m_dmaCount-1) && (word[1] != 0xabababab)) [[unlikely]] {
        //    if (!(m_dmaOverrun & 0x02)) {
        //        const auto th = (const Pds::TimingHeader*)buffer;
        //        logging::error("(%014lx, %u.%09u, %s) DMA buffer[%zu] pool overrun: %08x %08x vs %08x",
        //                       th.pulseId(), th->time.seconds(), th->time.nanoseconds(),
        //                       TransitionId::name(th->service()), idx, word[0], word[1], 0xabababab);
        //        m_dmaOverrun |= 0x02;
        //    }
        //}
    }

    return dmaRetIndexes(m_fd, count, indices);
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
    m_tmo         {100},                // ms
    m_us          {1},                  // Must not be 0
    dmaRet        (maxRetCnt),
    dmaIndex      (maxRetCnt),
    dest          (maxRetCnt),
    dmaFlags      (maxRetCnt),
    dmaErrors     (maxRetCnt),
    m_lastComplete(0),
    m_lastTid     (TransitionId::Unconfigure),
    m_dmaIndices  (maxRetCnt),
    m_dmaRetCnt   (dmaFreeCnt),
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
    m_nNoTrDgrams (0),
    m_dmaOverrun  (false)
{
    // Ensure there are more DMA buffers than the size of the batch used to free them
    if (pool.dmaCount() < m_dmaIndices.size()) {
        logging::critical("nDmaIndices (%zu) must be >= dmaCount (%u)",
                          m_dmaIndices.size(), pool.dmaCount());
        abort();
    }

    m_pfd.fd = pool.fd();
    m_pfd.events = POLLIN;
    m_t0 = Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC_COARSE);

    pool.resetCounters();
    memset(m_lastData, 0, 24);
}

PgpReader::~PgpReader()
{
    flush();
}

int32_t PgpReader::read()
{
    if (m_tmo) {                     // Interrupt mode
        // Wait for DMAed data to become available
        if (poll(&m_pfd, 1, m_tmo) < 0) {
            logging::error("%s: poll() error: %m", __PRETTY_FUNCTION__);
        }
    }                                // Else polling mode

    auto rc = m_pool.readDma(dmaRet.size(), dmaRet.data(), dmaIndex.data(), dmaFlags.data(), dmaErrors.data(), dest.data());
    if (rc > 0) {
        auto t1 { Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC_COARSE) };

        auto dt = std::chrono::duration_cast<ms_t>(t1 - m_t0).count();
        m_tmo = dt/rc < 1 ? 0 : 1;   // Polling if rate > 1 kHz else interrupt mode
        if (!m_tmo)  m_us = 1;
        m_t0  = t1;
    } else {                         // Exponential back-off
        usleep(m_us);
        if (m_us < 1024)  m_us <<= 1;
    }

    return rc;
}

void PgpReader::flush()
{
    // DMA buffers must be freeable from multiple threads
    std::lock_guard<std::mutex> lock(m_lock);

    // Return DMA buffers queued for freeing
    if (m_count && !m_pool.freeDma(m_count, m_dmaIndices.data())) {
        m_count = 0;
    }

    // Also return DMA buffers queued for reading, without adjusting counters
    int32_t ret;
    while ( (ret = read()) > 0 ) {
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
    if (size >= m_pool.dmaSize()) [[unlikely]] {
        logging::critical("DMA overflowed buffer: %d vs %d", size, m_pool.dmaSize());
        abort();
    }
    if (index > m_pool.dmaCount()-1) [[unlikely]] {
        if (!m_dmaOverrun) {
            logging::error("DMA buffer index (%u) is out of range [0:%u]", index, m_pool.dmaCount()-1);
            m_dmaOverrun = true;
        }
    }

    const Pds::TimingHeader* timingHeader = det->getTimingHeader(index);

    // Measure TimingHeader arrival latency as early as possible
    if (timingHeader->pulseId() - m_latPid > 1300000/14) { // 10 Hz
        m_latency = std::chrono::duration_cast<us_t>(age(timingHeader->time)).count();
        m_latPid = timingHeader->pulseId();
    }
    if (timingHeader->error()) [[unlikely]] {
        if (m_nTmgHdrError++ < 5) {     // Limit prints at rate
            logging::error("Timing header error bit is set");
        }
    }

    uint32_t evtCounter = timingHeader->evtCounter & EvtCtrMask;
    uint32_t pgpIndex = evtCounter & (m_pool.nDmaBuffers() - 1);
    PGPEvent* event = &m_pool.pgpEvents[pgpIndex];
    DmaBuffer* buffer = &event->buffers[lane];
    buffer->size = size;
    buffer->index = index;
    if (((1 << lane) & m_para.laneMask) == 0) [[unlikely]] {
        logging::error("Lane %u is not in laneMask 0x%02", lane, m_para.laneMask);
    }
    if (event->mask & (1 << lane)) [[unlikely]] {
        logging::error("Lane %u is already set in event mask 0x%02x", lane, event->mask);
    }
    event->mask |= (1 << lane);

    m_pool.allocateDma(); // DMA buffer was allocated when f/w incremented evtCounter

    uint32_t flag = dmaFlags[current];
    uint32_t err  = dmaErrors[current];
    if (err) [[unlikely]] {
        if (m_nDmaErrors++ < 5) {       // Limit prints at rate
            logging::error("DMA with error 0x%x  flag 0x%x",err,flag);
        }
        // This assumes the DMA succeeded well enough that evtCounter is valid
        ++m_lastComplete;
        handleBrokenEvent(*event);
        freeDma(event);                 // Leaves event mask = 0
        return nullptr;
    }

    TransitionId::Value transitionId = timingHeader->service();
    const uint32_t* data = reinterpret_cast<const uint32_t*>(timingHeader);
    auto pid = reinterpret_cast<const uint64_t*>(data)[0]; // PulseId
    auto ts  = reinterpret_cast<const uint64_t*>(data)[1]; // Timestamp
    auto env = reinterpret_cast<const uint32_t*>(data)[4]; // env
    if (pid == 0ul || ts == 0ul || env == 0ul) [[unlikely]] {
        logging::critical("PGPReader received invalid data:");
        logging::critical("PGPReader  lane %u.%u  size %u  hdr %016lx.%016lx.%08x  flag 0x%x  err 0x%x",
                          lane, dest[current] & 0xff, size, pid, ts, env, flag, err);
        abort();
    }
    logging::debug("PGPReader  lane %u.%u  size %u  hdr %016lx.%016lx.%08x  flag 0x%x  err 0x%x",
                       lane, dest[current] & 0xff, size, pid, ts, env, flag, err);

    if (event->mask == m_para.laneMask) {
        if (transitionId == TransitionId::BeginRun) {
            resetEventCounter();        // Compensate for the ClearReadout sent before BeginRun
        }
        if (evtCounter != ((m_lastComplete + 1) & EvtCtrMask)) [[unlikely]] {
            if (m_lastTid != TransitionId::Unconfigure) {
                if ((m_nPgpJumps < 5) || m_para.verbose) { // Limit prints at rate
                    auto evtCntDiff = evtCounter - m_lastComplete;
                    logging::error("%sPGPReader: Jump in TimingHeader evtCounter %u -> %u | difference %d, DMA size %u%s",
                                   RED_ON, m_lastComplete, evtCounter, evtCntDiff, size, RED_OFF);
                    logging::error("new data: %08x %08x %08x %08x %08x %08x  (%s)",
                                   data[0], data[1], data[2], data[3], data[4], data[5], TransitionId::name(transitionId));
                    logging::error("lastData: %08x %08x %08x %08x %08x %08x  (%s)",
                                   m_lastData[0], m_lastData[1], m_lastData[2], m_lastData[3], m_lastData[4], m_lastData[5], TransitionId::name(m_lastTid));
                }
                // For multi-lane detectors, discard events with flakey lanes
                if (__builtin_popcount(m_para.laneMask) > 1) {
                    auto pgpIdx = (m_lastComplete + 1) & (m_pool.nDmaBuffers() - 1);
                    while (pgpIdx != pgpIndex) {
                        auto evt = &m_pool.pgpEvents[pgpIdx];
                        if (evt->mask != m_para.laneMask) {
                            if ((m_nPgpJumps < 5) || m_para.verbose) { // Limit prints at rate
                                logging::error("Discarding incomplete event at evtCounter %u: lanes 0x%02x",
                                               pgpIdx, evt->mask);
                            }
                            handleBrokenEvent(*evt);
                            freeDma(evt);
                        }
                        pgpIdx = (pgpIdx + 1) & (m_pool.nDmaBuffers() - 1);
                    }
                }
                ++m_nPgpJumps;
                // Try to handle out-of-sequence events
            } else if (transitionId != TransitionId::Configure) { // m_lastTid == Unconfigure
                freeDma(event);         // Leaves event mask = 0
                return nullptr;         // Drain everything before Configure
            }
        }
        m_lastComplete = evtCounter;
        m_lastTid = transitionId;
        memcpy(m_lastData, data, 24);

        auto rogs = timingHeader->readoutGroups();
        if ((rogs & (1 << m_para.partition)) == 0) [[unlikely]] {
            // Events without the common readout group would mess up the TEB and MEB, so filter them out here
            logging::debug("%s @ %u.%09u (%014lx) without common readout group (%u) in env 0x%08x",
                           TransitionId::name(transitionId),
                           timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                           timingHeader->pulseId(), m_para.partition, timingHeader->env);
            handleBrokenEvent(*event);
            freeDma(event);             // Leaves event mask = 0
            ++m_nNoComRoG;
            return nullptr;
        }
        if (transitionId == TransitionId::SlowUpdate) {
            uint16_t missingRogs = m_para.rogMask & ~rogs;
            if (missingRogs) [[unlikely]] {
                // SlowUpdates that don't have all readout groups triggered would mess up psana, so filter them out here
                // This is true for other transitions as well, but those are caught by control.py
                logging::debug("%s @ %u.%09u (%014lx) missing readout group(s) (0x%04x) in env 0x%08x",
                               TransitionId::name(transitionId),
                               timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                               timingHeader->pulseId(), missingRogs, timingHeader->env);
                handleBrokenEvent(*event);
                freeDma(event);         // Leaves event mask = 0
                ++m_nMissingRoGs;
                return nullptr;
            }
        }

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
        }

        // Allocate a transition datagram from the pool.  Since a
        // SPSCQueue is used (not an SPMC queue), this can be done here,
        // but not in the workers or there will be concurrency issues.
        Pds::EbDgram* trDgram = nullptr;
        if (transitionId != TransitionId::L1Accept) {
            trDgram = m_pool.allocateTr();
            if (!trDgram) [[unlikely]] {
                freeDma(event);         // Leaves event mask = 0
                ++m_nNoTrDgrams;
                return nullptr;         // Can happen during shutdown
            }
        }

        // Allocate a pebble buffer once the event is built
        event->pebbleIndex = m_pool.allocate(); // This can block

        if (transitionId != TransitionId::L1Accept) {
            // Store the empty transition dgram allocated above in the pebble
            m_pool.transitionDgrams[event->pebbleIndex] = trDgram;
        }

        return timingHeader;
    }

    return nullptr;                     // Event is still incomplete
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

    // If the previous attempt failed, try again
    if (m_count == m_dmaIndices.size()) [[unlikely]] {
        if (m_pool.freeDma(m_count, m_dmaIndices.data())) {
            return;                     // Failed again
        }
        m_count = 0;                    // Success: reset
    }

    // Return buffers and reset event.  Careful with order here!
    // index could be reused as soon as dmaRetIndexes() completes
    unsigned laneMask = m_para.laneMask;
    for (unsigned i = 0; laneMask; laneMask &= ~(1 << i++)) {
        if (event->mask &  (1 << i)) {
            event->mask ^= (1 << i);    // Zero out mask before dmaRetIndexes()
            auto idx = event->buffers[i].index;
            if (idx < m_pool.dmaCount()) [[likely]] {
                m_dmaIndices[m_count++] = idx;
                if (m_count >= m_dmaRetCnt) {
                    // Return buffers.  An index could be reused as soon as dmaRetIndexes() completes
                    if (!m_pool.freeDma(m_count, m_dmaIndices.data())) {
                        m_count = 0;    // Reset only on success
                    }
                }
            } else {
                logging::error("DMA buffer index %u is out of range [0:%u]\n",
                               idx, m_pool.dmaCount() - 1);
            }
        }
    }
}

std::string Drp::FileParameters::runName() const
{
    std::ostringstream ss;
    ss << m_experimentName <<
          "-r" << std::setfill('0') << std::setw(4) << m_runNumber <<
          "-s" << std::setw(3) << m_nodeId <<
          "-c" << std::setw(3) << m_chunkId;
    return ss.str();
}

TebReceiverBase::TebReceiverBase(const Parameters& para, DrpBase& drp) :
  EbCtrbInBase(drp.tebPrms()),
  m_pool(drp.pool),
  m_drp(drp),
  m_tsId(-1u),
  m_writing(false),
  m_inprocSend(drp.inprocSend()),
  m_offset(0),
  m_chunkOffset(0),
  m_chunkPending(false),
  m_chunkRequest(false),
  m_configureBuffer(para.maxTrSize),
  m_evtSize(0),
  m_latPid(0),
  m_latency(0),
  m_damage(0),
  m_para(para)
{
}

int TebReceiverBase::_setupMetrics(const std::shared_ptr<Pds::MetricExporter> exporter)
{
    std::map<std::string, std::string> labels
        {{"instrument", m_para.instrument},
         {"partition", std::to_string(m_para.partition)},
         {"detname", m_para.detName},
         {"alias", m_para.alias}};
    m_dmgType = exporter->histogram("DRP_DamageType", labels, 16);
    exporter->add("DRP_Damage"    ,   labels, Pds::MetricType::Gauge,   [&](){ return m_damage; });
    exporter->add("DRP_RecordSize",   labels, Pds::MetricType::Counter, [&](){ return m_offset; });
    exporter->add("DRP_evtSize",      labels, Pds::MetricType::Gauge,   [&](){ return m_evtSize; });
    exporter->add("DRP_evtLatency",   labels, Pds::MetricType::Gauge,   [&](){ return m_latency; });
    exporter->add("DRP_transitionId", labels, Pds::MetricType::Gauge,   [&](){ return m_lastTid; });

    return setupMetrics(exporter, labels);
}

int TebReceiverBase::connect(const std::shared_ptr<Pds::MetricExporter> exporter)
{
    m_lastTid = TransitionId::Unconfigure;

    if (exporter) {
      int rc = _setupMetrics(exporter);
      if (rc)  return rc;
    }

    // On the timing system DRP, TebReceiver needs to know its node ID
    if (m_para.detType == "ts")  m_tsId = m_drp.nodeId();

    int rc = this->EbCtrbInBase::connect(exporter);
    if (rc)  return rc;

    return 0;
}

void TebReceiverBase::unconfigure()
{
    closeFiles();                       // Close files when BeginRun has failed
    this->EbCtrbInBase::unconfigure();
}

std::string TebReceiverBase::openFiles(const RunInfo& runInfo)
{
    std::string retVal{};               // return empty string on success
    if (runInfo.runNumber) {
        m_chunkOffset = m_offset = 0;
        std::ostringstream ss;
        ss << runInfo.experimentName <<
              "-r" << std::setfill('0') << std::setw(4) << runInfo.runNumber <<
              "-s" << std::setw(3) << m_drp.nodeId() <<
              "-c000";
        std::string runName = ss.str();
        // data
        std::string exptDir{m_para.outputDir + "/" + m_para.instrument + "/" + runInfo.experimentName};
        local_mkdir(exptDir.c_str());
        std::string dataDir{exptDir + "/xtc"};
        local_mkdir(dataDir.c_str());
        std::string path{"/" + m_para.instrument + "/" + runInfo.experimentName + "/xtc/" + runName + ".xtc2"};
        std::string absolute_path{m_para.outputDir + path};
        std::string hostname{_getHostName()};
        // cpo suggests leaving this print statement in because
        // filesystems can hang in ways we can't timeout/detect
        // and this print statement may speed up debugging significantly.
        std::cout << "Opening file " << absolute_path << std::endl;
        logging::info("Opening file '%s'", absolute_path.c_str());
        if (fileWriter().open(absolute_path) == 0) {
            timespec tt; clock_gettime(CLOCK_REALTIME,&tt);
            json msg = createFileReportMsg(path, absolute_path, tt, tt, runInfo.runNumber, hostname);
            m_inprocSend.send(msg.dump());
        } else if (retVal.empty()) {
            retVal = {"Failed to open file '" + absolute_path + "'"};
        }
        // smalldata
        std::string smalldataDir{m_para.outputDir + "/" + m_para.instrument + "/" + runInfo.experimentName + "/xtc/smalldata"};
        local_mkdir(smalldataDir.c_str());
        std::string smalldata_path{"/" + m_para.instrument + "/" + runInfo.experimentName + "/xtc/smalldata/" + runName + ".smd.xtc2"};
        std::string smalldata_absolute_path{m_para.outputDir + smalldata_path};
        logging::info("Opening file '%s'", smalldata_absolute_path.c_str());
        if (smdWriter().open(smalldata_absolute_path) == 0) {
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
            m_fileParameters = std::make_unique<FileParameters>(m_para, runInfo, hostname, m_drp.nodeId());
        }
    }
    return retVal;
}

// return true if incremented chunkId
bool TebReceiverBase::advanceChunkId()
{
    bool status = false;
//  m_chunkPending_sem.take();
    if (!m_chunkPending) {
        logging::debug("%s: m_fileParameters->advanceChunkId()", __PRETTY_FUNCTION__);
        m_fileParameters->advanceChunkId();
        logging::debug("%s: m_chunkPending = true  chunkId = %u", __PRETTY_FUNCTION__, m_fileParameters->chunkId());
        m_chunkPending = true;
        status = true;
    }
//  m_chunkPending_sem.give();
    return status;
}

std::string TebReceiverBase::reopenFiles()
{
    logging::debug("entered %s", __PRETTY_FUNCTION__);
    if (m_writing == false) {
        logging::error("%s: m_writing is false", __PRETTY_FUNCTION__);
        return std::string("reopenFiles: m_writing is false");
    }
    const std::string& outputDir = m_fileParameters->outputDir();
    const std::string& instrument = m_fileParameters->instrument();
    const std::string& experimentName = m_fileParameters->experimentName();
    unsigned           runNumber = m_fileParameters->runNumber();
    const std::string& hostname = m_fileParameters->hostname();

    std::string retVal{};               // return empty string on success
    m_chunkRequest = false;
    m_chunkOffset = m_offset;

    // close data file (for old chunk)
    logging::debug("%s: calling fileWriter.close()...", __PRETTY_FUNCTION__);
    fileWriter().close();

    // open data file (for new chunk)
    const std::string& runName = m_fileParameters->runName();
    std::string exptDir{outputDir + "/" + instrument + "/" + experimentName};
    local_mkdir(exptDir.c_str());
    std::string dataDir{exptDir + "/xtc"};
    local_mkdir(dataDir.c_str());
    std::string path{"/" + instrument + "/" + experimentName + "/xtc/" + runName + ".xtc2"};
    std::string absolute_path{outputDir + path};
    // cpo suggests leaving this print statement in because
    // filesystems can hang in ways we can't timeout/detect
    // and this print statement may speed up debugging significantly.
    std::cout << "Opening file " << absolute_path << std::endl;
    logging::info("%s: Opening file '%s'", __PRETTY_FUNCTION__, absolute_path.c_str());
    if (fileWriter().open(absolute_path) == 0) {
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

std::string TebReceiverBase::closeFiles()
{
    logging::debug("%s: m_writing is %s", __PRETTY_FUNCTION__, m_writing ? "true" : "false");
    if (m_writing) {
        m_writing = false;
        logging::debug("calling smdWriter.close()...");
        smdWriter().close();
        logging::debug("calling fileWriter.close()...");
        fileWriter().close();
    }
    return std::string{};
}

void TebReceiverBase::chunkRequestSet()
{
    m_chunkRequest = true;
}

void TebReceiverBase::chunkReset()
{
    // clean up the state left behind by a previous run
    m_chunkOffset = 0;
    m_chunkRequest = false;
//  m_chunkPending_sem = Pds::Semaphore::FULL;
    m_chunkPending = false;
}

void TebReceiverBase::resetCounters(bool all = false)
{
    EbCtrbInBase::resetCounters();

    if (all)  m_lastIndex = -1u;
    m_damage = 0;
    if (m_dmgType)  m_dmgType->clear();
    m_latency = 0;
}

void TebReceiverBase::process(const ResultDgram& result, unsigned index)
{
    bool error = false;
    if (index != ((m_lastIndex + 1) & (m_pool.nbuffers() - 1))) {
        logging::critical("%sTebReceiver: jumping index %u  previous index %u  diff %d%s",
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
        logging::critical("%spulseId %014lx, ts %u.%09u, tid %d, env %08x%s",
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
        logging::critical("lastIdx %8u, lastPid %014lx, lastTid %s, lastEnv %08x", m_lastIndex, m_lastPid, TransitionId::name(m_lastTid), m_lastEnv);
        abort();
    }

    m_lastIndex = index;
    m_lastPid = pulseId;
    m_lastTid = transitionId;
    m_lastEnv = dgram->env;

    // Transfer Result damage to the datagram
    dgram->xtc.damage.increase(result.xtc.damage.value());
    uint16_t damage = dgram->xtc.damage.value();
    if (damage) {
        m_damage++;
        while (damage) {
            unsigned dmgType = __builtin_ffsl(damage) - 1;
            damage &= ~(1 << dmgType);
            if (m_dmgType)  m_dmgType->observe(dmgType);
        }
    }

    // pass everything except L1 accepts and slow updates to control level
    if ((transitionId != TransitionId::L1Accept)) {
        if (transitionId != TransitionId::SlowUpdate) {
            if (transitionId == TransitionId::Configure) {
                // Cache Configure Dgram for writing out after files are opened
                Dgram* configDgram = dgram;
                size_t size = sizeof(*configDgram) + configDgram->xtc.sizeofPayload();
                m_configureIndex = index;
                memcpy(m_configureBuffer.data(), configDgram, size);
            }
            if (transitionId == TransitionId::BeginRun)
              m_offset = 0;// reset for monitoring (and not recording)
            // send pulseId to inproc so it gets forwarded to the collection
            json msg = createPulseIdMsg(pulseId);
            m_inprocSend.send(msg.dump());

            logging::info("TebRcvr    saw %s @ %u.%09u (%014lx)",
                           TransitionId::name(transitionId),
                          dgram->time.seconds(), dgram->time.nanoseconds(), pulseId);
        }
        else {
            logging::debug("TebRcvr    saw %s @ %u.%09u (%014lx)",
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

    // Complete processing and dispose of the event
    complete(index, result);
}


class PV : public Pds_Epics::PVBase
{
public:
    PV(const char* pvName) : PVBase(pvName), m_ready(getComplete(5)) {} // seconds
    virtual ~PV() {}
public:
    void updated()
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        getVectorAs<double>(m_vector);
    }
    bool ready()
    {
        return m_ready;
    }
    double value(unsigned element)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        return m_vector[element];
    }
private:
    std::mutex                       m_mutex;
    pvd::shared_vector<const double> m_vector;
    bool                             m_ready;
};

static bool _pvGetVecElem(const std::shared_ptr<PV> pv, unsigned element, double& value)
{
    if (!pv || !pv->ready() || !pv->connected()) {
        if (pv) {
            logging::critical("PV %s didn't connect", pv->name().c_str());
            abort();
        }
        return false;
    }

    value = pv->value(element);

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
    m_tebContributor = std::make_unique<TebContributor>(m_tPrms, pool.nbuffers());

    m_mPrms.instrument = para.instrument;
    m_mPrms.partition  = para.partition;
    m_mPrms.alias      = para.alias;
    m_tPrms.detName    = para.detName;
    m_tPrms.detSegment = para.detSegment;
    m_mPrms.maxEvSize  = pool.pebble.bufferSize();
    m_mPrms.maxTrSize  = pool.pebble.trBufSize();
    m_mPrms.verbose    = para.verbose;
    m_mPrms.kwargs     = para.kwargs;
    m_mebContributor = std::make_unique<MebContributor>(m_mPrms);

    m_inprocSend.connect("inproc://drp");

    if (para.outputDir.empty()) {
        logging::info("Output dir: n/a");
    } else {
        // Induce the automounter to mount in case user enables recording
        struct stat statBuf;
        std::string statPth{para.outputDir + "/" + para.instrument};
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
        logging::info("Setting env %s", envBuff);
        if (setenv("EPICS_PVA_ADDR_LIST",envBuff,1))
            perror("setenv pva_addr");
    }
}

void DrpBase::shutdown()
{
    // If connect() ran but the system didn't get into the Connected state,
    // there won't be a Disconnect transition, so disconnect() here
    disconnect();                       // Does no harm if already done

    m_tebContributor->shutdown();
    m_mebContributor->shutdown();
    m_tebReceiver->shutdown();
}

json DrpBase::connectionInfo(const std::string& ip)
{
    m_tPrms.ifAddr = ip;
    m_tPrms.port.clear();               // Use an ephemeral port

    int rc = m_tebReceiver->startConnection(m_tPrms.port);
    if (rc)  {
        logging::critical("Error starting TebReceiver connection");
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

    exporter->add("drp_trbufs_in_use", labels, Pds::MetricType::Gauge,
                  [&](){return pool.trInUse();});
    exporter->constant("drp_trbufs_in_use_max", labels, pool.pebble.nTrBuffers());

    exporter->addFloat("drp_deadtime", labels,
                       [&](double& value){return _pvGetVecElem(m_deadtimePv, m_xpmPort, value);});

    return 0;
}

std::string DrpBase::connect(const json& msg, size_t id)
{
    // Save a copy of the json so we can use it to connect to the config database on configure
    m_connectMsg = msg;
    m_collectionId = id;

    // Parse the connection parameters before they're used by the following stuff
    int rc = parseConnectionParams(msg["body"], id);
    if (rc) {
        return std::string{"Connection parameters error - see log"};
    }

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

    rc = m_tebReceiver->connect(m_exporter);
    if (rc) {
        return std::string{"TebReceiver connect failed"};
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

    rc = m_tebReceiver->EbCtrbInBase::configure(m_numTebBuffers);
    if (rc) {
        return std::string{"TebReceiver configure failed"};
    }

    printParams();

    // start eb receiver thread
    m_tebContributor->startup(*m_tebReceiver);

    // Same time as the TEBs and MEBs
    m_tebContributor->resetCounters();
    m_mebContributor->resetCounters();
    m_tebReceiver->resetCounters(true);
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
    // Check for monitoring only detectors - only check if run_number != 0 (i.e. recording)
    // If recording, set run_number back to 0 if a "monitoring only" detector
    // Setting back to run_number = 0 convinces this DRP it is not recording.
    if (run_number) {
        if (phase1Info.find("monitor_info") != phase1Info.end()) {
            std::string unique_id = m_para.detName + "_" + std::to_string(m_para.detSegment);
            for (auto it = phase1Info["monitor_info"].begin(); it != phase1Info["monitor_info"].end(); ++it) {
                if (it.key() == unique_id) {
                    if (it.value() == 1) {
                        logging::info("Detector %s selected for monitor only. No data will be recorded.",
                                      unique_id.c_str());
                        run_number = 0;
                    }
                }
            }
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
            msg = m_tebReceiver->openFiles(runInfo);
        }
    }

    // Same time as the TEBs and MEBs
    m_tebContributor->resetCounters();
    m_mebContributor->resetCounters();
    m_tebReceiver->resetCounters();
    m_tebReceiver->chunkReset();
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
    std::string retval{};

    logging::debug("%s: writing() is %s", __PRETTY_FUNCTION__, m_tebReceiver->writing() ? "true" : "false");
    chunkRequest = false;
    if (m_tebReceiver->writing()) {
        logging::debug("%s: chunkSize() = %lu", __PRETTY_FUNCTION__, m_tebReceiver->chunkSize());
        if (m_tebReceiver->chunkSize() > TebReceiverBase::DefaultChunkThresh / 2ull) {
            if (m_tebReceiver->advanceChunkId()) {
                logging::debug("%s: advanceChunkId() returned true", __PRETTY_FUNCTION__);
                // request new chunk after this Enable dgram is written
                chunkRequest = true;
                m_tebReceiver->chunkRequestSet();
                chunkInfo.filename = m_tebReceiver->fileParameters().runName() + ".xtc2";
                chunkInfo.chunkId = m_tebReceiver->fileParameters().chunkId();
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
    m_tebReceiver->unconfigure();
}

void DrpBase::disconnect()
{
    // If configure() ran but the system didn't get into the Configured state,
    // there won't be an Unconfigure transition, so unconfigure() here
    unconfigure();                      // Does no harm if already done

    m_tebContributor->disconnect();
    if (m_mPrms.addrs.size() != 0) {
        m_mebContributor->disconnect();
    }
    m_tebReceiver->disconnect();

    if (m_exporter) {
        m_exporter.reset();
    }
}

int DrpBase::setupTriggerPrimitives(const json& body)
{
    const std::string configAlias   = body["config_alias"];
    const std::string triggerConfig = body["trigger_config"];
    const json&       top           = body["trigger_body"];

    bool buildAll = (top.find("buildAll") != top.end()) && (top["buildAll"] == 1);

    std::string buildDets("---");
    if (top.find("buildDets") != top.end())
        buildDets = top["buildDets"];

    // In the following, _0 is added in prints to show the required segment number
    if (!(buildAll || buildDets.find(m_para.detName))) {
        logging::info("This DRP is not contributing trigger input data: "
                      "buildAll is False and '%s' was not found in ConfigDb %s/%s/%s_0",
                      m_para.detName.c_str(), m_para.instrument.c_str(), configAlias.c_str(), triggerConfig.c_str());
        m_tPrms.contractor = 0;    // This DRP won't provide trigger input data
        m_triggerPrimitive = nullptr;
        m_tPrms.maxInputSize = sizeof(Pds::EbDgram); // Revisit: 0 is bad
        return 0;
    }
    m_tPrms.contractor = m_tPrms.readoutGroup;

    if (top.find("soname") == top.end()) {
        logging::error("Key 'soname' was not found in configDb %s/%s/%s_0",
                       m_para.instrument.c_str(), configAlias.c_str(), triggerConfig.c_str());
        return -1;
    }
    std::string soname{top["soname"]};
    if (m_det.gpuDetector()) {
        auto found = soname.rfind('.');
        if (found == std::string::npos) {
            logging::error("Trigger library name is missing its extension: %s", soname.c_str());
            return -1;
        }
        soname = soname.substr(0, found) + "_gpu" + soname.substr(found, soname.size()-found);
    }

    //  Look for the detector-specific producer first
    std::string symbol("create_producer");
    std::string symbol2(symbol + "_" + m_para.detName);
    m_triggerPrimitive = m_trigPrimFactory.create(soname, symbol2);
    if (m_triggerPrimitive) {
        logging::info("Created detector-specific TriggerPrimitive [%s]", symbol2.c_str());
    }
    else {
        // Now try the generic producer
        m_triggerPrimitive = m_trigPrimFactory.create(soname, symbol);
        if (!m_triggerPrimitive) {
            logging::error("Failed to create TriggerPrimitive from '%s' with '%s' or '%s'; try '-v'",
                           soname.c_str(), symbol.c_str(), symbol2.c_str());
            return -1;
        }
        logging::info("Created generic TriggerPrimitive [%s]", symbol.c_str());
    }
    m_tPrms.maxInputSize = sizeof(Pds::EbDgram) + m_triggerPrimitive->size();

    if (m_triggerPrimitive->configure(body, m_connectMsg, m_collectionId)) {
        logging::error("TriggerPrimitive::configure() failed");
        return -1;
    }

    logging::info("Trigger configured from configDb %s/%s/%s_0 using %s",
                  m_para.instrument.c_str(), configAlias.c_str(), triggerConfig.c_str(),
                  soname.c_str());

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

    // Require a consistent Instrument name from  command line and connect_info
    std::string instrument = body["control"]["0"]["control_info"]["instrument"];
    if (instrument != m_para.instrument) {
        logging::error("Instrument name mismatch: connect_info '%s' vs '%s' from -P",
                       m_para.instrument.c_str(), instrument.c_str());
        return -1;
    }

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
        logging::error("Pebble buffer count (%u) must be <= the common RoG's (%u)",
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

    logging::info("");
    logging::info("Parameters of Contributor ID %d (%s:%s):",           m_tPrms.id,
                                                                        m_tPrms.ifAddr.c_str(), m_tPrms.port.c_str());
    logging::info("  Thread core numbers:          %d, %d",             m_tPrms.core[0], m_tPrms.core[1]);
    logging::info("  Instrument:                   %s",                 m_tPrms.instrument.c_str());
    logging::info("  Partition:                    %u",                 m_tPrms.partition);
    logging::info("  Alias (detName, detSeg):      %s ('%s', %u)",      m_tPrms.alias.c_str(), m_tPrms.detName.c_str(), m_tPrms.detSegment);
    logging::info("  Readout group receipient:     0x%02x",             m_tPrms.readoutGroup);
    logging::info("  Readout group contractor:     0x%02x",             m_tPrms.contractor);
    logging::info("  Bit list of TEBs:             0x%016lx, cnt: %zu", m_tPrms.builders,
                                                                        std::bitset<64>(m_tPrms.builders).count());
    logging::info("  Number of MEBs:               %zu",                m_mPrms.addrs.size());
    logging::info("  Batching state:               %s",                 m_tPrms.maxEntries > 1 ? "Enabled" : "Disabled");
    logging::info("  Batch duration:               0x%014x = %u ticks", m_tPrms.maxEntries, m_tPrms.maxEntries);
    logging::info("  Batch pool depth:             0x%08x = %u",        pool.nbuffers() / m_tPrms.maxEntries, pool.nbuffers() / m_tPrms.maxEntries);
    logging::info("  Max # of entries / batch:     0x%08x = %u",        m_tPrms.maxEntries, m_tPrms.maxEntries);
    logging::info("  # of TEB contrib.   buffers:  0x%08x = %u",        pool.nbuffers(), pool.nbuffers());
    logging::info("  # of TEB transition buffers:  0x%08x = %u",        TEB_TR_BUFFERS, TEB_TR_BUFFERS);
    logging::info("  Max  TEB contribution  size:  0x%08zx = %zu",      m_tPrms.maxInputSize, m_tPrms.maxInputSize);
    logging::info("  Max  MEB L1Accept      size:  0x%08zx = %zu",      m_mPrms.maxEvSize, m_mPrms.maxEvSize);
    logging::info("  Max  MEB transition    size:  0x%08zx = %zu",      m_mPrms.maxTrSize, m_mPrms.maxTrSize);
    for (unsigned i = 0; i < m_mPrms.maxEvents.size(); ++i)
      logging::info("  # of MEB %u contrib. buffers:  0x%08x = %u",      i, m_mPrms.maxEvents[i], m_mPrms.maxEvents[i]);
    logging::info("  # of MEB transition buffers:  0x%08x = %u",        MEB_TR_BUFFERS, MEB_TR_BUFFERS);
    logging::info("");
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
