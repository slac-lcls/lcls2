#include "PGPDetector.hh"

#include "Detector.hh"
#include "Reader.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psdaq/eb/ResultDgram.hh"
#include "drp/DrpBase.hh"
#include "xtcdata/xtc/Smd.hh"

#include <cuda_runtime.h>

#include <sys/prctl.h>

using namespace XtcData;
using namespace Pds;
using namespace Drp;
using namespace Drp::Gpu;
using namespace Pds::Eb;
using json    = nlohmann::json;
using logging = psalg::SysLog;
using us_t    = std::chrono::microseconds;

struct drp_domain{ static constexpr char const* name{"PGPDrp"}; };
using drp_scoped_range = nvtx3::scoped_range_in<drp_domain>;
struct tr_domain{ static constexpr char const* name{"TebReceiver"}; };
using tr_scoped_range = nvtx3::scoped_range_in<tr_domain>;


TebReceiver::TebReceiver(const Parameters&        para,
                         DrpBase&                 drp,
                         const std::atomic<bool>& terminate) :
  TebReceiverBase(para, drp),
  m_mon          (drp.mebContributor()),
  m_terminate    (terminate),
  m_stream       (0),
  m_worker       (0),
  m_recordQueue  (drp.pool.nbuffers()),
  m_para         (para)
{
}

TebReceiver::~TebReceiver()
{
  printf("*** TebRcvr::dtor: 1\n");

  if (m_stream) {
    chkError(cudaStreamDestroy(m_stream));
  }

  printf("*** TebRcvr::dtor: 2\n");
  teardown();

  printf("*** TebRcvr::dtor: 3\n");
}

//bool     lDisable{false};
//uint64_t lReduceStarts{0};
//uint64_t lReduceReceives{0};
//unsigned lLastIndex{0};
unsigned lState{0};

int TebReceiver::setupMetrics(const std::shared_ptr<MetricExporter> exporter,
                              std::map<std::string, std::string>&   labels)
{
  exporter->add("DRP_smdWriting",  labels, MetricType::Gauge, [&](){ return m_smdWriter ? m_smdWriter->writing() : 0; });
  exporter->add("DRP_fileWriting", labels, MetricType::Gauge, [&](){ return m_fileWriter ? m_fileWriter->writing() : 0; });
  exporter->add("DRP_recordQueue", labels, MetricType::Gauge, [&](){ return m_recordQueue.guess_size(); });
  exporter->add("DRP_recordState", labels, MetricType::Gauge, [&](){ return lState; });

  return 0;
}

void TebReceiver::setup(cudaExecutionContext_t green_ctx)
{
  printf("*** TebRcvr::setup: 1\n");
  m_worker = 0;

  auto& memPool = *m_pool.getAs<MemPoolGpu>();

  // Set up the file writers
  // NB: this fails when done in _recorder() due to cuFileDriverOpen() hanging
  auto bufSize = memPool.reduceBufsSize() + memPool.reduceBufsReserved();
  size_t maxBufSize = 32 * 1024 * 1024UL; // Max pinned memory size
  m_fileWriter = nullptr; //std::make_unique<FileWriter>(maxBufSize, true);
  //m_fileWriter = std::make_unique<FileWriterAsync>(maxBufSize/2, true); // For 2 ping pong buffers
  printf("*** TebRcvr::setup: 2\n");
  m_smdWriter  = std::make_unique<SmdWriter>(bufSize, m_para.maxTrSize);
  printf("*** TebRcvr::setup: 3\n");

  // Reset the record queue
  m_recordQueue.startup();

  // Start the Data Recorder
  m_recorderThread = std::thread(&TebReceiver::_recorder, this, green_ctx);
}

void TebReceiver::teardown()
{
  // Unblock the record queue
  m_recordQueue.shutdown();

  if (m_recorderThread.joinable()) {
    m_recorderThread.join();
    logging::info("Recorder thread finished");
  }
}

void TebReceiver::_writeDgram(Dgram* dgram, void* devPtr)
{
  //uint32_t* p = (uint32_t*)dgram;
  //printf("wDg: ");
  //for (unsigned i = 0; i < 18; ++i)  printf("%08x ", p[i]);
  //printf("\n");

  size_t size = sizeof(*dgram) + dgram->xtc.sizeofPayload();
  if (m_fileWriter)  m_fileWriter->writeEvent(devPtr, size, dgram->time);

  // small data writing
  Smd smd;
  const void* bufEnd = m_smdWriter->buffer.data() + m_smdWriter->buffer.size();
  NamesId namesId(dgram->xtc.src.value(), NamesIndex::OFFSETINFO);
  Dgram* smdDgram = smd.generate(dgram, m_smdWriter->buffer.data(), bufEnd, chunkSize(), size,
                                 m_smdWriter->namesLookup, namesId);
  m_smdWriter->writeEvent(smdDgram, sizeof(Dgram) + smdDgram->xtc.sizeofPayload(), smdDgram->time);
  offsetAppend(size);
}

/** This function is called by the base class's process() method to complete
 *  processing and dispose of the event.  It presumes that the caller has
 *  already vetted index and result
 */

void TebReceiver::complete(unsigned index, const ResultDgram& result)
{
  tr_scoped_range r{/*"TebReceiver::complete", */nvtx3::payload{index}}; // Expose function name via NVTX

  // Set the nworkers to 0 to run without the Reducer workers in the loop
  if (m_para.nworkers == 0) {
    *(uint32_t*)const_cast<ResultDgram&>(result).xtc.payload() = 0;
  }

  logging::debug("TebRcvr::complete: Posting  %s, pid %014lx, prescale %d, persist %d, monitor %d to Recorder",
                 TransitionId::name(result.service()), result.pulseId(), result.prescale(), result.persist(), result.monitor());

//  if (result.service() == TransitionId::Disable) {
//    printf("*** TebRcvr::complete: Disable seen, recordQueue occ %d, reduceStarts %lu, reduceReceives %lu, lastIdx %u, idx %u, state %u\n",
//           m_recordQueue.guess_size(), lReduceStarts, lReduceReceives, lLastIndex, index, lState);
//    static_cast<PGPDrp&>(m_drp).reducerDump();
//    lDisable = true;
//    const_cast<Parameters*>(&m_para)->verbose++;
//  }

  // Pass parameters to the recorder thread
  m_recordQueue.push({index, &result});

  // Start up a reducer only when there is a need for its result
  // Running the reducer on transitions is a no-op, so avoid its overhead
  if (result.persist() || result.monitor()) {
    nvtx3::mark("Reducer start", nvtx3::payload{m_worker});
    //printf("*** TebRcvr::complete: wkr %u, idx %u\n", m_worker, index);
    //bool wait{false};
    while (!static_cast<PGPDrp&>(m_drp).reducerStart(m_worker, index)) {
      if (m_terminate.load(std::memory_order_acquire)) [[unlikely]] {
        printf("*** TebRcvr::complete: inputQueue full\n");
        return;                         // @todo: Revisit
      }
      //if (!wait) {
      //  wait = true;
      //  printf("*** TebRcvr::complete: wait T, next %d, tail %d\n", next, tail);
      //}
    }
    //if (wait)
    //  printf("*** TebRcvr::complete: wait F, next %d, tail %d\n", next, tail);
    m_worker = (m_worker + 1) % m_para.nworkers;
//    ++lReduceStarts;
//    lLastIndex = index;
  }
}

void TebReceiver::_recorder(cudaExecutionContext_t green_ctx)
{
  tr_scoped_range r{/*"TebReceiver::_recorder"*/}; // Expose function name via NVTX

  logging::info("Recorder is starting with process ID %lu\n", syscall(SYS_gettid));
  if (prctl(PR_SET_NAME, "drp_gpu/Recorder", 0, 0, 0) == -1) {
    perror("prctl");
  }

  // Establish context in this thread
  auto& memPool = *m_pool.getAs<MemPoolGpu>();
  chkError(cudaSetDevice(memPool.context().deviceNo()));
  chkError(cuCtxSetCurrent(memPool.context().context()));

  // Get the range of priorities available [ greatest_priority, lowest_priority ]
  int prioLo;
  int prioHi;
  chkError(cudaDeviceGetStreamPriorityRange(&prioLo, &prioHi));
  int prio{prioHi};
  logging::debug("Recorder stream priority (range: LOW: %d to HIGH: %d): %d", prioLo, prioHi, prio);

  // Create a GPU stream in the recorder thread context and register it with the
  // fileWriter during phase 1 of Configure before files are opened during BeginRun
  // The highest priority is to dispose of the data
  chkFatal(cudaStreamCreateWithPriority(&m_stream, cudaStreamNonBlocking, prio));
  //chkFatal(cudaExecutionCtxStreamCreate(&m_stream, green_ctx, cudaStreamDefault, prio));
  if (m_fileWriter)  m_fileWriter->registerStream(m_stream);

  auto maxSize = memPool.reduceBufsReserved() + memPool.reduceBufsSize();
  //printf("*** TebRcvr::recorder: redBufsSz %zu + rsvdSz %zu = maxSize %zu\n", memPool.reduceBufsSize(), memPool.reduceBufsReserved(), maxSize);

  auto& drp = static_cast<PGPDrp&>(m_drp);

  // Collect completion information from the reducer kernels in time order
  unsigned worker = 0;
  while (!m_terminate.load(std::memory_order_acquire)) {
    lState = 1;
    // Wait for a new Result to appear from the TEB via the complete() method above
    ResultTuple items;
    if (!m_recordQueue.pop(items)) {
      printf("*** TebRcvr::recorder: No item from recordQueue\n");
      continue;
    }
    lState = 2;
    const auto index  = std::get<0>(items);
    const auto result = std::get<1>(items);
    nvtx3::mark("Recorder", nvtx3::payload{index});
//    if (lDisable) {
//      printf("*** TebRcvr::recorder: Disable T, idx %u, result->svc %u\n", index, result->service());
//    }
    logging::debug("TebRcvr::recorder: Handling %s, pid %014lx, prescale %d, persist %d, monitor %d",
                   TransitionId::name(result->service()), result->pulseId(), result->prescale(), result->persist(), result->monitor());

    // If needed, wait for the next GPU Reducer in sequence to complete
    size_t dataSize;
    if (result->persist() || result->monitor()) {
      nvtx3::mark("Recorder reducerReceive", nvtx3::payload{worker});
      lState = 3;
      ReducerTuple rt;
      //bool wait{false};
      while (!drp.reducerReceive(worker, &rt)) { // Block here until a result is ready from GPU
        if (m_terminate.load(std::memory_order_acquire)) [[unlikely]] {
          printf("*** TebRcvr::recorder: outputQueue empty\n");
          return;                       // @todo: Revisit
        }
        //if (!wait) {
        //  wait = true;
        //  printf("*** TebRcvr::recorder: wait T, tail %d, head %d\n", tail, head);
        //}
      }
      //if (wait)
      //  printf("*** TebRcvr::recorder: wait F, tail %d, head %d\n", tail, head);
      lState = 4;
      //printf("*** TebRcvr::recorder: wkr %u, rt idx %u, sz %zu\n", worker, rt.index, rt.dataSize);
      worker = (worker + 1) % m_para.nworkers;
//      ++lReduceReceives;

      if (rt.index != index) [[unlikely]] { // Sanity check
        logging::critical("Recorder vs Reducer index mismatch: %u vs %u", index, rt.index);
        //abort();
      }
      dataSize = rt.dataSize;
    }
    lState = 5;

//    if (lDisable) {
//      printf("*** TebRcvr::recorder: reduceReceives %lu\n", lReduceReceives);
//      drp.reducerDump();
//      printf("*** TebRcvr::recorder: 1 idx %u, dataSize %zu\n", index, dataSize);
//      printf("*** TebRcvr::recorder: 1 pid %014lx, svc %u, prescale %d, persist %d, monitor %d\n",
//             result->pulseId(), result->service(), result->prescale(), result->persist(), result->monitor());
//    }

    // Release the GPU intermediate buffers for reuse
    drp.freeBuffers(index);             // Leaves event mask = 0
    nvtx3::mark("Recorder freeBufs()", nvtx3::payload{index});
    //printf("*** TebRcvr::recorder: 2, freeBufs idx %u\n", index);
    lState = 6;

    // Look up the datagram, whether transition or L1Accept
    auto dgram = result->isEvent() ? (EbDgram*)m_pool.pebble[index] : m_pool.transitionDgrams[index];
    auto pulseId = dgram->pulseId();
    if (pulseId != result->pulseId()) { // Sanity check
      logging::critical("Pulse IDs differ: idx %u, %014lx, %014lx\n",
                        index, pulseId, result->pulseId());
      abort();
    }
    //printf("*** TebRcvr::recorder: 2 dg[%u] pid %014lx, env %08x\n", index, pulseId, dgram->env);

    // pass everything except L1 accepts and slow updates to control level
    TransitionId::Value transitionId = dgram->service();
    if (transitionId != TransitionId::L1Accept) {
      if (transitionId != TransitionId::SlowUpdate) {
        // send pulseId to inproc so it gets forwarded to the collection
        json msg = createPulseIdMsg(pulseId);
        m_inprocSend.send(msg.dump());

        logging::info("Recorder   saw %12s @ %u.%09u (%014lx)",
                      TransitionId::name(transitionId),
                      dgram->time.seconds(), dgram->time.nanoseconds(),
                      pulseId);
      }
      else {
        logging::debug("Recorder   saw %12s @ %u.%09u (%014lx)",
                       TransitionId::name(transitionId),
                       dgram->time.seconds(), dgram->time.nanoseconds(),
                       pulseId);
      }
    }
    lState = 7;

    // Find the location of where the Xtc payload is on the GPU or where it will go for transitions
    auto buffer = &memPool.reduceBuffers_d()[index * maxSize];
    //printf("*** TebRcvr::recorder: 3 idx %u, buf %p, maxSize %zu\n", index, buffer, maxSize);
    size_t cpSize, dgSize;
    if (dgram->isEvent() && (result->persist() || result->monitor())) {
      // dgram must fit in the GPU's reduce buffer, so _not_ pebble bufferSize() here
      void* bufEnd = (char*)((Dgram*)dgram) + maxSize;
      //printf("*** TebRcvr::recorder: 3 dg %p + %zu = bufEnd %p\n", (Dgram*)dgram, maxSize, bufEnd);
      drp.reducerEvent(dgram->xtc, bufEnd, dataSize);

      // Measure the size of the header block
      auto headerSize = (uint8_t*)dgram->xtc.next() - (uint8_t*)((Dgram*)dgram) - dataSize;
      //printf("*** TebRcvr::recorder: 3 payloadSz %u, length %p - %p - %zu = %zd\n",
      //       dgram->xtc.sizeofPayload(), dgram->xtc.next(), (Dgram*)dgram, dataSize, headerSize);

      // Make sure the header will fit in the space reserved for it on the GPU
      if (size_t(headerSize) > memPool.reduceBufsReserved()) {
        printf("*** TebRcvr::recorder: 3 payloadSz %u, length %p - %p - %zu = %zd\n",
               dgram->xtc.sizeofPayload(), dgram->xtc.next(), (Dgram*)dgram, dataSize, headerSize);
        logging::critical("Header is too large (%zu) for reduce buffer's reserved space (%zu)",
                          headerSize, memPool.reduceBufsReserved());
        abort();
      }
      // Make sure the header has fit into the pebble buffer on the CPU
      if (size_t(headerSize) > memPool.pebble.bufferSize()) {
        logging::critical("Header is too large (%zu) for pebble buffer (%zu)",
                          headerSize, memPool.pebble.bufferSize());
        abort();
      }

      cpSize  = headerSize;
      buffer -= headerSize;             // Points to the start of the Dgram
      dgSize  = sizeof(Dgram) + dgram->xtc.sizeofPayload(); // Not *dgram, or get sizeof(EbDgram)!
    } else {  // Transitions
      cpSize  = sizeof(Dgram) + dgram->xtc.sizeofPayload(); // Not *dgram, or get sizeof(EbDgram)!
      buffer -= sizeof(Dgram);          // Points to the start of the Dgram
      dgSize  = cpSize;
    }
    lState = 8;

    //printf("*** TebRcvr::recorder: 3 idx %u, buf %p, tr %u, cpSz %zu, extent %u, dgSz %zu\n", index, buffer, dgram->service(), cpSize, dgram->xtc.extent, dgSize);
    if (dgSize > maxSize) {
      logging::critical("Datagram is too large (%zu) for reduce buffer (%zu) [pid %014lx, ts %016lx, env %08x]",
                        dgSize, maxSize, pulseId, dgram->time.value(), dgram->env);
      abort();
    }

    //uint32_t* p = (uint32_t*)((Dgram*)dgram);
    //printf("all: ");
    //for (unsigned i = 0; i < 18; ++i)  printf("%08x ", p[i]);
    //printf("\n");

    if (writing() || transitionId == TransitionId::Configure) {
      // Copy the dgram header to the GPU if it's an L1Accept or the whole datagram when it's a transition
      chkError(cudaMemcpyAsync(buffer, (void*)((Dgram*)dgram), cpSize, cudaMemcpyHostToDevice, m_stream));
      //if (dgram->isEvent()) {
      //  const Xtc& parent = dgram->xtc;
      //  const Xtc& shapesData = (Xtc&)*parent.payload();
      //  auto p = (uint32_t*)&shapesData;
      //  printf("*** 1st: %p: %08x %08x %08x\n", p, p[0], p[1], p[2]);
      //  const Xtc& data = (Xtc&)*shapesData.payload();
      //  p = (uint32_t*)&data;
      //  printf("*** pld: %p: %08x %08x %08x %08x %08x %08x %08x %08x\n", p, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
      //  const Xtc& shapes = *data.next();
      //  p = (uint32_t*)&shapes;
      //  printf("*** 2nd: %p: %08x %08x %08x\n", p, p[0], p[1], p[2]);
      //  unsigned sz = sizeof(shapes) + shapes.sizeofPayload();
      //  printf("*** shapes size %u, data size %u, total size %zu\n", sz, shapes.sizeofPayload(), (uint8_t*)shapes.next() - (uint8_t*)((Dgram*)dgram));
      //}
    }

    if (writing()) {                  // Won't ever be true for Configure
      //printf("*** TebRcvr::recorder: writing %zu bytes\n", dgSize);
      // write event to file if it passes event builder or if it's a transition
      if (result->persist() || result->prescale()) {
        //printf("*** TebRcvr::recorder: persist or prescale\n");
        //uint32_t* p = (uint32_t*)dgram;
        //printf("l1:  ");
        //for (unsigned i = 0; i < 18; ++i)  printf("%08x ", p[i]);
        //printf("\n");
        _writeDgram(dgram, buffer);     // Only (some) L1Accepts written here
      }
      else if (transitionId != TransitionId::L1Accept) {
        //printf("*** TebRcvr::recorder: transitionId %u\n", transitionId);
        if (transitionId == TransitionId::BeginRun) {
          offsetReset(); // reset offset when writing out a new file
          //printf("*** TebRcvr::recorder: BeginRun 1\n");
          auto cfgDgram = reinterpret_cast<Dgram*>(m_configureBuffer.data());
          //printf("*** TebRcvr::recorder: BeginRun 2 cfgDg %p\n", cfgDgram);
          auto cfgSize  = sizeof(*cfgDgram) + cfgDgram->xtc.sizeofPayload();
          //printf("*** TebRcvr::recorder: BeginRun 3 cfgSz %zu\n", cfgSize);
          auto cfgBuf   = &memPool.reduceBuffers_d()[m_configureIndex * maxSize] - sizeof(Dgram);
          //printf("*** TebRcvr::recorder: BeginRun 4 cfgBuf %p\n", cfgBuf);
          if (cfgSize > maxSize) {
            logging::critical("Configure dgram (%zu) is too big for GPU's buffer (%zu)",
                              cfgSize, maxSize);
            abort();
          }
          //printf("*** TebRcvr::recorder: 4a idx %u, cfgBuf %p, cfgDg %p, sz %zu\n", m_configureIndex, cfgBuf, cfgDgram, cfgSize);
          chkError(cudaMemcpyAsync((void*)cfgBuf, cfgDgram, cfgSize, cudaMemcpyHostToDevice, m_stream));
          //printf("*** TebRcvr::recorder: 4b idx %u\n", m_configureIndex);
          //uint32_t* p = (uint32_t*)cfgDgram;
          //printf("cfg: ");
          //for (unsigned i = 0; i < 18; ++i)  printf("%08x ", p[i]);
          //printf("\n");
          _writeDgram(cfgDgram, cfgBuf);
        }
        //uint32_t* p = (uint32_t*)dgram;
        //printf("%02d:  ", transitionId);
        //for (unsigned i = 0; i < 18; ++i)  printf("%08x ", p[i]);
        //printf("\n");
        _writeDgram(dgram, buffer);
        if ((transitionId == TransitionId::Enable) && m_chunkRequest) {
          logging::debug("%s calling reopenFiles()", __PRETTY_FUNCTION__);
          reopenFiles();
        } else if (transitionId == TransitionId::EndRun) {
          logging::debug("%s calling closeFiles()", __PRETTY_FUNCTION__);
          closeFiles();
        }
      }
    }
    //printf("*** TebRcvr::recorder: 5 sz %zu, writing %d\n", dgSize, writing());

    // Measure latency before sending dgram for monitoring
    if (pulseId - m_latPid > 1300000/14) { // 10 Hz
      m_latency = Eb::latency<us_t>(dgram->time);
      m_latPid = pulseId;
    }
    //printf("*** TebRcvr::recorder: 6 latency %ld us\n", m_latency);
    lState = 9;

    m_evtSize = dgSize;

    if (m_mon.enabled()) {
      if (result->isEvent()) {          // L1Accept
        if (result->monitor()) {
          // Fetch the reduced data from the GPU and construct the dgram to send to the MEB
          auto payload       = dgram->xtc.payload();
          auto sizeofPayload = dgram->xtc.sizeofPayload();
          const auto data    = &memPool.reduceBuffers_d()[index * maxSize];
          chkError(cudaMemcpyAsync((void*)payload, data, sizeofPayload, cudaMemcpyDeviceToHost, m_stream));
          chkError(cudaStreamSynchronize(m_stream)); // Ensure payload is on CPU before posting

          m_mon.post(dgram, result->monBufNo());
        }
      } else {                          // Other Transition already on the CPU
        m_mon.post(dgram);
      }
    }
    //printf("*** TebRcvr::recorder: 7, mon %d\n", m_mon.enabled());
    lState = 10;

    // Synchronize before releasing buffers
    //chkError(cudaStreamSynchronize(m_stream)); // @todo: Needed???

    // Free the transition datagram buffer
    if (!dgram->isEvent()) {
        m_pool.freeTr(dgram);
        //printf("*** TebRcvr::recorder: 8, freeTr %014lx\n", pulseId);
    }
    lState = 11;

    // Free the pebble datagram buffer
    m_pool.freePebble(index);
    //printf("*** TebRcvr::recorder: 8, freePebble %d\n", index);
    lState = 12;

    //// Free the reduce buffer
    //m_reducer->release(index);
    //printf("*** TebRcvr::recorder: 8, released reduce buffer %u\n", index);
    nvtx3::mark("Pebble released", nvtx3::payload{index});
  }

  logging::info("Recorder thread is exiting");
}


PGPDrp::PGPDrp(Parameters&    parameters,
               MemPoolGpu&    memPool,
               Gpu::Detector& detector,
               ZmqContext&    zmqContext) :
  DrpBase      (parameters, memPool, detector, zmqContext),
  m_para       (parameters),
  m_det        (detector),
  m_green_ctx  {{}, {}, {}},
  m_terminate  (false),
  m_terminate_d(nullptr),
  m_nNoTrDgrams(0)
{
  if (pool.setMaskBytes(m_para.laneMask, m_det.virtChan)) {
    logging::critical("Failed to allocate lane/vc: '%m' "
                      "- does another process have %s:%02x open?",
                      m_para.device.c_str(), m_para.laneMask);
    abort();
  }

  // Partition the GPU for the various kernels to be run
//  _setupGreenContexts(memPool);

  // Set up thread termination flags
  chkError(cudaMalloc(&m_terminate_d,    sizeof(*m_terminate_d)));
  chkError(cudaMemset( m_terminate_d, 0, sizeof(*m_terminate_d)));

  // Set the TebReceiver we will use in the base class
  setTebReceiver(std::make_unique<TebReceiver>(m_para, *this, m_terminate));
}

PGPDrp::~PGPDrp()
{
  printf("*** PGPDrp::dtor: 1\n");
  chkError(cudaFree(m_terminate_d));
  printf("*** PGPDrp::dtor: 2\n");
}

void PGPDrp::_setupGreenContexts(MemPoolGpu& memPool)
{
  auto gpu_device_index{memPool.context().deviceNo()};
  chkError(cudaSetDevice(gpu_device_index));

  // Get all available GPU SM resources
  cudaDevResource initial_SM_resources = {};
  chkError(cudaDeviceGetDevResource(gpu_device_index,        // GPU device
                                    &initial_SM_resources,   // Device resource to populate
                                    cudaDevResourceTypeSm)); // Resource type

  logging::info("Initial SM resources: %d SMs", initial_SM_resources.sm.smCount); // number of available SMs

  // Special fields relevant for partitioning
  logging::info("  - Min. SM partition size: %d SMs", initial_SM_resources.sm.minSmPartitionSize);
  logging::info("  - SM co-scheduled alignment: %d SMs", initial_SM_resources.sm.smCoscheduledAlignment);

  //// Get all available GPU Work Queue resources
  //cudaDevResource initial_WQ_config_resources = {};
  //chkError(cudaDeviceGetDevResource(gpu_device_index,                     // GPU device
  //                                  &initial_WQ_config_resources,         // Device resource to populate
  //                                  cudaDevResourceTypeWorkqueueConfig)); // Resource type
  //
  //logging::info("Initial WQ config. resources: ");
  //logging::info("  - WQ concurrency limit: %d", initial_WQ_config_resources.wqConfig.wqConcurrencyLimit);
  //logging::info("  - WQ sharing scope: %d", initial_WQ_config_resources.wqConfig.sharingScope);

  // Split SM resources
  constexpr int nbGroups {2};
  constexpr unsigned default_split_flags {0};
  cudaDevResource remainder {};
  cudaDevResource result[nbGroups] {{}, {}};
  cudaDevSmResourceGroupParams group_params[2] =  {
    {.smCount=4,  .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0},
    {.smCount=40, .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0}};
  chkError(cudaDevSmResourceSplit(&result[0], nbGroups, &initial_SM_resources, &remainder, default_split_flags, &group_params[0]));

  // Generate resource descriptors for each resource
  cudaDevResourceDesc_t resource_desc[nbGroups+1] {{}, {}, {}};
  chkError(cudaDevResourceGenerateDesc(&resource_desc[0], &result[0], 1));
  chkError(cudaDevResourceGenerateDesc(&resource_desc[1], &result[1], 1));
  chkError(cudaDevResourceGenerateDesc(&resource_desc[2], &remainder, 1));

  // Create green contexts
  chkError(cudaGreenCtxCreate(&m_green_ctx[0], resource_desc[0], gpu_device_index, 0));
  chkError(cudaGreenCtxCreate(&m_green_ctx[1], resource_desc[1], gpu_device_index, 0));
  chkError(cudaGreenCtxCreate(&m_green_ctx[2], resource_desc[2], gpu_device_index, 0));

//  CUcontext green_primary;
//  cuCtxFromGreenCtx(&green_primary, m_green_ctx[0]);
//  cuCtxSetCurrent(green_primary);
//
//  cudaStream_t stream;
//  cudaStreamCreate(&stream);

  // Get all available GPU SM resources
  for (unsigned i = 0; i < nbGroups+1; ++i) {
    cudaDevResource final_SM_resources = {};
    chkError(cudaExecutionCtxGetDevResource(m_green_ctx[i],          // GPU context
                                            &final_SM_resources,     // Device resource to populate
                                            cudaDevResourceTypeSm)); // Resource type

    logging::info("Final SM resources for context %u: %d SMs", i, final_SM_resources.sm.smCount); // number of available SMs

    // Special fields relevant for partitioning
    logging::info("  - Min. SM partition size: %d SMs", final_SM_resources.sm.minSmPartitionSize);
    logging::info("  - SM co-scheduled alignment: %d SMs", final_SM_resources.sm.smCoscheduledAlignment);
  }
}

std::string PGPDrp::configure(const json& msg)
{
  std::string errorMsg = DrpBase::configure(msg);
  if (!errorMsg.empty()) {
    return errorMsg;
  }

  m_terminate.store(false, std::memory_order_release);
  chkError(cudaMemset(m_terminate_d, 0, sizeof(*m_terminate_d)));

  // Set up the communication queues between the various stages
  auto& memPool = *pool.getAs<MemPoolGpu>();
  auto trgPrimitive = triggerPrimitive();
  //printf("*** PGPDrp: trgPrm %p, sz %zu\n", trgPrimitive, sizeof(*trgPrimitive));
  //printf("*** PGPDrp: trgPrm->size %zu\n", trgPrimitive->size());
  auto tpSz = trgPrimitive ? trgPrimitive->size() : 0;

  // Set up a Reader to receive DMAed data and calibrate it
  m_reader = std::make_shared<Reader>(m_para, memPool, m_det, tpSz, m_green_ctx[0], *m_terminate_d);

  // Create the event building collector, which calculates the TEB input data
  // The TriggerPrimitive object in det is dynamically loaded to pick up the
  // TEB input data creation algorithm, e.g., peak finder
  m_collector = std::make_unique<Collector>(m_para, memPool, m_reader, trgPrimitive, m_green_ctx[0], m_terminate, *m_terminate_d);

  // Create the data reducer
  // The data reduction object is dynamically loaded to pick up the
  // problem-specific reduction algorithm, e.g., SZ, angular integration, etc.
  m_reducer = std::make_unique<Reducer>(m_para, memPool, m_det, m_green_ctx[1], m_terminate, *m_terminate_d);

  // Set up the TebReceiver
  static_cast<TebReceiver&>(tebReceiver()).setup(m_green_ctx[2]);

  // Start the Reducers
  m_reducer->startup();

  // Launch the Collector thread
  m_collectorThread = std::thread(&PGPDrp::_collector, std::ref(*this));

  return std::string{};
}

unsigned PGPDrp::unconfigure()
{
  logging::info("PGPDrp::unconfigure: Shutting down");

  DrpBase::unconfigure(); // TebContributor must be shut down before the reader
  printf("*** PGPDrp::unconfigure 1\n");

  // @todo: Right place for this?
  m_terminate.store(true, std::memory_order_release);
  chkError(cudaMemset(m_terminate_d, 1, sizeof(unsigned)));
  printf("*** PGPDrp::unconfigure 2\n");

  if (m_reducer)  m_reducer->shutdown();
  static_cast<TebReceiver&>(tebReceiver()).teardown();
  printf("*** PGPDrp::unconfigure 3\n");

  if (m_collectorThread.joinable()) {
    m_collectorThread.join();
    logging::info("Collector thread finished");
  }
  printf("*** PGPDrp::unconfigure 4\n");

  m_reducer.reset();
  printf("*** PGPDrp::unconfigure 5\n");
  m_collector.reset();
  printf("*** PGPDrp::unconfigure 6\n");
  m_reader.reset();
  printf("*** PGPDrp::unconfigure 7\n");

  return 0;
}

int PGPDrp::_setupMetrics(const std::shared_ptr<MetricExporter> exporter)
{
  auto& memPool = *pool.getAs<MemPoolGpu>();

  std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                            {"partition", std::to_string(m_para.partition)},
                                            {"detname", m_para.detName},
                                            {"alias", m_para.alias}};
  //auto queueLength = [](std::vector<SPSCQueue<Batch> >& vec)
  //    { size_t sum = 0;  for (auto& q: vec) sum += q.guess_size();  return sum; };
  //uint64_t nbuffers = memPool.nbuffers();
  //exporter->constant("drp_worker_queue_depth", labels, nbuffers);
  //
  //exporter->add("drp_worker_output_queue", labels, MetricType::Gauge,
  //              [&](){return queueLength(m_workerQueues);});
  m_nNoTrDgrams = 0;
  exporter->add("drp_num_no_tr_dgram", labels, MetricType::Gauge,
                [&](){return m_nNoTrDgrams;});

  exporter->constant("drp_num_pgp_bufs", labels, pool.dmaCount());
  exporter->add("drp_num_pgp_in_user",  labels, MetricType::Gauge,
                [&](){return memPool.nPgpInUser();});
  exporter->add("drp_num_pgp_in_hw",    labels, MetricType::Gauge,
                [&](){return memPool.nPgpInHw();});
  exporter->add("drp_num_pgp_in_prehw", labels, MetricType::Gauge,
                [&](){return memPool.nPgpInPreHw();});
  exporter->add("drp_num_pgp_in_rx",    labels, MetricType::Gauge,
                [&](){return memPool.nPgpInRx();});

  m_reader->setupMetrics(exporter, labels);

  m_collector->setupMetrics(exporter, labels);

  m_reducer->setupMetrics(exporter, labels);

  return 0;
}

void PGPDrp::freeBuffers(unsigned index)
{
  const auto timingHeader = m_det.getTimingHeader(index);
  const auto bufferMask = pool.nbuffers() - 1;
  const auto pgpIndex = timingHeader->evtCounter & bufferMask;
  auto event = &pool.pgpEvents[pgpIndex];
  //printf("*** PGPDrp::freeBuffers: bufIndex, %u, pgpIndex %u, event %p\n", index, pgpIndex, event);
  m_collector->freeDma(event);
}

void PGPDrp::_collector()
{
  drp_scoped_range r{/*"PGPDrp::_collector"*/}; // Expose function name via NVTX

  logging::info("Collector is starting with process ID %lu\n", syscall(SYS_gettid));
  if (prctl(PR_SET_NAME, "drp_gpu/Collector", 0, 0, 0) == -1) {
    perror("prctl");
  }

  pool.resetCounters();                 // Avoid jumps in TebReceiver

  // Set up monitoring
  auto exporter = std::make_shared<MetricExporter>();
  if (exposer()) {
    exposer()->RegisterCollectable(exporter);

    if (_setupMetrics(exporter))
      logging::error("PGPDrp::_collector: setupMetrics failed");
  }

  // Establish context in this thread
//  auto& memPool = *pool.getAs<MemPoolGpu>();
//  chkError(cudaSetDevice(memPool.context().deviceNo()));
//  chkError(cuCtxSetCurrent(memPool.context().context()));

  // Start the PGP reader on the GPU
  m_reader->start();

  // Start the Collector on the GPU
  m_collector->start();

  // Now run the CPU side of the Collector
  auto trgPrimitive = triggerPrimitive();
  const uint32_t bufferMask = pool.nbuffers() - 1;
  uint64_t lastPid = 0;
  unsigned bufIndex = 0;                // Intermediate buffer index
  while (true) {
    drp_scoped_range loop_range{nvtx3::category{0}, nvtx3::payload{bufIndex}};
    if (m_terminate.load(std::memory_order_relaxed)) {
      break;
    }

    auto nRet = m_collector->receive(&m_det); // This can block

    for (unsigned b = 0; b < nRet; ++b) {
      drp_scoped_range loop_range{nvtx3::category{1}, nvtx3::payload{bufIndex}};
      auto timingHeader = m_det.getTimingHeader(bufIndex);
      //printf("*** PGPDrp::collector: bufIdx %u, th %p\n", bufIndex, timingHeader);
      //auto p = (const uint32_t*)timingHeader;
      //printf("*** PGPDrp::collector: th %08x %08x %08x %08x %08x %08x %08x %08x\n",
      //       p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
      auto pgpIndex = timingHeader->evtCounter & bufferMask;
      //printf("*** PGPDrp::collector: pgpIndex %u\n", pgpIndex);
      auto event = &pool.pgpEvents[pgpIndex];
      if (event->mask == 0) {
        printf("*** PGPDrp::collector: Broken event %p skipped, pgpIdx %u, bufIdx %u\n", event, pgpIndex, bufIndex);
        continue;                       // Skip broken event
      }
      //printf("*** PGPDrp::collector: event %p\n", event);

      auto pid = timingHeader->pulseId();
      //printf("*** P: pid %014lx, svc %u\n", pid, timingHeader->service());
      if (pid <= lastPid)
        logging::error("%s: PulseId did not advance: %014lx <= %014lx", __PRETTY_FUNCTION__, pid, lastPid);
      lastPid = pid;

      // Allocate transition datagrams before allocating the pebble buffer
      // since pebble buffers can't be freed out of order if this fails
      TransitionId::Value transitionId = timingHeader->service();
      EbDgram* trDgram;
      if (transitionId != TransitionId::L1Accept) {
        // Allocate a transition datagram from the pool
        trDgram = pool.allocateTr();
        if (!trDgram) [[unlikely]] {
          m_collector->freeDma(event);  // Leaves event mask = 0
          ++m_nNoTrDgrams;
          continue;                     // Can happen during shutdown
        }
      }

      // Allocate a pebble buffer
      //printf("*** Drp::collector: bufIndex %u, evtCtr %u\n", bufIndex, timingHeader->evtCounter);
      auto pebbleIndex = pool.allocate(); // This can block
      //printf("*** Drp::collector: pblIdx %u\n", pebbleIndex);
      event->pebbleIndex = pebbleIndex;
      Src src{m_det.nodeId};

      // Make a new dgram in the pebble
      // It must be an EbDgram in order to be able to send it to the MEB
      auto dgram = new(pool.pebble[pebbleIndex]) EbDgram(*timingHeader, src, m_para.rogMask);

      // Prepare the trigger primitive with whatever input is needed for the TEB to make trigger decisions
      auto l3InpBuf = tebContributor().fetch(pebbleIndex);
      auto l3InpDg  = new(l3InpBuf) EbDgram(*dgram);

      if (transitionId == TransitionId::L1Accept) {
        // @todo: Can event() here help with prescaled raw/calibrated data?
        //m_det.event(dgram, bufEnd);
        //const auto p = (uint32_t*)&timingHeader[1];
        //printf("*** PGPDrp::Collector: trgPrim %p, sz %zu, pyld %08x %08x\n", trgPrimitive, trgPrimitive->size(), p[0], p[1]);
        if (trgPrimitive) { // else this DRP doesn't provide TEB input
          // Copy the TEB input data from the GPU into the TEB input datagram
          auto tpSz = trgPrimitive->size();
          const void* l3BufEnd = (char*)l3InpDg + sizeof(*l3InpDg) + tpSz;
          auto buf = l3InpDg->xtc.alloc(tpSz, l3BufEnd);
          memcpy(buf, &timingHeader[1], tpSz);
        }
      } else {  // Transition
        logging::debug("Collector  saw %s @ %u.%09u (%014lx)",
                       TransitionId::name(transitionId),
                       dgram->time.seconds(), dgram->time.nanoseconds(), dgram->pulseId());

        // Store the empty transition dgram allocated above in the pebble
        pool.transitionDgrams[pebbleIndex] = trDgram;

        // Initialize the transition dgram's header
        memcpy((void*)trDgram, dgram, sizeof(*dgram) - sizeof(dgram->xtc));

        if (transitionId == TransitionId::SlowUpdate) {
          // Store the SlowUpdate's payload in the transition datagram
          const void* bufEnd  = (char*)trDgram + m_para.maxTrSize;
          m_det.slowupdate(trDgram->xtc, bufEnd);
          //printf("*** Collector: slowUpdate xtc extent %u\n", trDgram->xtc.extent);
        } else {                // Transition
          // copy the temporary xtc created on phase 1 of the transition
          // into the real location
          Xtc& trXtc = m_det.transitionXtc();
          trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
          const void* bufEnd  = (char*)trDgram + m_para.maxTrSize;
          auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
          memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());
          //printf("*** Collector: dg[%u] %014lx, tr %u, sz %zu\n", pebbleIndex, trDgram->pulseId(), trDgram->service(), sizeof(*trDgram) + trDgram->xtc.sizeofPayload());
        }
      }

      // Post level-3 input datagram to the TEB
      //printf("*** P: Sending input %u (%014lx, %08x) to TEB\n", pebbleIndex, pid, dgram->env);
      tebContributor().process(pebbleIndex);
      nvtx3::mark("Sent to TEB", nvtx3::payload{pebbleIndex});

      // Time out batches for the TEB
      /// while (!m_workerQueues[worker].try_pop(batch)) { // Poll
      ///     if (tebContributor.timeout()) {              // After batch is timed out,
      ///         rc = m_workerQueues[worker].popW(batch); // pend
      ///         break;
      ///     }
      /// }
      /// logging::debug("Worker %d popped batch %u, size %zu\n", worker, batch.start, batch.size);

      bufIndex = (bufIndex + 1) & bufferMask;
    }
  }

  // Flush the Reader buffers
  // @todo: dmaFlush();
  pool.flushPebble();

  if (exposer())  exporter.reset();

  logging::info("Collector is exiting");
}
