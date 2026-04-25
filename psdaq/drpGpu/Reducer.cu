#include "Reducer.hh"

#include "Detector.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/eb/ResultDgram.hh"
#include "ReducerAlgo.hh"

#include <sys/prctl.h>

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Drp;
using namespace Drp::Gpu;
using namespace Pds;
using namespace Pds::Eb;

using us_t = std::chrono::microseconds;

struct red_domain{ static constexpr char const* name{"Reducer"}; };
using red_scoped_range = nvtx3::scoped_range_in<red_domain>;

static inline unsigned nxtPwrOf2(unsigned n)
{
  return n > 1 ? 1 << (32 - __builtin_clz(n - 1)) : 0;
}


Reducer::Reducer(const Parameters&                  para,
                 MemPoolGpu&                        pool,
                 Detector&                          det,
                 cudaExecutionContext_t             green_ctx,
                 const std::atomic<bool>&           terminate,
                 const cuda::std::atomic<unsigned>& terminate_d) :
  m_pool       (pool),
  m_terminate  (terminate),
  m_terminate_d(terminate_d),
  m_reduce_us  (0),
  m_para       (para)
{
  // Get the range of priorities available [ greatest_priority, lowest_priority ]
  int prioLo;
  int prioHi;
  chkError(cudaDeviceGetStreamPriorityRange(&prioLo, &prioHi));
  int prio{prioHi+1};
  logging::debug("Reducer stream priority (range: LOW: %d to HIGH: %d): %d", prioLo, prioHi, prio);

  // Create the Reducer streams
  m_streams.resize(m_para.nworkers);
  m_t0.resize(m_para.nworkers);
  m_indices_h.resize(m_para.nworkers);
  m_indices_d.resize(m_para.nworkers);
  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    //chkFatal(cudaStreamCreateWithPriority(&m_streams[i], cudaStreamNonBlocking, prio));
    chkFatal(cudaExecutionCtxStreamCreate(&m_streams[i], green_ctx, cudaStreamNonBlocking, prio));

    // Keep track of the head and tail indices of the Reducer stream
    chkError(cudaHostAlloc(&m_indices_h[i], sizeof(*m_indices_h[i]), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_indices_d[i], m_indices_h[i], 0));
    *m_indices_h[i] = 0;
  }
  logging::debug("Done with creating %u Reducer streams", m_streams.size());

  // Set up the reducer algorithm instances
  m_algos.resize(m_para.nworkers);
  if (!_setupAlgos(det)) {
    logging::critical("Error setting up Reducer Algorithm instances");
    abort();
  }

  // The header consists of the Dgram with the parent Xtc, the ShapesData Xtc, the
  // Shapes Xtc with its payload and Data Xtc, the payload of which is on the GPU.
  auto headerSize  = sizeof(Dgram) + 3 * sizeof(Xtc) + MaxRank * sizeof(uint32_t);
  auto payloadSize = m_algos.size() ? m_algos[0]->payloadSize() : 0; // Each instance returns the same value
  auto totalSize   = headerSize + payloadSize;
  if (totalSize < m_para.maxTrSize)  payloadSize = m_para.maxTrSize - headerSize;

  // Prepare buffers to receive the reduced data,
  // prepended with some reserved space for the datagram header.
  // The application sees only the pointer to the data buffer.
  m_pool.createReduceBuffers(payloadSize, headerSize);

  // Set up the worker queues to fit all buffers
  if (m_para.nworkers) {
    auto nEntries{nxtPwrOf2((m_pool.nbuffers() + m_para.nworkers-1) / m_para.nworkers)};
#ifndef HOST_LAUNCHED_REDUCERS
    if (m_algos[0]->hasGraph()) {         // Same value for all instances
      m_inputQueues2.resize(m_para.nworkers);
      m_outputQueues2.resize(m_para.nworkers);
      for (unsigned i = 0; i < m_para.nworkers; ++i) {
        printf("*** Reducer::ctor: 1 wkr %u\n", i);
        auto& iq = m_inputQueues2[i];
        iq.h = new RingQueueHtoD<unsigned>(nEntries);
        chkError(cudaMalloc(&iq.d,       sizeof(*iq.d)));
        chkError(cudaMemcpy( iq.d, iq.h, sizeof(*iq.d), cudaMemcpyHostToDevice));
        printf("*** Reducer::ctor: 2 wkr %u\n", i);
        auto& oq = m_outputQueues2[i];
        oq.h = new RingQueueDtoH<ReducerTuple>(nEntries);
        chkError(cudaMalloc(&oq.d,       sizeof(*oq.d)));
        chkError(cudaMemcpy( oq.d, oq.h, sizeof(*oq.d), cudaMemcpyHostToDevice));
        printf("*** Reducer::ctor: 3 wkr %u\n", i);
      }
    } else
#endif
    {
      for (unsigned i = 0; i < m_para.nworkers; ++i) {
        m_inputQueues.emplace_back(nEntries);
        m_outputQueues.emplace_back(nEntries);
      }
    }

    // @todo: TBD: Location to retrieve error return code from
    m_retCode_d.resize(m_para.nworkers);
    for (unsigned i = 0; i < m_para.nworkers; ++i) {
      chkError(cudaMalloc(&m_retCode_d[i],    sizeof(*m_retCode_d[i])));
      chkError(cudaMemset( m_retCode_d[i], 0, sizeof(*m_retCode_d[i])));
    }

    // Set up a state variable
    m_state_d.resize(m_para.nworkers);
    for (unsigned i = 0; i < m_para.nworkers; ++i) {
      chkError(cudaMalloc(&m_state_d[i],    sizeof(*m_state_d[i])));
      chkError(cudaMemset( m_state_d[i], 0, sizeof(*m_state_d[i])));
    }

    // Prepare metrics for tracking kernel state and execution progress
    m_metrics.state.resize(m_para.nworkers);
    m_metrics.inpWtCtr.resize(m_para.nworkers);
    m_metrics.outWtCtr.resize(m_para.nworkers);
    for (unsigned i = 0; i < m_para.nworkers; ++i) {
      if (!m_metrics.state[i].h) {
        chkError(cudaHostAlloc(&m_metrics.state[i].h, sizeof(*m_metrics.state[i].h), cudaHostAllocDefault));
        chkError(cudaHostGetDevicePointer(&m_metrics.state[i].d, m_metrics.state[i].h, 0));
      }
      *m_metrics.state[i].h = 0;

      chkError(cudaHostAlloc(&m_metrics.inpWtCtr[i].h, sizeof(*m_metrics.inpWtCtr[i].h), cudaHostAllocDefault));
      chkError(cudaHostGetDevicePointer(&m_metrics.inpWtCtr[i].d, m_metrics.inpWtCtr[i].h, 0));
      *m_metrics.inpWtCtr[i].h = 0;
      chkError(cudaHostAlloc(&m_metrics.outWtCtr[i].h, sizeof(*m_metrics.outWtCtr[i].h), cudaHostAllocDefault));
      chkError(cudaHostGetDevicePointer(&m_metrics.outWtCtr[i].d, m_metrics.outWtCtr[i].h, 0));
      *m_metrics.outWtCtr[i].h = 0;
    }

    if (m_algos[0]->hasGraph()) {         // Same value for all instances
      // Prepare the CUDA graphs
      m_graphExecs.resize(m_para.nworkers);
      for (unsigned i = 0; i < m_para.nworkers; ++i) {
        if (_setupGraph(i)) {
          logging::critical("Failed to set up Reducer graph[%u]", i);
          abort();
        }
      }
    }
  }

  printf("*** Reducer::ctor end\n");
}

Reducer::~Reducer()
{
  printf("*** Reducer::dtor\n");
  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    if (m_metrics.inpWtCtr[i].h) {
      chkError(cudaFreeHost(m_metrics.inpWtCtr[i].h));
      m_metrics.inpWtCtr[i].h = nullptr;
      m_metrics.inpWtCtr[i].d = nullptr;
    }
    if (m_metrics.outWtCtr[i].h) {
      chkError(cudaFreeHost(m_metrics.outWtCtr[i].h));
      m_metrics.outWtCtr[i].h = nullptr;
      m_metrics.outWtCtr[i].d = nullptr;
    }

    if (m_metrics.state[i].h) {
      cudaFreeHost(m_metrics.state[i].h);
      m_metrics.state[i].h = nullptr;
      m_metrics.state[i].d = nullptr;
    }
  }
  m_metrics.inpWtCtr.clear();
  m_metrics.outWtCtr.clear();
  m_metrics.state.clear();

  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    if (m_state_d[i])  chkError(cudaFree(m_state_d[i]));
    m_state_d[i] = nullptr;
  }
  m_state_d.clear();

  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    if (m_retCode_d[i])  chkError(cudaFree(m_retCode_d[i]));
    m_retCode_d[i] = nullptr;
  }
  m_retCode_d.clear();

#ifndef HOST_LAUNCHED_REDUCERS
  if (m_algos.size() && m_algos[0]->hasGraph()) { // Same value for all workers
    for (unsigned i = 0; i < m_para.nworkers; ++i) {
      if (m_inputQueues2[i].d)  chkError(cudaFree(m_inputQueues2[i].d));
      if (m_inputQueues2[i].h)  delete m_inputQueues2[i].h;
      if (m_outputQueues2[i].d)  chkError(cudaFree(m_outputQueues2[i].d));
      if (m_outputQueues2[i].h)  delete m_outputQueues2[i].h;
    }
    m_inputQueues2.clear();
    m_outputQueues2.clear();
  } else
#endif
  {
    if (m_threads.size()) {
      logging::info("Shutting down reducer workers");
    }
    for (unsigned i = 0; i < m_inputQueues.size(); i++) {
      m_inputQueues[i].shutdown();
    }
    for (unsigned i = 0; i < m_threads.size(); i++) {
      if (m_threads[i].joinable()) {
        m_threads[i].join();
      }
    }
    for (unsigned i = 0; i < m_outputQueues.size(); i++) {
      m_outputQueues[i].shutdown();
    }
    if (m_threads.size()) {
      logging::info("Reducer worker threads finished");
    }
    m_outputQueues.clear();
    m_inputQueues.clear();
    m_threads.clear();
  }

  printf("*** Reducer dtor 1\n");
  for (auto& graphExec : m_graphExecs) {
    chkError(cudaGraphExecDestroy(graphExec));
  }
  m_graphExecs.clear();
  printf("*** Reducer dtor 2\n");

  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    if (m_algos[i])  delete m_algos[i];
  }
  m_algos.clear();
  m_dl.close();
  printf("*** Reducer dtor 3\n");

  m_pool.destroyReduceBuffers();
  printf("*** Reducer dtor 4\n");

  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    chkError(cudaFreeHost(m_indices_h[i]));

    chkError(cudaStreamDestroy(m_streams[i]));
  }
  m_indices_h.clear();
  m_indices_d.clear();
  m_streams.clear();
  printf("*** Reducer dtor 5\n");

  printf("*** Reducer dtor end\n");
}

int Reducer::setupMetrics(const std::shared_ptr<MetricExporter> exporter,
                          std::map<std::string, std::string>&   labels)
{
  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    auto wkr = std::to_string(i);
    exporter->add("DRP_redState"+wkr, labels, MetricType::Gauge, [&, i](){ return m_metrics.state[i].h ? *m_metrics.state[i].h : 0; });

    *m_metrics.inpWtCtr[i].h = 0;
    *m_metrics.outWtCtr[i].h = 0;
    exporter->add("DRP_inpWtCtr"+wkr, labels, MetricType::Counter, [&, i](){ return m_metrics.inpWtCtr[i].h ? *m_metrics.inpWtCtr[i].h : 0; });
    exporter->add("DRP_outWtCtr"+wkr, labels, MetricType::Counter, [&, i](){ return m_metrics.outWtCtr[i].h ? *m_metrics.outWtCtr[i].h : 0; });
  }

  if (m_algos.size() && m_algos[0]->hasGraph()) {         // Same value for all workers
    for (unsigned i = 0; i < m_inputQueues2.size(); ++i) {
      auto wkr = std::to_string(i);
      exporter->add("DRP_inputQueue"+wkr,  labels, MetricType::Gauge, [&, i](){ return m_inputQueues2[i].h->occupancy(); });
      exporter->add("DRP_outputQueue"+wkr, labels, MetricType::Gauge, [&, i](){ return m_outputQueues2[i].h->occupancy(); });
    }
  } else {
    for (unsigned i = 0; i < m_inputQueues.size(); ++i) {
      auto wkr = std::to_string(i);
      exporter->add("DRP_inputQueue"+wkr,  labels, MetricType::Gauge, [&, i](){ return m_inputQueues[i].guess_size(); });
      exporter->add("DRP_outputQueue"+wkr, labels, MetricType::Gauge, [&, i](){ return m_outputQueues[i].guess_size(); });
    }
  }

  exporter->add("DRP_reduceTime", labels, MetricType::Gauge, [&](){ return m_reduce_us; });

  return 0;
}

bool Reducer::_setupAlgos(Detector& det)
{
  // @todo: In the future, find out which Reducer to load from the Detector's configDb entry
  //        For now, load it according to a command line kwarg parameter
  std::string reducer;
  if (m_para.kwargs.find("reducer") == m_para.kwargs.end()) {
    logging::error("Missing required kwarg 'reducer'");
    return false;
  }
  reducer = const_cast<Parameters&>(m_para).kwargs["reducer"];

  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    if (m_algos[i])  delete m_algos[i]; // If the object exists, delete it
  }
  m_dl.close();                         // If a lib is open, close it first

  const std::string soName("lib"+reducer+".so");
  logging::debug("Loading library '%s'", soName.c_str());
  if (m_dl.open(soName, RTLD_LAZY)) {
    logging::error("Error opening library '%s'", soName.c_str());
    return false;
  }
  const std::string symName("createReducer");
  auto createFn = m_dl.loadSymbol(symName.c_str());
  if (!createFn) {
    logging::error("Symbol '%s' not found in %s",
                   symName.c_str(), soName.c_str());
    return false;
  }
  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    auto instance = reinterpret_cast<reducerAlgoFactoryFn_t*>(createFn)(m_para, m_pool, det);
    if (!instance)
    {
      logging::error("Error calling %s from %s", symName.c_str(), soName.c_str());
      return false;
    }
    m_algos[i] = instance;
  }
  logging::info("Loaded reducer library '%s'", soName.c_str());
  return true;
}

int Reducer::_setupGraph(unsigned worker)
{
  cudaGraph_t      graph;
  cudaGraphExec_t& graphExec = m_graphExecs[worker];
  cudaStream_t     stream    = m_streams[worker];

  // Build the graph
  logging::debug("Recording Reducer graph %u", worker);
  graph = _recordGraph(worker);
  if (graph == 0) {
    return -1;
  }

  // Instantiate the executable graph
  if (chkError(cudaGraphInstantiate(&graphExec, graph, cudaGraphInstantiateFlagDeviceLaunch),
               "Reducer graph create failed")) {
    return -1;
  }

  // No need to hang on to the stream info
  cudaGraphDestroy(graph);

  // Upload the graph so it can be launched by the scheduler kernel later
  logging::debug("Uploading Reducer graph %u...", worker);
  if (chkError(cudaGraphUpload(graphExec, stream), "Reducer graph upload failed")) {
    return -1;
  }

  return 0;
}

#ifndef HOST_LAUNCHED_REDUCERS
/** This kernel receives a message from TebReceiver that indicates which
 * calibBuffer is ready for reducing.
 */
static __global__
void _reducerRcv(unsigned*                const __restrict__ state,
                 unsigned*                const __restrict__ index,
                 RingQueueHtoD<unsigned>* const __restrict__ inputQueue,
                 uint64_t*                const __restrict__ stateMon,
                 uint64_t* const                __restrict__ inpWtCtr)
{
  if (*state == 0) {
    //*stateMon = 1;
    //printf("### reducerRcv: wait for idx\n");
    unsigned ns{8};
    while (!inputQueue->pop(index)) {
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
      else {
        //*stateMon = 2;
        return;
      }
    }
    //printf("### _reducerRcv: got idx %u\n", *index);
    *state = 1;
    //*stateMon = 3;
    //++(*inpWtCtr);
  }
}

/** This will re-launch the current graph */
static __global__
void _reducerLoop(unsigned*                    const __restrict__ state,
                  unsigned const*              const __restrict__ index,
                  uint8_t*                     const __restrict__ dataBuffers,
                  size_t                       const              dataBufsCnt,
                  RingQueueDtoH<ReducerTuple>* const __restrict__ outputQueue,
                  uint64_t*                    const __restrict__ stateMon,
                  uint64_t*                    const __restrict__ outWtCtr,
                  cuda::std::atomic<unsigned>  const&             terminate)
{
  if (*state == 2) {
    //*stateMon = 4;
    auto const __restrict__ data = &dataBuffers[*index * dataBufsCnt];
    auto dataSize = ((size_t*)data)[-1];
    //printf("### _reducerLoop: pushing {%u, %lu}\n", *index, dataSize);
    bool rc;
    unsigned ns{8};
    while ( (rc = !outputQueue->push({*index, dataSize})) ) {
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
      else {
        //*stateMon = 5;
        break;
      }
    }
    if (!rc) {
      //printf("### _reducerLoop: pushed {%u, %lu}\n", *index, dataSize);
      *state = 0;
      //*stateMon = 6;
      //++(*outWtCtr);
    }
  }

  // This will re-launch the current graph
  //printf("### _reducerLoop: relaunch\n");
  if (!terminate.load(cuda::std::memory_order_acquire))  {
    cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
  }
}
#endif

cudaGraph_t Reducer::_recordGraph(unsigned worker)
{
  red_scoped_range r{/*"Reducer::_recordGraph"*/}; // Expose function name via NVTX

  auto stream       = m_streams[worker];
  auto calibBuffers = m_pool.calibBuffers_d();
  auto calibBufsSz  = m_pool.calibBufsSize();
  auto calibBufsCnt = calibBufsSz / sizeof(*calibBuffers);
  auto dataBuffers  = m_pool.reduceBuffers_d();
  auto dataBufsRsvd = m_pool.reduceBufsReserved();
  auto dataBufsSz   = m_pool.reduceBufsSize();
  auto dataBufsCnt  = (dataBufsRsvd + dataBufsSz) / sizeof(*dataBuffers);

  if (chkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "Reducer stream begin capture failed")) {
    return 0;
  }

#ifndef HOST_LAUNCHED_REDUCERS
  // Handle messages from TebReceiver to process an event
  //printf("*** Reducer::_recordGraph: worker %d, iq h %p, d %p\n", worker, m_inputQueues2[worker].h, m_inputQueues2[worker].d);
  _reducerRcv<<<1, 1, 0, stream>>>(m_state_d[worker],
                                   m_indices_d[worker],
                                   m_inputQueues2[worker].d,
                                   m_metrics.state[worker].d,
                                   m_metrics.inpWtCtr[worker].d);
  chkError(cudaGetLastError(), "Launch of _reducerRcv kernel failed");
#endif

  // Perform the reduction algorithm
  printf("*** Reducer::recordGraph: state[%u] %p\n", worker, m_metrics.state[worker].d);
  m_algos[worker]->recordGraph(stream,
                               m_state_d[worker],
                               m_indices_d[worker],
                               calibBuffers,
                               calibBufsCnt,
                               dataBuffers,
                               dataBufsCnt);

#ifndef HOST_LAUNCHED_REDUCERS
  // Post the completed buffer results and relaunch
  //printf("*** Reducer::_recordGraph: worker %d, oq h %p, d %p\n", worker, m_outputQueues2[worker].h, m_outputQueues2[worker].d);
  _reducerLoop<<<1, 1, 0, stream>>>(m_state_d[worker],
                                    m_indices_d[worker],
                                    dataBuffers,
                                    dataBufsCnt,
                                    m_outputQueues2[worker].d,
                                    m_metrics.state[worker].d,
                                    m_metrics.outWtCtr[worker].d,
                                    m_terminate_d);
  chkError(cudaGetLastError(), "Launch of _reducerLoop kernel failed");
#endif

  cudaGraph_t graph;
  if (chkError(cudaStreamEndCapture(stream, &graph),
               "Reducer stream end capture failed")) {
    return 0;
  }

  return graph;
}

void Reducer::startup()
{
#ifndef HOST_LAUNCHED_REDUCERS
  printf("*** Reducer::startup: 1\n");
  if (m_algos.size() && m_algos[0]->hasGraph()) {           // Same value for all workers
    printf("*** Reducer::startup: 2\n");

    // Launch the Reducer graphs
    for (unsigned i = 0; i < m_graphExecs.size(); ++i) {
      printf("*** Reducer::startup: 3, wkr %d\n", i);
      chkFatal(cudaGraphLaunch(m_graphExecs[i], m_streams[i]));
    }
  }
  printf("*** Reducer::startup: 4\n");
#else
  // Start the worker threads
  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    m_threads.emplace_back(&Reducer::_worker, std::ref(*this), i);
  }
#endif
}

void Reducer::shutdown()
{
  if (m_algos.size() && !m_algos[0]->hasGraph()) { // Same value for all workers
    for (auto& outputQueue: m_outputQueues)
      outputQueue.shutdown();
    for (auto& inputQueue: m_inputQueues)
      inputQueue.shutdown();
  }
}

void Reducer::dump() const
{
  if (m_algos[0]->hasGraph()) {
    for (unsigned i = 0; i < m_para.nworkers; ++i) {
      printf("Reducer %u: in: head %u, tail %u, out: head %u, tail %u\n", i,
             m_inputQueues2[i].h->head(), m_inputQueues2[i].h->tail(),
             m_outputQueues2[i].h->head(), m_outputQueues2[i].h->tail());
    }
  }
}

void Reducer::_worker(unsigned worker)
{
  red_scoped_range r{/*"Reducer::_worker"*/}; // Expose function name via NVTX

  logging::info("Reducer worker %u is starting with process ID %lu", worker, syscall(SYS_gettid));
  char nameBuf[16];
  snprintf(nameBuf, sizeof(nameBuf), "ReducerWkr%d", worker);
  if (prctl(PR_SET_NAME, nameBuf, 0, 0, 0) == -1) {
    perror("prctl");
  }

  // Establish context in this thread
  chkError(cudaSetDevice(m_pool.context().deviceNo()));

  auto algo         = m_algos[worker];
  auto index        = m_indices_h[worker];
  auto stream       = m_streams[worker];
  auto& inputQueue  = m_inputQueues[worker];
  auto& outputQueue = m_outputQueues[worker];
  //auto calibBufsSz  = m_pool.calibBufsSize();
  cudaGraphExec_t graph{0};
  if (algo->hasGraph())  graph = m_graphExecs[worker];

  unsigned idx;
  while (inputQueue.pop(idx)) {
    red_scoped_range loop_range{/*"Reducer::_worker", */nvtx3::payload{idx}};
    if  (algo->hasGraph())  *index = idx;
    //printf("*** Reducer::_worker: worker %u index %u\n", worker, idx);

    auto t0{fast_monotonic_clock::now(CLOCK_MONOTONIC)};

    // Launch the Reducer
    size_t   dataSize{0};
    unsigned retCode{0};
    algo->reduce(graph, stream, idx, &dataSize, &retCode);

    if  (algo->hasGraph()) {
      // Wait for the graph to complete
      chkError(cudaStreamSynchronize(stream));
    }

    auto now{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
    m_reduce_us = std::chrono::duration_cast<us_t>(now - t0).count();
    //printf("*** Reducer::_worker: dt %lu\n", m_reduce_us);
    //auto ratio{double(calibBufsSz)/double(cmpSize1)};
    //printf("*** dt %lu us, in %zu / out %zu = %f\n", dt, calibBufsSz, cmpSize1, ratio);

    if (retCode)
      logging::error("Reducer found mismatch errors", retCode);

    // Signal completion to the recorder
    outputQueue.push({idx, dataSize});
  }

  logging::info("Reducer worker %u is exiting", worker);
}
