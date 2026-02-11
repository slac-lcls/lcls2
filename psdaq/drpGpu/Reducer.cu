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

  // Create a 'done' flag
  chkError(cudaMalloc(&m_done_d,    sizeof(*m_done_d)));
  chkError(cudaMemset( m_done_d, 0, sizeof(*m_done_d)));

  // Create the Reducer streams
  m_streams.resize(m_para.nworkers);
  m_t0.resize(m_para.nworkers);
  m_heads_h.resize(m_para.nworkers);
  m_heads_d.resize(m_para.nworkers);
  m_tails_h.resize(m_para.nworkers);
  m_tails_d.resize(m_para.nworkers);
  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    chkFatal(cudaStreamCreateWithPriority(&m_streams[i], cudaStreamNonBlocking, prio));

    // Keep track of the head and tail indices of the Reducer stream
    chkError(cudaHostAlloc(&m_heads_h[i], sizeof(*m_heads_h[i]), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_heads_d[i], m_heads_h[i], 0));
    *m_heads_h[i] = 0;
    chkError(cudaHostAlloc(&m_tails_h[i], sizeof(*m_tails_h[i]), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_tails_d[i], m_tails_h[i], 0));
    *m_tails_h[i] = 0;
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
    if (m_algos[0]->hasGraph()) {         // Same value for all instances
      m_inputQueues2.resize(m_para.nworkers);
      m_outputQueues2.resize(m_para.nworkers);
      for (unsigned i = 0; i < m_para.nworkers; ++i) {
        printf("*** Reducer::ctor: 1 wkr %u\n", i);
        auto& iq = m_inputQueues2[i];
        iq.h = new RingQueueHtoD<unsigned>(nEntries, m_terminate, m_terminate_d);
        chkError(cudaMalloc(&iq.d,       sizeof(*iq.d)));
        chkError(cudaMemcpy( iq.d, iq.h, sizeof(*iq.d), cudaMemcpyHostToDevice));
        printf("*** Reducer::ctor: 2 wkr %u\n", i);
        auto& oq = m_outputQueues2[i];
        oq.h = new RingQueueDtoH<ReducerTuple>(nEntries, m_terminate, m_terminate_d);
        chkError(cudaMalloc(&oq.d,       sizeof(*oq.d)));
        chkError(cudaMemcpy( oq.d, oq.h, sizeof(*oq.d), cudaMemcpyHostToDevice));
        printf("*** Reducer::ctor: 3 wkr %u\n", i);
      }
    } else {
      for (unsigned i = 0; i < m_para.nworkers; ++i) {
        m_inputQueues.emplace_back(nEntries);
        m_outputQueues.emplace_back(nEntries);
      }

      // Start the worker threads
      for (unsigned i = 0; i < m_para.nworkers; ++i) {
        m_threads.emplace_back(&Reducer::_worker, std::ref(*this), i);
      }
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
  if (m_algos.size() && m_algos[0]->hasGraph()) { // Same value for all instances
    for (unsigned i = 0; i < m_para.nworkers; ++i) {
      if (m_inputQueues2[i].d)  chkError(cudaFree(m_inputQueues2[i].d));
      if (m_inputQueues2[i].h)  delete m_inputQueues2[i].h;
      if (m_outputQueues2[i].d)  chkError(cudaFree(m_outputQueues2[i].d));
      if (m_outputQueues2[i].h)  delete m_outputQueues2[i].h;
    }
    m_inputQueues2.clear();
    m_outputQueues2.clear();
  } else {
    if (m_threads.size())
      logging::info("Shutting down reducer workers");
    for (unsigned i = 0; i < m_threads.size(); i++) {
      m_inputQueues[i].shutdown();
      if (m_threads[i].joinable()) {
        m_threads[i].join();
      }
    }
    if (m_threads.size()) {
      logging::info("Reducer worker threads finished");
    }
    for (unsigned i = 0; i < m_outputQueues.size(); i++) {
      m_outputQueues[i].shutdown();
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
    chkError(cudaFreeHost(m_tails_h[i]));
    chkError(cudaFreeHost(m_heads_h[i]));

    chkError(cudaStreamDestroy(m_streams[i]));
  }
  m_heads_h.clear();
  m_heads_d.clear();
  m_tails_h.clear();
  m_tails_d.clear();
  m_streams.clear();
  printf("*** Reducer dtor 5\n");

  if (m_done_d)  chkError(cudaFree(m_done_d));
  printf("*** Reducer dtor end\n");
}

int Reducer::setupMetrics(const std::shared_ptr<MetricExporter> exporter,
                          std::map<std::string, std::string>&   labels)
{
  if (m_algos.size() && m_algos[0]->hasGraph()) {         // Same value for all instances
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

int Reducer::_setupGraph(unsigned instance)
{
  cudaGraph_t      graph;
  cudaGraphExec_t& graphExec = m_graphExecs[instance];
  cudaStream_t     stream    = m_streams[instance];

  // Build the graph
  logging::debug("Recording Reducer graph %u", instance);
  graph = _recordGraph(instance);
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
  logging::debug("Uploading Reducer graph %u...", instance);
  if (chkError(cudaGraphUpload(graphExec, stream), "Reducer graph upload failed")) {
    return -1;
  }

  return 0;
}

/** This kernel receives a message from TebReceiver that indicates which
 * calibBuffer is ready for reducing.
 */
static __global__ void _receive(//unsigned* const       __restrict__ head,
                                //unsigned* const       __restrict__ tail,
                                //const cuda::std::atomic<unsigned>& terminate)
                                unsigned&                          index,
                                Gpu::RingQueueHtoD<unsigned>&      inputQueue,
                                unsigned&                          done)
{
//  //printf("### _receive 1 done %d, tail %u, head %u\n", terminate.load(), *tail, *head);
//
//  // Wait for the head to advance with respect to the tail
//  auto t = *tail;
//  unsigned ns = 8;
//  while (*head == t) {
//    if (terminate.load(cuda::std::memory_order_acquire))  break;
//    __nanosleep(ns);
//    if (ns < 256)  ns *= 2;
//  }
//  //printf("### Reducer receive:   h %u, t %u, d %d\n", *head, t, terminate.load());

  //printf("### Reducer receive: 1, done %u\n", done);
  done |= !inputQueue.pop(index);
  //printf("### Reducer receive: 2, idx %u, done %u\n", *index, done);
}

/** This will re-launch the current graph */
static __global__ void _graphLoop(//unsigned* const       __restrict__ head,
                                  //unsigned* const       __restrict__ tail,
                                  //const cuda::std::atomic<unsigned>& terminate)
                                  unsigned const&                    index,
                                  uint8_t* const        __restrict__ dataBuffers,
                                  size_t   const                     dataBufsCnt,
                                  Gpu::RingQueueDtoH<ReducerTuple>&  outputQueue,
                                  unsigned&                          done)
{
//  //printf("### Reducer graphLoop: 1, done %d, idx %u\n", terminate.load(), *head);
//  if (terminate.load(cuda::std::memory_order_acquire))  return;
//
//  //printf("### Reducer graphLoop: 2 t %u, h %u\n", *tail, *head);
//
//  // Signal that this worker is done
//  *tail = *head;                   // With nworkers > 1, head - tail may be > 1

  auto const __restrict__ data = &dataBuffers[index * dataBufsCnt];
  auto dataSize = ((size_t*)data)[-1];
  //printf("### Reducer graphLoop: push {%u, %lu}, done %u\n", index, dataSize, done);
  if (!done) {
    if (!(done = !outputQueue.push({index, dataSize}))) {
      cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
    }
  }
  //printf("### Reducer graphLoop: 3, done %u\n", done);
}

cudaGraph_t Reducer::_recordGraph(unsigned instance)
{
  red_scoped_range r{/*"Reducer::_recordGraph"*/}; // Expose function name via NVTX

  auto stream       = m_streams[instance];
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

  // Handle messages from TebReceiver to process an event
  //_receive<<<1, 1, 0, stream>>>(m_heads_d[instance], m_tails_d[instance], m_terminate_d);
  printf("*** Reducer::_recordGraph: instance %d, iq h %p, d %p\n", instance, m_inputQueues2[instance].h, m_inputQueues2[instance].d);
  _receive<<<1, 1, 0, stream>>>(*m_heads_d[instance], *m_inputQueues2[instance].d, *m_done_d);

  // Perform the reduction algorithm
  m_algos[instance]->recordGraph(stream,
                                 *m_heads_d[instance],
                                 calibBuffers,
                                 calibBufsCnt,
                                 dataBuffers,
                                 dataBufsCnt);

  // Re-launch! Additional behavior can be put in graphLoop as needed.
  //_graphLoop<<<1, 1, 0, stream>>>(m_heads_d[instance],
  //                                m_tails_d[instance],
  //                                m_terminate_d);
  printf("*** Reducer::_recordGraph: instance %d, oq h %p, d %p\n", instance, m_outputQueues2[instance].h, m_outputQueues2[instance].d);
  _graphLoop<<<1, 1, 0, stream>>>(*m_heads_d[instance],
                                  dataBuffers,
                                  dataBufsCnt,
                                  *m_outputQueues2[instance].d,
                                  *m_done_d);

  // Signal to the host that the worker is done
  //chkError(cudaEventRecord(event, stream));

  cudaGraph_t graph;
  if (chkError(cudaStreamEndCapture(stream, &graph),
               "Reducer stream end capture failed")) {
    return 0;
  }

  return graph;
}

void Reducer::startup()
{
  printf("*** Reducer::startup: 1\n");
  if (m_algos.size() && m_algos[0]->hasGraph()) {           // Same value for all instances
    chkError(cuCtxSetCurrent(m_pool.context().context()));  // Needed, else kernels misbehave
    printf("*** Reducer::startup: 2\n");

    // Launch the Reducer graphs
    for (unsigned i = 0; i < m_graphExecs.size(); ++i) {
      printf("*** Reducer::startup: 3, wkr %d\n", i);
      chkFatal(cudaGraphLaunch(m_graphExecs[i], m_streams[i]));
    }
  }
  printf("*** Reducer::startup: 4\n");
}

void Reducer::shutdown()
{
  if (m_algos.size() && !m_algos[0]->hasGraph()) { // Same value for all instances
    for (auto& outputQueue: m_outputQueues)
      outputQueue.shutdown();
    for (auto& inputQueue: m_inputQueues)
      inputQueue.shutdown();
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

  chkError(cuCtxSetCurrent(m_pool.context().context())); // Needed, else kernels misbehave

  auto algo         = m_algos[worker];
  auto head         = m_heads_h[worker];
  //auto tail         = m_tails_h[worker];
  auto stream       = m_streams[worker];
  auto& inputQueue  = m_inputQueues[worker];
  auto& outputQueue = m_outputQueues[worker];
  cudaGraphExec_t graph{0};
  if (algo->hasGraph())  graph = m_graphExecs[worker];

  unsigned index;
  while (inputQueue.pop(index)) {
    red_scoped_range loop_range{/*"Reducer::_worker", */nvtx3::payload{index}};
    size_t dataSize;
    if  (algo->hasGraph()) {
      //// Wait for the graph to finish executing before updating head
      //unsigned hd, tl;
      //do {
      //  chkError(cudaMemcpyAsync((void*)&hd, head, sizeof(*head), cudaMemcpyDeviceToHost, stream));
      //  chkError(cudaMemcpyAsync((void*)&tl, tail, sizeof(*tail), cudaMemcpyDeviceToHost, stream));
      //  chkError(cudaStreamSynchronize(stream));
      //  //printf("*** Reducer::start[%u]: tail %d, head %d\n", worker, tl, hd);
      //} while (hd != tl);                     // Wait if the kernel is still processing
      //chkError(cudaMemcpyAsync((void*)head, &index, sizeof(index), cudaMemcpyHostToDevice, stream));
      *head = index;
    }

    printf("*** Reducer worker: algo->reduce\n");

    auto t0{fast_monotonic_clock::now(CLOCK_MONOTONIC)};

    // Launch the Reducer
    algo->reduce(graph, stream, index, &dataSize);

    if  (algo->hasGraph()) {
      // Wait for the graph to complete
      chkError(cudaStreamSynchronize(stream));
    }

    auto now{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
    m_reduce_us = std::chrono::duration_cast<us_t>(now - t0).count();

    // Signal completion to the recorder
    outputQueue.push({index, dataSize});
  }

  logging::info("Reducer worker %u is exiting", worker);
}
