#include <getopt.h>
#include <cstddef>
#include <vector>
#include <thread>
#include <chrono>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "GpuAsyncLib.hh"
#include "psdaq/service/fast_monotonic_clock.hh"
#include "drp/spscqueue.hh"

namespace cg = cooperative_groups;

using namespace Pds;

using us_t = std::chrono::microseconds;

static const unsigned NumAsics   {   6 };
static const unsigned NumRows    { 192 };
static const unsigned NumCols    { 168 };
static const unsigned NPixels    { NumAsics*NumRows*NumCols };
static const unsigned GainOffset {  14 };
static const unsigned GainBits   {   2 };
static const unsigned NGains     {   4 };

static const unsigned NEvents    { 16 }; // Default number of events to process

struct PedGain1
{
  float ped[NGains];
  float gain[NGains];
};

struct PedGain2
{
  float ped;
  float gain;
};

using vecf32_t = std::vector<float>;
using vecu16_t = std::vector<uint16_t>;


static void check(char const* const           name,
                  std::vector<vecf32_t>       out,
                  std::vector<vecf32_t> const reference,
                  std::vector<uint64_t>       calibTimes,
                  float                       kernelTime=0.f)
{
  unsigned nEvents = calibTimes.size();
  double tTot = 0.0f;
  for (unsigned i = 0; i < nEvents; ++i) {
    if (out[i] != reference[i]) {
      printf("%s calibrate mismatch in event %u\n", name, i);
      unsigned count = 0;
      for (unsigned j = 0; j < reference[i].size(); ++j) {
        if (out[i][j] != reference[i][j]) {
          if (count > 10)  break;
          printf("out[%u] %f vs ref[%u] %f\n", j, out[i][j], j, reference[i][j]);
          ++count;
        }
      }
    }
    tTot += double(calibTimes[i]);
  }

  printf("%12s:", name);
  for (unsigned i = 0; i < nEvents; ++i) {
    printf("  %lu", calibTimes[i]);
  }
  if (kernelTime == 0.f) {
    printf("  avg: %f us\n", tTot / nEvents);
  } else {
    size_t nPixels = out[0].size();
    size_t mem_size = nPixels * (sizeof(uint16_t) + sizeof(float));
    float kernelBandwidth = 1000.0f * mem_size / (1024 * 1024 * 1024) / (kernelTime / nEvents);
    printf("  avg: %f us, throughput = %.4f GB/s, Time = %.5f ms\n",
           tTot / nEvents, kernelBandwidth, kernelTime / nEvents);
  }
}

static void calibrate(float          out[],
                      uint16_t const raw[],
                      float const    pedestals[],
                      float const    gains[],
                      unsigned       nPixels)
{
  for (unsigned i = 0; i < nPixels; ++i) {
    unsigned gainMode = (raw[i] >> GainOffset) & ((1 << GainBits) - 1);
    float    datum    = raw[i] & ((1 << GainOffset) - 1);
    auto     idx      = gainMode * nPixels + i;
    out[i] = (datum - pedestals[idx]) * gains[idx];
  }
}

static void basicCalib(std::vector<vecf32_t>       out,
                       std::vector<vecu16_t> const raw,
                       std::vector<vecf32_t> const pedestals,
                       std::vector<vecf32_t> const gains,
                       std::vector<uint64_t>       calibTimes,
                       std::vector<vecf32_t> const reference)
{
  // Clear previous results to avoid confusion
  for (auto& o: out) {
    memset(o.data(), 0, o.size() * sizeof(o[0]));
  }

  auto nPixels = raw[0].size();
  std::vector<float> ps(gains.size() * nPixels);
  std::vector<float> gs(gains.size() * nPixels);
  for (unsigned i = 0; i < gains.size(); ++i) {
    for (unsigned j = 0; j < nPixels; ++j) {
      ps[i * nPixels + j] = pedestals[i][j];
      gs[i * nPixels + j] = gains[i][j];
    }
  }

  // Wait for memmory to stabilize from previous work before performing the test
  asm volatile("mfence" ::: "memory");

  // Warm-up
  calibrate(out[0].data(), raw[0].data(), ps.data(), gs.data(), nPixels);

  // Run and time the calibration test for each event
  for (unsigned i = 0; i < raw.size(); ++i) {
    auto t0{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
    // Dereferencing a vector of vectors is expensive, so pass an arrays of vectors
    calibrate(out[i].data(), raw[i].data(), ps.data(), gs.data(), nPixels);
    auto t1{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
    calibTimes[i] = std::chrono::duration_cast<us_t>(t1 - t0).count();
  }

  // Check and print some results
  check("Basic", out, reference, calibTimes);
}

static void calibrate1(float          out[],
                       uint16_t const raw[],
                       PedGain1 const pg[],
                       unsigned const nPixels)
{
  for (unsigned i = 0; i < nPixels; ++i) {
    unsigned gainMode = (raw[i] >> GainOffset) & ((1 << GainBits) - 1);
    unsigned datum    = raw[i] & ((1 << GainOffset) - 1);
    out[i] = (datum - pg[i].ped[gainMode]) * pg[i].gain[gainMode];
  }
}

static void cacheCalib1(std::vector<vecf32_t>       out,
                        std::vector<vecu16_t> const raw,
                        std::vector<vecf32_t> const pedestals,
                        std::vector<vecf32_t> const gains,
                        std::vector<uint64_t>       calibTimes,
                        std::vector<vecf32_t> const reference)
{
  // Clear previous results to avoid confusion
  for (auto& o: out) {
    memset(o.data(), 0, o.size() * sizeof(o[0]));
  }

  if (gains.size() != NGains) {
    printf("Recompile with NGains set to %zu to run the cacheCalib1 test\n",
           gains.size());
    return;
  }

  // Rearrange calibration constants in a cache friendly way
  std::vector<PedGain1> pg(raw[0].size());
  for (unsigned i = 0; i < pg.size(); ++i) {
    for (unsigned j = 0; j < gains.size(); ++j) {
      pg[i].ped[j]  = pedestals[j][i];
      pg[i].gain[j] = gains[j][i];
    }
  }

  // Wait for memmory to stabilize from previous work before performing the test
  asm volatile("mfence" ::: "memory");

  // Warm-up
  calibrate1(out[0].data(), raw[0].data(), pg.data(), pg.size());

  // Run and time the calibration test for each event
  for (unsigned i = 0; i < raw.size(); ++i) {
    auto t0{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
    calibrate1(out[i].data(), raw[i].data(), pg.data(), pg.size());
    auto t1{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
    calibTimes[i] = std::chrono::duration_cast<us_t>(t1 - t0).count();
  }

  // Check and print some results
  check("Cache_NGains", out, reference, calibTimes);
}

static void calibrate2(float          out[],
                       uint16_t const raw[],
                       PedGain2 const pedGain[],
                       unsigned const nPixels)
{
  for (unsigned i = 0; i < nPixels; ++i) {
    unsigned gainMode = (raw[i] >> GainOffset) & ((1 << GainBits) - 1);
    unsigned datum    = raw[i] & ((1 << GainOffset) - 1);
    auto&    pg       = pedGain[gainMode * nPixels + i];
    out[i] = (datum - pg.ped) * pg.gain;
  }
}

static void cacheCalib2(std::vector<vecf32_t>       out,
                        std::vector<vecu16_t> const raw,
                        std::vector<vecf32_t> const pedestals,
                        std::vector<vecf32_t> const gains,
                        std::vector<uint64_t>       calibTimes,
                        std::vector<vecf32_t> const reference)
{
  // Clear previous results to avoid confusion
  for (auto& o: out) {
    memset(o.data(), 0, o.size() * sizeof(o[0]));
  }

  // Rearrange calibration constants in a cache friendly way
  auto nPixels = raw[0].size();
  std::vector<PedGain2> pg(gains.size() * nPixels);
  for (unsigned i = 0; i < gains.size(); ++i) {
    for (unsigned j = 0; j < nPixels; ++j) {
      pg[i * nPixels + j].ped  = pedestals[i][j];
      pg[i * nPixels + j].gain = gains[i][j];
    }
  }

  // Wait for memmory to stabilize from previous work before performing the test
  asm volatile("mfence" ::: "memory");

  // Warm-up
  calibrate2(out[0].data(), raw[0].data(), pg.data(), nPixels);

  // Run and time the calibration test for each event
  for (unsigned i = 0; i < raw.size(); ++i) {
    auto t0{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
    calibrate2(out[i].data(), raw[i].data(), pg.data(), nPixels);
    auto t1{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
    calibTimes[i] = std::chrono::duration_cast<us_t>(t1 - t0).count();
  }

  // Check and print some results
  check("Cache", out, reference, calibTimes);
}

static void calibrate3(float*          out,
                       uint16_t const* raw,
                       PedGain2 const* pedGain,
                       unsigned        segSize)
{
  for (unsigned i = 0; i < segSize; ++i) {
    unsigned gainMode = (raw[i] >> GainOffset) & ((1 << GainBits) - 1);
    unsigned datum    = raw[i] & ((1 << GainOffset) - 1);
    auto&    pg       = pedGain[gainMode * segSize + i];
    out[i] = (datum - pg.ped) * pg.gain;
  }
}

static void worker(unsigned                     id,
                   std::vector<vecf32_t>&       out,
                   std::vector<vecu16_t> const& raw,
                   std::vector<PedGain2> const& pg,
                   unsigned                     segSize,
                   SPSCQueue<unsigned>&         inQueue,
                   SPSCQueue<unsigned>&         outQueue)
{
  unsigned index;
  const unsigned offset{id * segSize};
  while (true) {
    if (!inQueue.pop(index))  break;

    calibrate3(&out[index][offset],
               &raw[index][offset],
               pg.data(),
               segSize);

    outQueue.push(index);
  }
}

static void threadCalib(std::vector<vecf32_t>       out,
                        std::vector<vecu16_t> const raw,
                        std::vector<vecf32_t> const pedestals,
                        std::vector<vecf32_t> const gains,
                        std::vector<uint64_t>       calibTimes,
                        std::vector<vecf32_t> const reference,
                        unsigned                    nThreads)
{
  // Clear previous results to avoid confusion
  for (auto& o: out) {
    memset(o.data(), 0, o.size() * sizeof(o[0]));
  }

  // Determine event portion each thread is to handle
  auto nPixels = raw[0].size();
  auto segSize = nPixels / nThreads; // Need to require this to divide evenly
  if (segSize * nThreads != nPixels) {
    printf("Error: Number of threads (%u) must divide evenly into event size (%zu)\n",
           nThreads, nPixels);
    return;
  }

  // Rearrange calibration constants for each thread in a cache friendly way
  std::vector<PedGain2> pg[nThreads];
  for (unsigned i = 0; i < nThreads; ++i) {
    auto offset = i * segSize;
    pg[i].resize(gains.size() * segSize);
    for (unsigned j = 0; j < gains.size(); ++j) {
      for (unsigned k = 0; k < segSize; ++k) {
        pg[i][j * segSize + k].ped  = pedestals[j][offset + k];
        pg[i][j * segSize + k].gain = gains[j][offset + k];
      }
    }
  }

  // Set up and start worker threads with communication queues
  std::vector<std::thread> threads;
  std::vector< SPSCQueue<unsigned> > inQueues;
  std::vector< SPSCQueue<unsigned> > outQueues;
  for (unsigned i = 0; i < nThreads; ++i) {
    inQueues.emplace_back(SPSCQueue<unsigned>(raw.size()));
    outQueues.emplace_back(SPSCQueue<unsigned>(raw.size()));
  }
  for (unsigned i = 0; i < nThreads; ++i) {
    threads.emplace_back(worker, i,
                         std::ref(out),
                         std::ref(raw),
                         std::ref(pg[i]),
                         segSize,
                         std::ref(inQueues[i]),
                         std::ref(outQueues[i]));
  }

  // Wait for memmory to stabilize from previous work before performing the test
  asm volatile("mfence" ::: "memory");

  // Warm up
  for (unsigned j = 0; j < threads.size(); ++j) {
    inQueues[j].push(0);                 // Launch work for event i
  }
  for (unsigned j = 0; j < threads.size(); ++j) {
    unsigned index;
    if (outQueues[j].pop(index))  break; // Wait for event i to be completed
    if (index != 0)
      printf("Error: Thread calibrate index mismatch: got %u, expected %u\n",
             index, 0);
  }

  // Run and time the calibration test for each event
  for (unsigned i = 0; i < raw.size(); ++i) {
    auto t0{fast_monotonic_clock::now(CLOCK_MONOTONIC)};

    for (unsigned j = 0; j < threads.size(); ++j) {
      inQueues[j].push(i);                 // Launch work for event i
    }
    for (unsigned j = 0; j < threads.size(); ++j) {
      unsigned index;
      if (outQueues[j].pop(index))  break; // Wait for event i to be completed
      if (index != i)
        printf("Error: Thread calibrate index mismatch: got %u, expected %u\n",
               index, i);
    }

    auto t1{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
    calibTimes[i] = std::chrono::duration_cast<us_t>(t1 - t0).count();
  }

  // Shut down workers
  for (unsigned i = 0; i < threads.size(); ++i) {
    inQueues[i].shutdown();
    outQueues[i].shutdown();
    if (threads[i].joinable())  threads[i].join();
  }

  // Check and print some results
  check("Thread", out, reference, calibTimes);
}

static __global__ void _calibrate(float*   const        __restrict__ calibBuffers,
                                  size_t   const                     calibBufsCnt,
                                  uint16_t const* const __restrict__ rawBuffers,
                                  unsigned const&                    index,
                                  unsigned const                     panel,
                                  float    const* const __restrict__ peds_,
                                  float    const* const __restrict__ gains_,
                                  unsigned const                     nPixels)
{
  // Place the calibrated data for a given panel in the calibBuffers array at the appropiate offset
  auto const __restrict__ out = &calibBuffers[index * calibBufsCnt + panel * nPixels];
  auto const __restrict__ in  = &rawBuffers[index * calibBufsCnt]; // Raw data for the given panel
  int stride = gridDim.x * blockDim.x;
  int pixel  = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = pixel; i < nPixels; i += stride) {
    const auto gain  = (in[i] >> GainOffset) & ((1 << GainBits) - 1);
    const auto peds  = &peds_ [gain * nPixels];
    const auto gains = &gains_[gain * nPixels];
    const auto data  = in[i] & ((1 << GainOffset) - 1);
    out[i] = (data - peds[i]) * gains[i];
  }
}

static void basicCalibGpu(std::vector<vecf32_t>       out,
                          std::vector<vecu16_t> const raw,
                          std::vector<vecf32_t> const pedestals,
                          std::vector<vecf32_t> const gains,
                          std::vector<uint64_t>       calibTimes,
                          std::vector<vecf32_t> const reference,
                          unsigned                    nThreads,
                          unsigned                    nBlocks)
{
  // Set up calibrated data buffers on the GPU
  auto nEvents = out.size();
  auto nPixels = out[0].size();
  float* out_d;
  chkError(cudaMalloc(&out_d,    nEvents * nPixels * sizeof(*out_d)));
  chkError(cudaMemset( out_d, 0, nEvents * nPixels * sizeof(*out_d)));

  // Set up raw data buffers on the GPU and transfer the data
  uint16_t* raw_d;
  chkError(cudaMalloc(&raw_d, nEvents * nPixels * sizeof(*raw_d)));
  uint16_t* r_d = raw_d;
  for (unsigned i = 0; i < raw.size(); ++i) {
    chkError(cudaMemcpy(r_d, raw[i].data(), nPixels * sizeof(*r_d), cudaMemcpyHostToDevice));
    r_d += nPixels;
  }

  // Put pedestals and gains for each gain range on the GPU
  auto nGains  = gains.size();
  float* p_d;
  float* g_d;
  chkError(cudaMalloc(&p_d, nGains * nPixels * sizeof(*p_d)));
  chkError(cudaMalloc(&g_d, nGains * nPixels * sizeof(*g_d)));
  auto peds_d  = p_d;
  auto gains_d = g_d;
  for (unsigned gn = 0; gn < nGains; ++gn) {
    chkError(cudaMemcpy(p_d, pedestals[gn].data(), nPixels * sizeof(*p_d), cudaMemcpyHostToDevice));
    chkError(cudaMemcpy(g_d, gains[gn].data(),     nPixels * sizeof(*g_d), cudaMemcpyHostToDevice));
    p_d += nPixels;
    g_d += nPixels;
  }

  // Manage an event buffer index similar to the DAQ
  unsigned* index_d;
  chkError(cudaMalloc(&index_d,    sizeof(index_d)));
  chkError(cudaMemset( index_d, 0, sizeof(index_d)));

  // Set up a cuda stream in which to do asynchronous work like in the DAQ
  cudaStream_t stream;
  chkFatal(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  //const unsigned chunks{128};       // Number of pixels handled per thread
  const unsigned tpb   {nThreads};  // Threads per block
  //const unsigned bpg   {(unsigned(nPixels) + chunks * tpb - 1) / (chunks * tpb)}; // Blocks per grid
  const unsigned bpg   {nBlocks};   // Blocks per grid
  const unsigned panel {0};

  // Initialize CUDA events
  cudaEvent_t start, stop;
  chkError(cudaEventCreate(&start));
  chkError(cudaEventCreate(&stop));

  // Wait for memmory to stabilize from previous work before performing the test
  asm volatile("mfence" ::: "memory");

  // Warm-up
  _calibrate<<<bpg, tpb, 0, stream>>>(out_d, nPixels, raw_d, *index_d, panel, peds_d, gains_d, nPixels);
  chkError(cudaStreamSynchronize(stream)); // Wait for completion of each event

  // Take measurements for loop over kernel launches
  chkError(cudaEventRecord(start, 0));

  // Run and time the calibration test for each event
  for (unsigned i = 0; i < raw.size(); ++i) {
    auto t0{fast_monotonic_clock::now(CLOCK_MONOTONIC)};

    chkError(cudaMemcpyAsync(index_d, &i, sizeof(i), cudaMemcpyHostToDevice, stream));

    _calibrate<<<bpg, tpb, 0, stream>>>(out_d, nPixels, raw_d, *index_d, panel, peds_d, gains_d, nPixels);
    chkError(cudaStreamSynchronize(stream)); // Wait for completion of each event

    auto t1{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
    calibTimes[i] = std::chrono::duration_cast<us_t>(t1 - t0).count();
  }

  chkError(cudaEventRecord(stop, 0));
  chkError(cudaEventSynchronize(stop));
  float kernelTime;
  chkError(cudaEventElapsedTime(&kernelTime, start, stop));

  // Retrieve the calibrated data from the GPU
  float* o_d = out_d;
  for (unsigned i = 0; i < raw.size(); ++i) {
    chkError(cudaMemcpy(out[i].data(), o_d, nPixels * sizeof(*out[i].data()), cudaMemcpyDeviceToHost));
    o_d += nPixels;
  }
  chkError(cudaDeviceSynchronize());

  // Check and print some results
  check("Basic_GPU", out, reference, calibTimes, kernelTime);

  // Clean up
  chkError(cudaStreamDestroy(stream));

  chkError(cudaFree(index_d));

  chkError(cudaFree(gains_d));
  chkError(cudaFree(peds_d));

  chkError(cudaFree(raw_d));
  chkError(cudaFree(out_d));
}

static __global__ void _calibrate2(float*   const        __restrict__ calibBuffers,
                                   size_t   const                     calibBufsCnt,
                                   uint16_t const* const __restrict__ rawBuffers,
                                   unsigned const&                    index,
                                   unsigned const                     panel,
                                   PedGain2 const* const __restrict__ pedGains,
                                   unsigned const                     nPixels)
{
  // Place the calibrated data for a given panel in the calibBuffers array at the appropiate offset
  auto const __restrict__ out = &calibBuffers[index * calibBufsCnt + panel * nPixels];
  auto const __restrict__ in  = &rawBuffers[index * calibBufsCnt]; // Raw data for the given panel
  int stride = gridDim.x * blockDim.x;
  int pixel  = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = pixel; i < nPixels; i += stride) {
    auto        datum = in[i];
    const auto  gain  = (datum >> GainOffset) & ((1 << GainBits) - 1);
    const auto& pg    = pedGains[gain * nPixels + i];
    datum &= ((1 << GainOffset) - 1);
    out[i] = (datum - pg.ped) * pg.gain;
  }
}

static void cacheCalibGpu(std::vector<vecf32_t>       out,
                          std::vector<vecu16_t> const raw,
                          std::vector<vecf32_t> const pedestals,
                          std::vector<vecf32_t> const gains,
                          std::vector<uint64_t>       calibTimes,
                          std::vector<vecf32_t> const reference,
                          unsigned                    nThreads,
                          unsigned                    nBlocks)
{
  // Rearrange calibration constants in a cache friendly way
  std::vector<PedGain2> pg[gains.size()];
  for (unsigned i = 0; i < gains.size(); ++i) {
    pg[i].resize(raw[0].size());
    for (unsigned j = 0; j < raw[0].size(); ++j) {
      pg[i][j].ped  = pedestals[i][j];
      pg[i][j].gain = gains[i][j];
    }
  }

  // Set up calibrated data buffers on the GPU
  auto nEvents = out.size();
  auto nPixels = out[0].size();
  float* out_d;
  chkError(cudaMalloc(&out_d,    nEvents * nPixels * sizeof(*out_d)));
  chkError(cudaMemset( out_d, 0, nEvents * nPixels * sizeof(*out_d)));

  // Set up raw data buffers on the GPU and transfer the data
  uint16_t* raw_d;
  chkError(cudaMalloc(&raw_d, nEvents * nPixels * sizeof(*raw_d)));
  uint16_t* r_d = raw_d;
  for (unsigned i = 0; i < raw.size(); ++i) {
    chkError(cudaMemcpy(r_d, raw[i].data(), nPixels * sizeof(*r_d), cudaMemcpyHostToDevice));
    r_d += nPixels;
  }

  // Put pedestals and gains for each gain range on the GPU
  auto nGains  = gains.size();
  PedGain2* pg_d;
  chkError(cudaMalloc(&pg_d, nGains * nPixels * sizeof(*pg_d)));
  auto pedGains_d = pg_d;
  for (unsigned gn = 0; gn < nGains; ++gn) {
    chkError(cudaMemcpy(pg_d, pg[gn].data(), nPixels * sizeof(*pg_d), cudaMemcpyHostToDevice));
    pg_d += nPixels;
  }

  // Manage an event buffer index similar to the DAQ
  unsigned* index_d;
  chkError(cudaMalloc(&index_d,    sizeof(index_d)));
  chkError(cudaMemset( index_d, 0, sizeof(index_d)));

  // Set up a cuda stream in which to do asynchronous work like in the DAQ
  cudaStream_t stream;
  chkFatal(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  //const unsigned chunks{128};       // Number of pixels handled per thread
  const unsigned tpb   {nThreads};  // Threads per block
  //const unsigned bpg   {(unsigned(nPixels) + chunks * tpb - 1) / (chunks * tpb)}; // Blocks per grid
  const unsigned bpg   {nBlocks};   // Blocks per grid

  // Initialize CUDA events
  cudaEvent_t start, stop;
  chkError(cudaEventCreate(&start));
  chkError(cudaEventCreate(&stop));

  // Wait for memmory to stabilize from previous work before performing the test
  asm volatile("mfence" ::: "memory");

  // Warm-up
  const unsigned panel {0};
  _calibrate2<<<bpg, tpb, 0, stream>>>(out_d, nPixels, raw_d, *index_d, panel, pedGains_d, nPixels);
  chkError(cudaStreamSynchronize(stream)); // Wait for completion of each event

  // Take measurements for loop over kernel launches
  chkError(cudaEventRecord(start, 0));

  // Run and time the calibration test for each event
  for (unsigned i = 0; i < raw.size(); ++i) {
    auto t0{fast_monotonic_clock::now(CLOCK_MONOTONIC)};

    chkError(cudaMemcpyAsync(index_d, &i, sizeof(i), cudaMemcpyHostToDevice, stream));

    _calibrate2<<<bpg, tpb, 0, stream>>>(out_d, nPixels, raw_d, *index_d, panel, pedGains_d, nPixels);
    chkError(cudaStreamSynchronize(stream)); // Wait for completion of each event

    auto t1{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
    calibTimes[i] = std::chrono::duration_cast<us_t>(t1 - t0).count();
  }

  chkError(cudaEventRecord(stop, 0));
  chkError(cudaEventSynchronize(stop));
  float kernelTime;
  chkError(cudaEventElapsedTime(&kernelTime, start, stop));

  // Retrieve the calibrated data from the GPU
  float* o_d = out_d;
  for (unsigned i = 0; i < raw.size(); ++i) {
    chkError(cudaMemcpy(out[i].data(), o_d, nPixels * sizeof(*out[i].data()), cudaMemcpyDeviceToHost));
    o_d += nPixels;
  }
  chkError(cudaDeviceSynchronize());

  // Check and print some results
  check("Cache_GPU", out, reference, calibTimes, kernelTime);

  // Clean up
  chkError(cudaStreamDestroy(stream));

  chkError(cudaFree(index_d));

  chkError(cudaFree(pedGains_d));

  chkError(cudaFree(raw_d));
  chkError(cudaFree(out_d));
}

static __global__ void _calibrate3(float*   const        __restrict__ calibBuffers,
                                   size_t   const                     calibBufsCnt,
                                   uint16_t const* const __restrict__ rawBuffers,
                                   unsigned const&                    index,
                                   unsigned const                     panel,
                                   float    const* const __restrict__ peds,
                                   float    const* const __restrict__ gains,
                                   unsigned const                     nPixels)
{
  // Place the calibrated data for a given panel in the calibBuffers array at the appropriate offset
  auto const __restrict__ out = &calibBuffers[index * calibBufsCnt + panel * nPixels];
  auto const __restrict__ in  = &rawBuffers[index * calibBufsCnt]; // Raw data for the given panel
  int stride = gridDim.x * blockDim.x;
  int pixel  = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ uint8_t smem[];
  //auto sIn    = (uint16_t*)smem;                      // uint16_t sIn[TPB];
  auto sPeds  = (float*)smem; //&sIn[blockDim.x];             // float    sPeds[1 << GainBits][TPB];
  auto sGains = &sPeds[(1 << GainBits) * blockDim.x]; // float    sGains[1 << GainBits][TPB];

  cg::thread_block cta = cg::this_thread_block();
  for (int i = pixel; i < nPixels; i += stride) {
    //sIn[threadIdx.x] = in[i];  // Seems like this wouldn't help due to single access
    //__syncwarp();
    //__syncthreads();
    //cg::sync(cta);
    auto       datum = in[i]; //sIn[threadIdx.x];
    const auto range = (datum >> GainOffset) & ((1 << GainBits) - 1);
    const auto pgIdx = (range * blockDim.x) + threadIdx.x;
    sPeds[pgIdx]  = peds[range * nPixels + i];
    sGains[pgIdx] = gains[range * nPixels + i];
    //__syncwarp();
    //__syncthreads();
    cg::sync(cta);
    datum &= ((1 << GainOffset) - 1);
    out[i] = (datum - sPeds[pgIdx]) * sGains[pgIdx];
  }
}

static void shmemCalibGpu(std::vector<vecf32_t>       out,
                          std::vector<vecu16_t> const raw,
                          std::vector<vecf32_t> const pedestals,
                          std::vector<vecf32_t> const gains,
                          std::vector<uint64_t>       calibTimes,
                          std::vector<vecf32_t> const reference,
                          unsigned                    nThreads,
                          unsigned                    nBlocks)
{
  // Set up calibrated data buffers on the GPU
  auto nEvents = out.size();
  auto nPixels = out[0].size();
  float* out_d;
  chkError(cudaMalloc(&out_d,    nEvents * nPixels * sizeof(*out_d)));
  chkError(cudaMemset( out_d, 0, nEvents * nPixels * sizeof(*out_d)));

  // Set up raw data buffers on the GPU and transfer the data
  uint16_t* raw_d;
  chkError(cudaMalloc(&raw_d, nEvents * nPixels * sizeof(*raw_d)));
  uint16_t* r_d = raw_d;
  for (unsigned i = 0; i < raw.size(); ++i) {
    chkError(cudaMemcpy(r_d, raw[i].data(), nPixels * sizeof(*r_d), cudaMemcpyHostToDevice));
    r_d += nPixels;
  }

  // Put pedestals and gains for each gain range on the GPU
  auto nGains  = gains.size();
  float* p_d;
  float* g_d;
  chkError(cudaMalloc(&p_d, nGains * nPixels * sizeof(*p_d)));
  chkError(cudaMalloc(&g_d, nGains * nPixels * sizeof(*g_d)));
  auto peds_d  = p_d;
  auto gains_d = g_d;
  for (unsigned gn = 0; gn < nGains; ++gn) {
    chkError(cudaMemcpy(p_d, pedestals[gn].data(), nPixels * sizeof(*p_d), cudaMemcpyHostToDevice));
    chkError(cudaMemcpy(g_d, gains[gn].data(),     nPixels * sizeof(*g_d), cudaMemcpyHostToDevice));
    p_d += nPixels;
    g_d += nPixels;
  }

  // Manage an event buffer index similar to the DAQ
  unsigned* index_d;
  chkError(cudaMalloc(&index_d,    sizeof(index_d)));
  chkError(cudaMemset( index_d, 0, sizeof(index_d)));

  // Set up a cuda stream in which to do asynchronous work like in the DAQ
  cudaStream_t stream;
  chkFatal(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const auto tpb{nThreads};             // Threads per block
  const auto bpg{nBlocks};              // Blocks per grid
  const auto shSize{tpb * (/*sizeof(uint16_t) +*/ (1 << GainBits) * (sizeof(*peds_d) + sizeof(*gains_d)))};

  // Initialize CUDA events
  cudaEvent_t start, stop;
  chkError(cudaEventCreate(&start));
  chkError(cudaEventCreate(&stop));

  // Wait for memmory to stabilize from previous work before performing the test
  asm volatile("mfence" ::: "memory");

  // Warm-up
  const unsigned panel {0};             // We use only one panel of data in this test
  _calibrate3<<<bpg, tpb, shSize, stream>>>(out_d, nPixels, raw_d, *index_d, panel, peds_d, gains_d, nPixels);
  chkError(cudaStreamSynchronize(stream)); // Wait for completion of each event

  // Take measurements for loop over kernel launches
  chkError(cudaEventRecord(start, 0));

  // Run and time the calibration test for each event
  for (unsigned i = 0; i < raw.size(); ++i) {
    auto t0{fast_monotonic_clock::now(CLOCK_MONOTONIC)};

    chkError(cudaMemcpyAsync(index_d, &i, sizeof(i), cudaMemcpyHostToDevice, stream));

    _calibrate3<<<bpg, tpb, shSize, stream>>>(out_d, nPixels, raw_d, *index_d, panel, peds_d, gains_d, nPixels);
    chkError(cudaStreamSynchronize(stream)); // Wait for completion of each event

    auto t1{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
    calibTimes[i] = std::chrono::duration_cast<us_t>(t1 - t0).count();
  }

  chkError(cudaEventRecord(stop, 0));
  chkError(cudaEventSynchronize(stop));
  float kernelTime;
  chkError(cudaEventElapsedTime(&kernelTime, start, stop));

  // Retrieve the calibrated data from the GPU
  float* o_d = out_d;
  for (unsigned i = 0; i < raw.size(); ++i) {
    chkError(cudaMemcpy(out[i].data(), o_d, nPixels * sizeof(*out[i].data()), cudaMemcpyDeviceToHost));
    o_d += nPixels;
  }
  chkError(cudaDeviceSynchronize());

  // Check and print some results
  check("shmem_GPU", out, reference, calibTimes, kernelTime);

  // Clean up
  chkError(cudaStreamDestroy(stream));

  chkError(cudaFree(index_d));

  chkError(cudaFree(gains_d));
  chkError(cudaFree(peds_d));

  chkError(cudaFree(raw_d));
  chkError(cudaFree(out_d));
}

static __global__ void _calibrate4(float*   const        __restrict__ calibBuffers,
                                   size_t   const                     calibBufsCnt,
                                   uint16_t const* const __restrict__ rawBuffers,
                                   unsigned const&                    index,
                                   unsigned const                     panel,
                                   PedGain2 const* const __restrict__ pedGains,
                                   unsigned const                     nPixels)
{
  // Place the calibrated data for a given panel in the calibBuffers array at the appropriate offset
  auto const __restrict__ out = &calibBuffers[index * calibBufsCnt + panel * nPixels];
  auto const __restrict__ in  = &rawBuffers[index * calibBufsCnt]; // Raw data for the given panel
  int stride = gridDim.x * blockDim.x;
  int pixel  = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ uint8_t smem[];
  //auto sIn       = (uint16_t*)smem;                      // uint16_t sIn[TPB];
  auto sPedGains = (PedGain2*)smem; //&sIn[blockDim.x];           // PedGain2  sPedGains[1 << GainBits][TPB];

  cg::thread_block cta = cg::this_thread_block();
  for (int i = pixel; i < nPixels; i += stride) {
    //sIn[threadIdx.x] = in[i];  // Seems like this wouldn't help due to single access
    //__syncwarp();
    //__syncthreads();
    //cg::sync(cta);
    auto       datum = in[i]; //sIn[threadIdx.x];
    const auto range = (datum >> GainOffset) & ((1 << GainBits) - 1);
    const auto pgIdx = (range * blockDim.x) + threadIdx.x;
    sPedGains[pgIdx] = pedGains[range * nPixels + i];
    //__syncwarp();
    //__syncthreads();
    cg::sync(cta);
    datum &= ((1 << GainOffset) - 1);
    out[i] = (datum - sPedGains[pgIdx].ped) * sPedGains[pgIdx].gain;
  }
}

static void shmemPgCalibGpu(std::vector<vecf32_t>       out,
                            std::vector<vecu16_t> const raw,
                            std::vector<vecf32_t> const pedestals,
                            std::vector<vecf32_t> const gains,
                            std::vector<uint64_t>       calibTimes,
                            std::vector<vecf32_t> const reference,
                            unsigned                    nThreads,
                            unsigned                    nBlocks)
{
  // Rearrange calibration constants in a cache friendly way
  std::vector<PedGain2> pg[gains.size()];
  for (unsigned i = 0; i < gains.size(); ++i) {
    pg[i].resize(raw[0].size());
    for (unsigned j = 0; j < raw[0].size(); ++j) {
      pg[i][j].ped  = pedestals[i][j];
      pg[i][j].gain = gains[i][j];
    }
  }

  // Set up calibrated data buffers on the GPU
  auto nEvents = out.size();
  auto nPixels = out[0].size();
  float* out_d;
  chkError(cudaMalloc(&out_d,    nEvents * nPixels * sizeof(*out_d)));
  chkError(cudaMemset( out_d, 0, nEvents * nPixels * sizeof(*out_d)));

  // Set up raw data buffers on the GPU and transfer the data
  uint16_t* raw_d;
  chkError(cudaMalloc(&raw_d, nEvents * nPixels * sizeof(*raw_d)));
  uint16_t* r_d = raw_d;
  for (unsigned i = 0; i < raw.size(); ++i) {
    chkError(cudaMemcpy(r_d, raw[i].data(), nPixels * sizeof(*r_d), cudaMemcpyHostToDevice));
    r_d += nPixels;
  }

  // Put pedestals and gains for each gain range on the GPU
  auto nGains  = gains.size();
  PedGain2* pg_d;
  chkError(cudaMalloc(&pg_d, nGains * nPixels * sizeof(*pg_d)));
  auto pedGains_d = pg_d;
  for (unsigned gn = 0; gn < nGains; ++gn) {
    chkError(cudaMemcpy(pg_d, pg[gn].data(), nPixels * sizeof(*pg_d), cudaMemcpyHostToDevice));
    pg_d += nPixels;
  }

  // Manage an event buffer index similar to the DAQ
  unsigned* index_d;
  chkError(cudaMalloc(&index_d,    sizeof(index_d)));
  chkError(cudaMemset( index_d, 0, sizeof(index_d)));

  // Set up a cuda stream in which to do asynchronous work like in the DAQ
  cudaStream_t stream;
  chkFatal(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const auto tpb{nThreads};             // Threads per block
  const auto bpg{nBlocks};              // Blocks per grid
  const auto shSize{tpb * (/*sizeof(uint16_t) +*/ (1 << GainBits) * sizeof(*pedGains_d))};

  // Initialize CUDA events
  cudaEvent_t start, stop;
  chkError(cudaEventCreate(&start));
  chkError(cudaEventCreate(&stop));

  // Wait for memmory to stabilize from previous work before performing the test
  asm volatile("mfence" ::: "memory");

  // Warm-up
  const unsigned panel {0};             // We use only one panel of data in this test
  _calibrate4<<<bpg, tpb, shSize, stream>>>(out_d, nPixels, raw_d, *index_d, panel, pedGains_d, nPixels);
  chkError(cudaStreamSynchronize(stream)); // Wait for completion of each event

  // Take measurements for loop over kernel launches
  chkError(cudaEventRecord(start, 0));

  // Run and time the calibration test for each event
  for (unsigned i = 0; i < raw.size(); ++i) {
    auto t0{fast_monotonic_clock::now(CLOCK_MONOTONIC)};

    chkError(cudaMemcpyAsync(index_d, &i, sizeof(i), cudaMemcpyHostToDevice, stream));

    _calibrate4<<<bpg, tpb, shSize, stream>>>(out_d, nPixels, raw_d, *index_d, panel, pedGains_d, nPixels);
    chkError(cudaStreamSynchronize(stream)); // Wait for completion of each event

    auto t1{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
    calibTimes[i] = std::chrono::duration_cast<us_t>(t1 - t0).count();
  }

  chkError(cudaEventRecord(stop, 0));
  chkError(cudaEventSynchronize(stop));
  float kernelTime;
  chkError(cudaEventElapsedTime(&kernelTime, start, stop));

  // Retrieve the calibrated data from the GPU
  float* o_d = out_d;
  for (unsigned i = 0; i < raw.size(); ++i) {
    chkError(cudaMemcpy(out[i].data(), o_d, nPixels * sizeof(*out[i].data()), cudaMemcpyDeviceToHost));
    o_d += nPixels;
  }
  chkError(cudaDeviceSynchronize());

  // Check and print some results
  check("shmem_PG_GPU", out, reference, calibTimes, kernelTime);

  // Clean up
  chkError(cudaStreamDestroy(stream));

  chkError(cudaFree(index_d));

  chkError(cudaFree(pedGains_d));

  chkError(cudaFree(raw_d));
  chkError(cudaFree(out_d));
}


int main(int argc, char **argv)
{
  unsigned nEvents{NEvents};
  unsigned nGains{NGains};
  unsigned nPixels{NPixels};
  unsigned nThreads{2};
  unsigned nGpuThreads{256};
  unsigned nBlocks{6};
  int c;
  while((c = getopt(argc, argv, "n:g:p:t:T:B:")) != EOF) {
    switch(c) {
      case 'n':  nEvents     = std::stoi(optarg);  break;
      case 'g':  nGains      = std::stoi(optarg);  break;
      case 'p':  nPixels     = std::stoi(optarg);  break;
      case 't':  nThreads    = std::stoi(optarg);  break;
      case 'T':  nGpuThreads = std::stoi(optarg);  break;
      case 'B':  nBlocks     = std::stoi(optarg);  break;
      default: {
        printf("%s [-n <nEvents (%u)>] [-g <nGains (%u)>] [-p <nPixels (%u)>] "
               "[t <nThreads (%u)] [t <nBlocks (%u)]\n",
               argv[0], nEvents, nGains, nPixels, nThreads, nBlocks);
        return 1;
      }
    }
  }

  if (nGains > (1 << GainBits)) {
    printf("Error: Insufficient bits (%u) for nGains (%u)\n", GainBits, nGains);
    return 1;
  }
  if (nThreads > std::thread::hardware_concurrency()) {
    printf("Error: nThreads (%u) is greater than max supported (%u)\n",
           nThreads, std::thread::hardware_concurrency());
    return 1;
  }

  srand(2025);                          // Set seed

  printf("Running %s with -n %u, -g %u, -p %u, -t %u, -T %u, -B %u\n\n",
         argv[0], nEvents, nGains, nPixels, nThreads, nGpuThreads, nBlocks);

  // Generate pedestals and gains
  std::vector<vecf32_t> pedestals(nGains);
  std::vector<vecf32_t> gains(nGains);
  for (unsigned i = 0; i < nGains; ++i) {
    pedestals[i].resize(nPixels);
    gains[i].resize(nPixels);

    for (unsigned j = 0; j < pedestals[i].size(); ++j) {
      pedestals[i][j] = (float(rand()) / float(RAND_MAX)) * (1 << GainOffset);
    }
    for (unsigned j = 0; j < gains[i].size(); ++j) {
      gains[i][j] = (float(rand()) / float(RAND_MAX)) * (1 << GainOffset); // What's appropriate here?
    }
  }

  // Determine the processing resources to use.  Slightly better times seem to
  // be achieved when nPixels/stride is an integer.  Adjusting nBlocks for this
  // might lead to a partially used SM.  Aim for maximum occupancy.
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const auto tpMP{prop.maxThreadsPerMultiProcessor};
  const auto stride{nBlocks * nGpuThreads};
  printf("GPU threads per SM: %d, total threads: %u, SMs %.1f, pixels per thread: %.1f\n",
         tpMP, stride, float(stride) / tpMP, float(nPixels) / stride);

  const auto nSMs{(nBlocks * nGpuThreads + tpMP-1) / tpMP};
  const auto shSize{nGpuThreads * (/*sizeof(uint16_t) +*/ (1 << GainBits) * (sizeof(pedestals[0][0]) + sizeof(gains[0][0])))};
  printf("GPU shared memory use size per block: %zu B, per SM: %zu B\n\n", shSize, nBlocks * shSize/nSMs);

  // Generate data for nEvents events
  std::vector<vecu16_t> raw(nEvents);
  std::vector<uint64_t> nGainEvts(nGains, 0);
  for (unsigned i = 0; i < nEvents; ++i) {
    raw[i].resize(nPixels);

    for (unsigned j = 0; j < raw[i].size(); ++j) {
      auto datum = uint16_t((float(rand()) / float(RAND_MAX)) * (1 << GainOffset));
      auto range = (float(rand()) / float(RAND_MAX));
      unsigned mode;
      if      (range <= (5.0/6.0))  mode = 0;
      else if (range <= 1.0)   mode = 1;
      //else if (range <= 0.75)  mode = 2;
      //else if (range <= 1.0)   mode = 3;
      else {
        printf("*** Bad range value %f\n", range);
        mode = 3;
      }
      raw[i][j] = (mode << GainOffset) | datum;
      ++nGainEvts[mode];
    }
  }
  printf("Number of pixels generated for each of %zu gain ranges:\n", nGainEvts.size());
  uint64_t total = 0;
  for (unsigned i = 0; i < nGainEvts.size(); ++i) {
    printf("  %9lu", nGainEvts[i]);
    total += nGainEvts[i];
  }
  printf("  total %lu\n", total);
  double sum = 0.0;
  for (unsigned i = 0; i < nGainEvts.size(); ++i) {
    printf("  %9f", double(nGainEvts[i]) / double(total));
    sum += double(nGainEvts[i]) / double(total);
  }
  printf("  total %f\n\n", sum);

  // Generate calibrated events for reference
  std::vector<vecf32_t> calib(nEvents);
  std::vector<vecf32_t> reference(nEvents);
  for (unsigned i = 0; i < nEvents; ++i) {
    calib[i].resize(nPixels);
    reference[i].resize(nPixels);
    for (unsigned j = 0; j < reference[i].size(); ++j) {
      auto gainMode = (raw[i][j] >> GainOffset) & ((1 << GainBits) - 1);
      auto datum    = float(raw[i][j] & ((1 << GainOffset) - 1));
      reference[i][j] = (datum - pedestals[gainMode][j]) * gains[gainMode][j];
    }
  }

  // Do the trial calibrations
  std::vector<uint64_t> calibTimes(nEvents, 0);
  basicCalib(calib, raw, pedestals, gains, calibTimes, reference);
  cacheCalib1(calib, raw, pedestals, gains, calibTimes, reference);
  cacheCalib2(calib, raw, pedestals, gains, calibTimes, reference);
  threadCalib(calib, raw, pedestals, gains, calibTimes, reference, nThreads);
  basicCalibGpu(calib, raw, pedestals, gains, calibTimes, reference, nGpuThreads, nBlocks);
  cacheCalibGpu(calib, raw, pedestals, gains, calibTimes, reference, nGpuThreads, nBlocks);
  shmemCalibGpu(calib, raw, pedestals, gains, calibTimes, reference, nGpuThreads, nBlocks);
  shmemPgCalibGpu(calib, raw, pedestals, gains, calibTimes, reference, nGpuThreads, nBlocks);

  return 0;
}
