#ifndef RINGQUEUE_HTOD_HH
#define RINGQUEUE_HTOD_HH

#include <atomic>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cuda/std/atomic>

#include "gpuUtils.hh"                 // hostAlloc

namespace Drp {
  namespace Gpu {

template <typename T>
class RingQueueHtoD
{
public:
  __host__ RingQueueHtoD(const unsigned capacity) :
    m_head_h       (nullptr),
    m_tail_h       (nullptr),
    m_tail_d       (nullptr),
    m_head_d       (nullptr),
    m_capacityMask (capacity-1),    // Range of the buffer queue [0, capacity-1]
    m_ringBuffer_h (nullptr)
  {
    assert((capacity & (capacity - 1)) == 0);  // Capacity must be a power of 2

    // The ring is empty when head == tail
    // Head points to the next index to be allocated
    chkError(hostAlloc(&m_head_h, sizeof(*m_head_h)));
    *m_head_h = 0;
    // Tail points to the next index after the last freed one
    chkError(hostAlloc(&m_tail_h, sizeof(*m_tail_h)));
    *m_tail_h = 0;

    // Use pinned memory for the ring buffer for low latency access from both Host and Device
    chkError(hostAlloc(&m_ringBuffer_h, capacity * sizeof(*m_ringBuffer_h)));

    chkError(cudaMalloc(&m_tail_d, sizeof(*m_tail_d)));
    chkError(cudaMalloc(&m_head_d, sizeof(*m_head_d)));
    _resetDevice();
  }

  __host__ ~RingQueueHtoD()
  {
    if (m_head_d)        chkError(cudaFree(m_head_d));
    if (m_tail_d)        chkError(cudaFree(m_tail_d));
    if (m_ringBuffer_h)  chkError(hostFree(m_ringBuffer_h));
    if (m_tail_h)        chkError(hostFree(m_tail_h));
    if (m_head_h)        chkError(hostFree(m_head_h));
  }

  __host__ bool push(const T& value) const             /** Store value at head and advance when not full */
  {
    using namespace cuda::std;
    auto tail = m_tail_h->load(memory_order_acquire);
    auto head = m_head_h->load(memory_order_acquire);
    auto next = (head+1) & m_capacityMask;
    if (next == tail)  return false;                   // Full: caller retries to wait for tail to advance
    asm volatile("mfence" ::: "memory");               // Avoid reordering of the head store and the tail load
    m_ringBuffer_h[head] = value;                      // Store value _before_ signaling it is available
    m_head_h->store(next, memory_order_release);       // Publish new head
    return true;
  }

  // Reads the host's head across PCIe only when the cached copy says the ring is empty
  __device__ bool pop(T* const __restrict__ value) const /** Fetch value at tail and advance when not empty */
  {
    using namespace cuda;
    auto tail = m_tail_d->load(memory_order_acquire);
    if (tail == m_head_d->load(memory_order_relaxed)) {
      auto head = m_head_h->load(memory_order_acquire);
      if (tail == head)  return false;                 // Empty: caller retries to wait for head to advance
      m_head_d->store(head, memory_order_relaxed);
    }
    *value = m_ringBuffer_h[tail];                     // Fetch value _before_ signaling it is available
    auto next = (tail+1) & m_capacityMask;
    m_tail_d->store(next, memory_order_release);       // Orders the head cache for the next launch
    m_tail_h->store(next, memory_order_release);       // Publish new tail
    return true;
  }

  __host__ __device__ unsigned head() const
  {
    using namespace cuda::std;
    return m_head_h->load(memory_order_acquire);
  }

  __host__ __device__ unsigned tail() const
  {
#ifdef __CUDA_ARCH__
    return m_tail_d->load(cuda::memory_order_acquire);
#else
    return m_tail_h->load(cuda::std::memory_order_acquire);
#endif
  }

  __host__ __device__ unsigned occupancy() const
  {
    using namespace cuda::std;
    auto head = m_head_h->load(memory_order_acquire);
    auto tail = m_tail_h->load(memory_order_acquire);
    return (head - tail) & m_capacityMask;
  }

  __host__ __device__ size_t size() const
  {
    return m_capacityMask + 1;
  }

  __host__ void reset()
  {
    *m_head_h = 0;
    *m_tail_h = 0;
    _resetDevice();
  }

private:
  __host__ void _resetDevice()
  {
    static_assert(sizeof(*m_tail_d) == sizeof(unsigned));
    chkError(cudaMemset(m_tail_d, 0, sizeof(*m_tail_d)));
    chkError(cudaMemset(m_head_d, 0, sizeof(*m_head_d))); // Equal to the tail, so forces a refresh
  }

private:
  using DevIndex = cuda::atomic<unsigned, cuda::thread_scope_device>;

  cuda::std::atomic<unsigned>* m_head_h; // Must stay coherent across device and host
  cuda::std::atomic<unsigned>* m_tail_h; // Must stay coherent across device and host
  DevIndex*                    m_tail_d; // Device copy of m_tail_h, written only by pop
  DevIndex*                    m_head_d; // Lagging cache of m_head_h, so pop rarely crosses PCIe
  const unsigned               m_capacityMask;
  T*                           m_ringBuffer_h;
};

  } // Gpu
} // Drp

#endif // RINGQUEUE_HTOD_HH
