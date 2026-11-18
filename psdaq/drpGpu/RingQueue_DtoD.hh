#ifndef RINGQUEUE_DTOD_HH
#define RINGQUEUE_DTOD_HH

#include <atomic>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda/std/atomic>

#include "gpuUtils.hh"                 // hostAlloc

namespace Drp {
  namespace Gpu {

template <typename T>
class RingQueueDtoD
{
public:
  __host__ RingQueueDtoD(const unsigned capacity) :
    m_head_h       (nullptr),
    m_tail_h       (nullptr),
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
  }

  __host__ ~RingQueueDtoD()
  {
    if (m_ringBuffer_h)  chkError(hostFree(m_ringBuffer_h));
    if (m_tail_h)        chkError(hostFree(m_tail_h));
    if (m_head_h)        chkError(hostFree(m_head_h));
  }

  __device__ bool push(const T& value) const           /** Store value at head and advance when not full */
  {
    using namespace cuda::std;
    auto tail = m_tail_h->load(memory_order_acquire);
    auto head = m_head_h->load(memory_order_acquire);
    auto next = (head+1) & m_capacityMask;
    if (next == tail)  return false;                   // Full: caller retries to wait for tail to advance
    m_ringBuffer_h[head] = value;                      // Store value _before_ signaling it is available
    m_head_h->store(next, memory_order_release);       // Publish new head
    return true;
  }

  __device__ bool pop(T* const __restrict__ value) const /** Fetch value at tail and advance when not empty */
  {
    using namespace cuda::std;
    auto tail = m_tail_h->load(memory_order_acquire);
    auto head = m_head_h->load(memory_order_acquire);
    if (tail == head)  return false;                   // Empty: caller retries to wait for head to advance
    *value = m_ringBuffer_h[tail];                     // Fetch value _before_ signaling it is available
    auto next = (tail+1) & m_capacityMask;
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
    using namespace cuda::std;
    return m_tail_h->load(memory_order_acquire);
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
  }

private:
  cuda::std::atomic<unsigned>* m_head_h; // Must stay coherent across device and host
  cuda::std::atomic<unsigned>* m_tail_h; // Must stay coherent across device and host
  const unsigned               m_capacityMask;
  T*                           m_ringBuffer_h;
};

  } // Gpu
} // Drp

#endif // RINGQUEUE_DTOD_HH
