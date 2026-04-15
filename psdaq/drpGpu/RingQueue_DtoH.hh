#ifndef RINGQUEUE_DTOH_HH
#define RINGQUEUE_DTOH_HH

#include <atomic>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda/std/atomic>

namespace Drp {
  namespace Gpu {

template <typename T>
class RingQueueDtoH
{
public:
  __host__ RingQueueDtoH(const unsigned capacity) :
    m_head_h      (nullptr),
    m_head_d      (nullptr),
    m_tail_h      (nullptr),
    m_tail_d      (nullptr),
    m_capacityMask(capacity-1),    // Range of the buffer queue [0, capacity-1]
    m_ringBuffer_h(nullptr),
    m_ringBuffer_d(nullptr)
  {
    assert(capacity & (capacity - 1));  // Capacity must be a power of 2

    // The ring is empty when head == tail
    // Head points to the next index to be allocated
    chkError(cudaHostAlloc(&m_head_h, sizeof(*m_head_h), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_head_d, m_head_h, 0));
    *m_head_h = 0;
    // Tail points to the next index after the last freed one
    chkError(cudaHostAlloc(&m_tail_h, sizeof(*m_tail_h), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_tail_d, m_tail_h, 0));
    *m_tail_h = 0;

    // Use pinned memory for the ring buffer for low latency access from both Host and Device
    chkError(cudaHostAlloc(&m_ringBuffer_h, capacity * sizeof(*m_ringBuffer_h), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_ringBuffer_d, m_ringBuffer_h, 0));
  }

  __host__ ~RingQueueDtoH()
  {
    if (m_ringBuffer_h)  chkError(cudaFreeHost(m_ringBuffer_h));
    if (m_tail_h)        chkError(cudaFreeHost(m_tail_h));
    if (m_head_h)        chkError(cudaFreeHost(m_head_h));
  }

  __device__ bool push(const T& value)                 // Store value at head and advance when not full
  {
    using namespace cuda::std;
    auto head = m_head_d->load(memory_order_acquire);
    auto next = (head+1) & m_capacityMask;
    auto tail = m_tail_d->load(memory_order_acquire);
    if (next == tail)  return false;                   // Full: caller retries to wait for tail to advance
    m_ringBuffer_d[head] = value;                      // Store value _before_ signaling it is available
    m_head_d->store(next, memory_order_release);       // Publish new head
    return true;
  }

  __host__ bool pop(T* const __restrict__ value)       // Fetch value at tail and advance when not empty
  {
    using namespace cuda::std;
    auto tail = m_tail_h->load(memory_order_acquire);
    auto head = m_head_h->load(memory_order_acquire);
    if (tail == head)  return false;                   // Empty: caller retries to wait for head to advance
    asm volatile("mfence" ::: "memory");               // Avoid reordering of the tail store and the head load
    *value = m_ringBuffer_h[tail];                     // Fetch value _before_ signaling it is available
    auto next = (tail+1) & m_capacityMask;
    m_tail_h->store(next, memory_order_release);       // Publish new tail
    return true;
  }

  __host__ unsigned head() const
  {
    using namespace cuda::std;
    return m_head_h->load(memory_order_acquire);
  }

  __host__ unsigned tail() const
  {
    using namespace cuda::std;
    return m_tail_h->load(memory_order_acquire);
  }

  __host__ unsigned occupancy() const
  {
    using namespace cuda::std;
    auto head = m_head_h->load(memory_order_acquire);
    auto tail = m_tail_h->load(memory_order_acquire);
    return (head - tail) & m_capacityMask;
  }

  size_t size() const
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
  cuda::std::atomic<unsigned>* m_head_d; // Must stay coherent across device and host
  cuda::std::atomic<unsigned>* m_tail_h; // Must stay coherent across device and host
  cuda::std::atomic<unsigned>* m_tail_d; // Must stay coherent across device and host
  const unsigned               m_capacityMask;
  T*                           m_ringBuffer_h;
  T*                           m_ringBuffer_d;
};

  } // Gpu
} // Drp

#endif // RINGQUEUE_HTOD_HH
