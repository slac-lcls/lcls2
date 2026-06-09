#ifndef RINGQUEUE_HTOD_HH
#define RINGQUEUE_HTOD_HH

#include <atomic>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda/std/atomic>

namespace Drp {
  namespace Gpu {

template <typename T>
class RingQueueHtoD
{
public:
  __host__ RingQueueHtoD(const unsigned capacity) :
    m_head        (nullptr),
    m_tail        (nullptr),
    m_capacityMask(capacity-1),    // Range of the buffer queue [0, capacity-1]
    m_ringBuffer  (nullptr)
  {
    assert(capacity & (capacity - 1));  // Capacity must be a power of 2

    // The ring is empty when head == tail
    // Head points to the next index to be allocated
    chkError(cudaHostAlloc(&m_head, sizeof(*m_head), cudaHostAllocDefault));
    *m_head = 0;
    // Tail points to the next index after the last freed one
    chkError(cudaHostAlloc(&m_tail, sizeof(*m_tail), cudaHostAllocDefault));
    *m_tail = 0;

    // Use pinned memory for the ring buffer for low latency access from both Host and Device
    chkError(cudaHostAlloc(&m_ringBuffer, capacity * sizeof(*m_ringBuffer), cudaHostAllocDefault));
  }

  __host__ ~RingQueueHtoD()
  {
    if (m_ringBuffer)  chkError(cudaFreeHost(m_ringBuffer));
    if (m_tail)        chkError(cudaFreeHost(m_tail));
    if (m_head)        chkError(cudaFreeHost(m_head));
  }

  __host__ bool push(const T& value) const             /** Store value at head and advance when not full */
  {
    using namespace cuda::std;
    auto tail = m_tail->load(memory_order_acquire);
    auto head = m_head->load(memory_order_acquire);
    auto next = (head+1) & m_capacityMask;
    if (next == tail)  return false;                   // Full: caller retries to wait for tail to advance
    asm volatile("mfence" ::: "memory");               // Avoid reordering of the head store and the tail load
    m_ringBuffer[head] = value;                        // Store value _before_ signaling it is available
    m_head->store(next, memory_order_release);         // Publish new head
    return true;
  }

  __device__ bool pop(T* const __restrict__ value) const /** Fetch value at tail and advance when not empty */
  {
    using namespace cuda::std;
    auto tail = m_tail->load(memory_order_acquire);
    auto head = m_head->load(memory_order_acquire);
    if (tail == head)  return false;                   // Empty: caller retries to wait for head to advance
    *value = m_ringBuffer[tail];                       // Fetch value _before_ signaling it is available
    auto next = (tail+1) & m_capacityMask;
    m_tail->store(next, memory_order_release);         // Publish new tail
    return true;
  }

  __host__ __device__ unsigned head() const
  {
    using namespace cuda::std;
    return m_head->load(memory_order_acquire);
  }

  __host__ __device__ unsigned tail() const
  {
    using namespace cuda::std;
    return m_tail->load(memory_order_acquire);
  }

  __host__ __device__ unsigned occupancy() const
  {
    using namespace cuda::std;
    auto head = m_head->load(memory_order_acquire);
    auto tail = m_tail->load(memory_order_acquire);
    return (head - tail) & m_capacityMask;
  }

  __host__ __device__ size_t size() const
  {
    return m_capacityMask + 1;
  }

  __host__ void reset()
  {
    *m_head = 0;
    *m_tail = 0;
  }

private:
  cuda::std::atomic<unsigned>* m_head; // Must stay coherent across device and host
  cuda::std::atomic<unsigned>* m_tail; // Must stay coherent across device and host
  const unsigned               m_capacityMask;
  T*                           m_ringBuffer;
};

  } // Gpu
} // Drp

#endif // RINGQUEUE_HTOD_HH
