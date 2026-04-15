#ifndef RINGINDEX_DTOD_HH
#define RINGINDEX_DTOD_HH

#include <cassert>

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cuda/std/atomic>

namespace Drp {
  namespace Gpu {

class RingIndexDtoD
{
public:
  __host__ RingIndexDtoD(const unsigned capacity) :
    m_head_h      (nullptr),
    m_head_d      (nullptr),
    m_tail_h      (nullptr),
    m_tail_d      (nullptr),
    m_capacityMask(capacity-1)     // Range of the buffer index [0, capacity-1]
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
  }

  __host__ ~RingIndexDtoD()
  {
    if (m_tail_h)  chkError(cudaFreeHost(m_tail_h));
    if (m_head_h)  chkError(cudaFreeHost(m_head_h));
  }

  __device__ bool allocate(unsigned* index)            // Return next index in monotonic fashion
  {
    using namespace cuda;
    auto head = m_head_d->load(memory_order_acquire);
    auto next = (head+1) & m_capacityMask;
    auto tail = m_tail_d->load(memory_order_acquire);
    if (next == tail)  return false;                   // Full: caller waits for tail to advance before retrying
    *index = head;                                     // Caller now fills buffer[head]
    return true;
  }

  __device__ bool post(unsigned index)                 // Move head forward when not full
  {
    using namespace cuda;
    auto head = m_head_d->load(memory_order_acquire);
    if (index != head)  return false;                  // Full: caller waits for tail to advance before retrying
    auto next = (index+1) & m_capacityMask;
    m_head_d->store(next, memory_order_release);       // Publish new head
    return true;
  }

  __device__ bool pend(unsigned* index) const          // Return current head
  {
    using namespace cuda;
    auto tail = m_tail_d->load(memory_order_acquire);
    auto head = m_head_d->load(memory_order_acquire);
    if (tail == head)  return false;                   // Empty: caller waits for head to advance before retrying
    *index = head;                                     // Caller now processes buffers up to [head]
    return true;
  }

  __host__ void release(unsigned index)                // Move tail forward
  {
    using namespace cuda;
    auto tail = m_tail_h->load(memory_order_acquire);
    assert(index == tail);                             // Require in-order freeing
    auto next = (tail+1) & m_capacityMask;
    m_tail_h->store(next, memory_order_release);       // Publish new tail
  }

  __host__ unsigned occupancy() const
  {
    using namespace cuda;
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
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_head_h; // Must stay coherent across streams
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_head_d; // Must stay coherent across streams
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_tail_h; // Must stay coherent across streams
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_tail_d; // Must stay coherent across streams
  const unsigned                                     m_capacityMask;
};

  }
}

#endif // RINGINDEX_DTOD_H
