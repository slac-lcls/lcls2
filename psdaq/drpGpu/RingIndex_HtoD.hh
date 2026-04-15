#ifndef RINGINDEX_HTOD_HH
#define RINGINDEX_HTOD_HH

#include <atomic>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda/std/atomic>

namespace Drp {
  namespace Gpu {

class RingIndexHtoD
{
public:
  __host__ RingIndexHtoD(const unsigned capacity) :
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

  __host__ ~RingIndexHtoD()
  {
    if (m_tail_h)  chkError(cudaFreeHost(m_tail_h));
    if (m_head_h)  chkError(cudaFreeHost(m_head_h));
  }

  __host__ bool produce(unsigned* index)               // Move  head forward when not full
  {
    using namespace cuda::std;
    auto next = (*index+1) & m_capacityMask;
    auto tail = m_tail_h->load(memory_order_acquire);
    if (next == tail)  return false;                   // Full: caller wait for tail to advance before retrying
    asm volatile("mfence" ::: "memory");               // Avoid reordering of the head store and the tail load
    m_head_h->store(next, memory_order_release);       // Publish new head
    *index = next;
    return true;
  }

  __device__ bool consume(unsigned* index)             // Return current head when not empty
  {
    using namespace cuda::std;
    auto tail = m_tail_d->load(memory_order_acquire);
    auto head = m_head_d->load(memory_order_acquire);
    if (tail == head)  return false;                   // Wait for head to advance while empty
    *index = head;                                     // Caller now processes buffers up to [head]
    return true;
  }

  __device__ void release(const unsigned index)        // Move tail forward
  {
    using namespace cuda::std;
    auto tail = m_tail_d->load(memory_order_acquire);
    assert(index == tail);                             // Require in-order freeing
    auto next = (index+1) & m_capacityMask;
    m_tail_d->store(next, memory_order_release);       // Publish new tail
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
};

  } // Gpu
} // Drp

#endif // RINGINDEX_HTOD_H
