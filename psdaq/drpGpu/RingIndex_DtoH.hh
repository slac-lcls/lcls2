#ifndef RINGINDEX_DTOH_HH
#define RINGINDEX_DTOH_HH

#include <time.h>
#include <atomic>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda/std/atomic>

#ifndef __NVCC__
#define __nanosleep(x) {}
#endif

namespace Drp {
  namespace Gpu {

class RingIndexDtoH
{
public:
  __host__ RingIndexDtoH(const unsigned capacity) :
    m_head_h      (nullptr),
    m_head_d      (nullptr),
    m_tail_h      (nullptr),
    m_tail_d      (nullptr),
    m_capacityMask(capacity-1)     // Range of the buffer index [0, capacity-1]
  {
    assert(capacity & (capacity - 1));  // Capacity must be a power of 2

    // The ring is empty when head == tail
    // Head refers to the next index to be allocated
    chkError(cudaHostAlloc(&m_head_h, sizeof(*m_head_h), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_head_d, m_head_h, 0));
    *m_head_h = 0;    // Initialize to empty
    // Tail refers to the next index after the last freed one
    chkError(cudaHostAlloc(&m_tail_h, sizeof(*m_tail_h), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_tail_d, m_tail_h, 0));
    *m_tail_h = 0;
  }

  __host__ ~RingIndexDtoH()
  {
    if (m_tail_h)  chkError(cudaFreeHost(m_tail_h));
    if (m_head_h)  chkError(cudaFreeHost(m_head_h));
  }

  __device__ bool push(unsigned index) const           /** Move head forward when not full */
  {
    using namespace cuda::std;
    auto tail = m_tail_h->load(memory_order_acquire);
    auto head = m_head_h->load(memory_order_acquire);
    auto next = (head+1)  & m_capacityMask;
    index     = (index+1) & m_capacityMask;
    while (next != index) {
      if (next == tail)  break;                        // Full: caller retries to wait for tail to advance
      next = (next+1) & m_capacityMask;
    }
    if (next != head) {
      m_head_h->store(next, memory_order_release);     // Publish new head
    }
    return next != tail;
  }

  __host__ bool pop(unsigned* index) const             /** Return current head when not empty */
  {
    using namespace cuda::std;
    auto tail = m_tail_d->load(memory_order_acquire);
    auto head = m_head_d->load(memory_order_acquire);
    if (tail == head)  return false;                   // Empty: caller retries to wait for head to advance
    *index = tail;                                     // Caller now processes buffer at [tail]
    auto next = (tail+1) & m_capacityMask;
    asm volatile("mfence" ::: "memory");               // Avoid reordering of the tail store and the head load
    m_tail_d->store(next, memory_order_release);       // Publish new tail
    return true;
  }

  __host__ unsigned occupancy() const
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
  __host__ static int _nsSleep(unsigned ns)
  {
    struct timespec ts{0, ns};
    return nanosleep(&ts, nullptr);
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

#endif // RINGINDEX_DTOH_H
