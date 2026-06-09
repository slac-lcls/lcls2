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
    m_head        (nullptr),
    m_tail        (nullptr),
    m_capacityMask(capacity-1)     // Range of the buffer index [0, capacity-1]
  {
    assert(capacity & (capacity - 1));  // Capacity must be a power of 2

    // The ring is empty when head == tail
    // Head refers to the next index to be allocated
    chkError(cudaHostAlloc(&m_head, sizeof(*m_head), cudaHostAllocDefault));
    *m_head = 0;      // Initialize to empty
    // Tail refers to the next index after the last freed one
    chkError(cudaHostAlloc(&m_tail, sizeof(*m_tail), cudaHostAllocDefault));
    *m_tail = 0;
  }

  __host__ ~RingIndexDtoH()
  {
    if (m_tail)  chkError(cudaFreeHost(m_tail));
    if (m_head)  chkError(cudaFreeHost(m_head));
  }

  __device__ bool push(unsigned index) const           /** Move head forward when not full */
  {
    using namespace cuda::std;
    auto tail = m_tail->load(memory_order_acquire);
    auto head = m_head->load(memory_order_acquire);
    //if (index != head)  printf("### Expected index %u, got %u\n", head, index);
    auto next = (head+1)  & m_capacityMask;
    if (next == tail)  return false;                   // Full: caller retries to wait for tail to advance
    m_head->store(next, memory_order_release);       // Publish new head
    return true; //next != tail;
  }

  __host__ bool pop(unsigned* index) const             /** Return current head when not empty */
  {
    using namespace cuda::std;
    auto tail = m_tail->load(memory_order_acquire);
    auto head = m_head->load(memory_order_acquire);
    if (tail == head)  return false;                   // Empty: caller retries to wait for head to advance
    *index = tail;                                     // Caller now processes buffer at [tail]
    auto next = (tail+1) & m_capacityMask;
    asm volatile("mfence" ::: "memory");               // Avoid reordering of the tail store and the head load
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
  __host__ static int _nsSleep(unsigned ns)
  {
    struct timespec ts{0, ns};
    return nanosleep(&ts, nullptr);
  }

private:
  cuda::std::atomic<unsigned>* m_head; // Must stay coherent across device and host
  cuda::std::atomic<unsigned>* m_tail; // Must stay coherent across device and host
  const unsigned               m_capacityMask;
};

  } // Gpu
} // Drp

#endif // RINGINDEX_DTOH_H
