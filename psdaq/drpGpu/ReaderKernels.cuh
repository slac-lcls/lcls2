// The Reader's per-event kernel, as a template on a detector-supplied policy.  Not a
// virtual or a device function pointer: device code can't be linked across the
// dlopen'd detector .so and drp_gpu, so each .so instantiates the kernel and the
// policy inlines.
//
// A policy supplies
//   __device__ void process(const EventPayload&, unsigned tid, unsigned stride) const
// and owns the payload's interpretation, including where each sub-frame goes: tdests
// need not be contiguous (Jungfrau: 2, 4, 5, 6, ...) and batches may nest.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "MemPool.hh"                   // For DmaDsc
#include "EventBatcher.hh"
#include "psdaq/service/EbDgram.hh"     // For Pds::TimingHeader

namespace Drp {
  namespace Gpu {

// Per-event status in the last word of the event's host-visible buffer, which
// TrgInpGen::_receiver() reports, since a kernel can't log.
enum EventStatus : uint32_t {
  EventStatusOk = 0,
  EventStatusDmaSizeTooSmall,           // DMA shorter than a TimingHeader
  EventStatusBatchUnintelligible,       // Sub-frame scan failed; the reason is in
                                        // Reader::batcherStatus()
};

// Where that code sits within one event's host-visible buffer
__host__ __device__
inline size_t eventStatusIndex(size_t hdrBufsCnt) { return hdrBufsCnt - 1; }

// What the _event kernel needs that no detector has to know about.  Assembled by
// Reader::_recordGraph and handed to Gpu::Detector::recordEvent().
struct EventKernelArgs
{
  unsigned                     reader;
  unsigned*                    state;
  unsigned*                    dmaBufferIdx;
  unsigned*                    pebbleIdx;
  uint8_t const* const*        dmaBuffers;    // [dmaCount][maxDmaSize]
  size_t                       frameSize;     // Bytes of the line bearing the DmaDsc
  uint32_t*                    hdrBuffers;    // [nBuffers * hdrBufsCnt]
  size_t                       hdrBufsCnt;
  float*                       calibBuffers;  // [nBuffers * calibBufsCnt]
  size_t                       calibBufsCnt;
  // These describe wherever this mode's raw block goes, so they stay mode-neutral:
  // ahead of each reduce buffer in pass-through, or in the prescale pool in BEAM.
  uint8_t*                     rawBuffers;
  size_t                       rawStride;
  size_t                       rawBufsCnt;
  unsigned                     rawBufCnt;     // Pool slots; 0 in pass-through, where
                                              // the slot is the pebble index
  size_t                       rawBufsRsvd;   // Bytes before the raw block in a buffer
  unsigned*                    rawTicket;     // Claim counter, device memory
  cuda::std::atomic<unsigned>*       rawOverflow; // Raised here, acted on by the host
  cuda::std::atomic<unsigned> const* rawReleased; // Pinned; advanced by the recorder
  EvtBatcherSubFrames const*   subFrames;     // nullptr when the data isn't batched
  unsigned const*              evtStatus;     // An EventStatus, set by _waitForDMA
  uint64_t*                    stateMon;
  unsigned*                    smMask;        // SMs _event ran on; nullptr unless readerHist
};

static __device__ __forceinline__ unsigned _smid()
{
  unsigned id;
  asm volatile("mov.u32 %0, %smid;" : "=r"(id));
  return id;
}

static __device__ __forceinline__ void _markSm(unsigned* const mask)
{
  auto const id{_smid()};
  atomicOr(&mask[(id >> 5) & 7], 1u << (id & 31));
}

// This event's bucket of bytes, and where its results go.  Handed to the policy.
struct EventPayload
{
  uint8_t const*             data;       // The DMA payload, past the DmaDsc line
  size_t                     size;       // DmaDsc::size, i.e. bytes of 'data'
  float*                     out;        // This event's calibrated buffer
  size_t                     outCnt;     // Elements available in 'out'
  uint8_t*                   raw;        // Raw block to fill, or null: the only test
  size_t                     rawCnt;     // a policy needs.  Pass-through writes it
                                         // instead of 'out'.
  EvtBatcherSubFrames const* subFrames;  // The cached scan; null when not batched
  bool                       batched;    // The Detector presents sub-frames
  bool                       hasData;    // False for a transition or a payload
                                         // whose size or layout wasn't understood
  unsigned                   pebbleIdx;  // For indexing per-event side buffers
};

// Claims a prescaled event's prescale slot, <<<1,1>>> ahead of _event so all its
// blocks agree.  The self-relaunching graph has no per-event host step to do it in.
// Outrunning the pool only raises rawOverflow, for the recorder to abort on: a
// post-mortem, tolerable because the pebble pool bounds it (NPrescaleBuffers, Reducer.cu).
static __global__
void _claimRawSlot(EventKernelArgs const args)
{
  if (*args.state != 2)  return;

  auto const __restrict__ in  = (uint32_t const*)args.dmaBuffers[*args.dmaBufferIdx];
  auto const __restrict__ hdr = args.hdrBuffers + *args.pebbleIdx * args.hdrBufsCnt;
  auto const     nFrmWds = args.frameSize/sizeof(*in);
  auto const __restrict__ payload = (uint8_t const*)&in[nFrmWds];
  auto const __restrict__ th = (uint32_t const*)
    (args.subFrames ? payload + ((EvtBatcherHeader const*)payload)->lineWidth()
                    : payload);
  auto const dmaSize = in[1];
  auto const hasData = args.subFrames ? args.subFrames->matches(dmaSize)
                                      : dmaSize > sizeof(Pds::TimingHeader);
  auto const keepRaw = ((th[4] >> 22) & 1) && hasData;

  if (!keepRaw) {
    hdr[rawSlotIndex(args.hdrBufsCnt)] = NoRawSlot;
    return;
  }

  // Acquire-load released: if its advance were never seen, the monotonic ticket would
  // trip the test after rawBufCnt claims whatever the true occupancy.
  auto const ticket   = atomicAdd(args.rawTicket, 1u);
  auto const released = args.rawReleased->load(cuda::std::memory_order_acquire);
  if (ticket - released >= args.rawBufCnt) {
    args.rawOverflow->store(1, cuda::std::memory_order_release);
  }
  hdr[rawSlotIndex(args.hdrBufsCnt)] = ticket % args.rawBufCnt;
}

// Publish the DmaDsc and TimingHeader to the host-visible buffer, then hand the rest
// to the detector's policy, passed by value so it can carry its own constants.
template<class Calib>
__global__
void _event(EventKernelArgs const args, Calib const calib)
{
  if (*args.state != 2)  return;

  auto const tid    = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;

  auto const dmaBufIdx{*args.dmaBufferIdx}; // All threads load these into a register
  auto const pblBufIdx{*args.pebbleIdx};    // from global memory

  auto const __restrict__ in  = (uint32_t const*)args.dmaBuffers[dmaBufIdx];
  auto const __restrict__ hdr = args.hdrBuffers + pblBufIdx * args.hdrBufsCnt;

  constexpr auto nDscWds = sizeof(DmaDsc)/sizeof(*in);
  constexpr auto nHdrWds = sizeof(Pds::TimingHeader)/sizeof(*in);
  constexpr auto nLdrWds = nDscWds + nHdrWds;
  auto const     nFrmWds = args.frameSize/sizeof(*in);

  // When batched, the TimingHeader is sub-frame 0, the line after the batcher header
  // (as BEBDetector::getTimingHeader()); transitions are batched too, so no scan needed.
  auto const              dmaSize = in[1];
  auto const              batched = args.subFrames != nullptr;
  auto const __restrict__ payload = (uint8_t const*)&in[nFrmWds];
  auto const __restrict__ th      = (uint32_t const*)
    (batched ? payload + ((EvtBatcherHeader const*)payload)->lineWidth()
             : payload);
  // Whether there is anything for the policy to interpret
  auto const hasData = batched ? args.subFrames->matches(dmaSize)
                               : dmaSize > sizeof(Pds::TimingHeader);
  // keepRaw from the env word (pulseId 2 words, time 2, env): TransitionBase::keepRaw()
  // is host-only, and CUDA attributes in xtcdata would force nvcc on psana.
  static_assert(sizeof(Pds::TimingHeader) == 8*sizeof(uint32_t),
                "TimingHeader layout changed: re-check the env word index below");
  auto const keepRaw = ((th[4] >> 22) & 1) && hasData;
  if      (tid < nDscWds)   { hdr[tid] = in[tid]; }
  else if (tid < nLdrWds)   { hdr[tid] = th[tid - nDscWds]; }
  // Pass _waitForDMA's verdict on this event to the host to report
  else if (tid == nLdrWds)  { hdr[eventStatusIndex(args.hdrBufsCnt)] = *args.evtStatus; }

  // Pass-through: every event's raw block, ahead of its reduce buffer, keepRaw
  // irrelevant.  Prescaling: marked events only, in the slot _claimRawSlot claimed.
  uint8_t* rawPtr = nullptr;
  if (args.rawBuffers && hasData) {
    if (args.rawBufCnt == 0) {          // Pass-through: one per reduce buffer
      rawPtr = &args.rawBuffers[pblBufIdx * args.rawStride];
    } else if (keepRaw) {               // Prescaling: the claimed slot's raw region
      auto const slot = hdr[rawSlotIndex(args.hdrBufsCnt)];
      rawPtr = &args.rawBuffers[slot * args.rawStride + args.rawBufsRsvd];
    }
  }

  EventPayload const pyld{payload,
                          dmaSize,
                          &args.calibBuffers[pblBufIdx * args.calibBufsCnt],
                          args.calibBufsCnt,
                          rawPtr,
                          args.rawBufsCnt,
                          args.subFrames,
                          batched,
                          hasData,
                          pblBufIdx};
  calib.process(pyld, tid, stride);

  // Sampled 1 event in 64, since this is an atomic per block
  if (args.smMask && threadIdx.x == 0 && (pblBufIdx & 0x3f) == 0)  _markSm(args.smMask);

  // The state variable is likely set before the last thread is done, but the
  // next kernel won't check it before all threads of this kernel complete
  if (tid == 0)  *args.state = 3;
}

// Grid-stride pedestal/gain calibration of one contiguous run of raw elements.
// Pedestals and gains are [range][pgStride] over the whole frame: pgStride is the
// frame's element count and pgOffset the run's position in it.  Range and data fields
// are both given, since detectors order them differently.
__device__
inline void pedGainCalibrate(float*   const        __restrict__ calib,
                             uint16_t const* const __restrict__ raw,
                             unsigned const                     nElements,
                             unsigned const                     rangeOffset,
                             unsigned const                     rangeBits,
                             unsigned const                     dataOffset,
                             unsigned const                     dataBits,
                             float    const* const __restrict__ pedArray,
                             float    const* const __restrict__ gainArray,
                             unsigned const                     pgStride,
                             unsigned const                     pgOffset,
                             float    const* const __restrict__ ref,
                             unsigned const                     tid,
                             unsigned const                     stride)
{
  auto const rangeMask{(1u << rangeBits) - 1u};
  auto const dataMask {(1u << dataBits)  - 1u};
  for (auto i = tid; i < nElements; i += stride) {
    auto const              range = (raw[i] >> rangeOffset) & rangeMask;
    auto const __restrict__ peds  = &pedArray [range * pgStride + pgOffset];
    auto const __restrict__ gains = &gainArray[range * pgStride + pgOffset];
    auto const              data  = (raw[i] >> dataOffset) & dataMask;
    calib[i] = (float(data) - peds[i]) * gains[i];

    //if (ref && (calib[i] != ref[i])) {
    //  printf("### Reader: blk %d, thr %d, Mismatch @ %u: calib %f != ref %f\n",
    //         blockIdx.x, threadIdx.x, i, calib[i], ref[i]);
    //}
  }
}

// The policy for a payload of contiguous uint16_t after the TimingHeader, calibrated
// from pedestals and gains.  Batched detectors need their own.
struct PedGainCalib
{
  float const* peds;
  float const* gains;
  float const* ref;                     // Simulator mode only; null otherwise
  unsigned     refBufCnt;
  unsigned     rangeOffset;
  unsigned     rangeBits;
  unsigned     dataOffset;
  unsigned     dataBits;

  __device__
  void process(const EventPayload& pyld, unsigned tid, unsigned stride) const
  {
    if (!pyld.hasData)  return;         // A transition, or nothing intelligible

    auto const __restrict__ raw = (uint16_t const*)(pyld.data + sizeof(Pds::TimingHeader));
    auto const payloadCnt = (pyld.size - sizeof(Pds::TimingHeader))/sizeof(*raw);
    auto const elementCnt = payloadCnt > pyld.outCnt ? pyld.outCnt : payloadCnt;
    auto const __restrict__ refBuf = ref && refBufCnt
                                   ? &ref[(pyld.pebbleIdx % refBufCnt) * pyld.outCnt]
                                   : (float const*)nullptr;
    // pgStride is the frame size, not this event's element count, which may be short
    pedGainCalibrate(pyld.out, raw, elementCnt, rangeOffset, rangeBits,
                     dataOffset, dataBits,
                     peds, gains, pyld.outCnt, 0, refBuf, tid, stride);
  }
};

  } // Gpu
} // Drp
