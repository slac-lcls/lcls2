#ifndef Pds_Tpr_Queues_hh
#define Pds_Tpr_Queues_hh

#define MOD_SHARED 14
#define MAX_TPR_ALLQ (32*1024)
#define MAX_TPR_BSAQ  1024
#define MSG_SIZE      32

namespace Pds {
  namespace Tpr {

// DMA Buffer Size, Bytes (could be as small as 512B)
#define BUF_SIZE 4096
#define NUMBER_OF_RX_BUFFERS 256

    class TprEntry {
    public:
      uint32_t word[MSG_SIZE];
    };

    class TprQIndex {
    public:
      long long idx[MAX_TPR_ALLQ];
    };

    class Queues {
    public:
      TprEntry  allq  [MAX_TPR_ALLQ]; // master queue of shared messages
      TprEntry  bsaq  [MAX_TPR_BSAQ]; // queue of BSA messages
      TprQIndex allrp [MOD_SHARED];   // indices into allq
      long long allwp [MOD_SHARED];   // write pointer into allrp
      long long bsawp;                // write pointer into bsaq
      long long gwp;
      int       fifofull;
    };
  };
};

#endif
