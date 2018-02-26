//---------------------------------------------------------------------------------
// Title         : Kernel Module For PGP To PCI Bridge Card G3
// Project       : 8 lane PGP To PCI-E Bridge Card
//---------------------------------------------------------------------------------
// File          : pgpcardG3.h
// Author        : jackp@slac.stanford.edu
// Created       : 2015.3.24
//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
// Copyright (c) 2015 by SLAC National Accelerator Laboratory. All rights reserved.
//---------------------------------------------------------------------------------
#include <linux/init.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/interrupt.h>
#include <linux/fs.h>
#include <linux/poll.h>
#include <linux/cdev.h>
#include <asm/uaccess.h>
#include <linux/types.h>
#include <linux/spinlock.h>
#include <asm/spinlock.h>

#define RHEL7 1

//#define MON_ONLY

// PCI IDs
#define PCI_VENDOR_ID_SLAC        0x1a4a
#define PCI_DEVICE_ID_SLAC_PRI    0x2031
#define PCI_DEVICE_ID_SLAC_SEC    0x2032

// Max number of devices to support
#define MAX_PCI_DEVICES 8
#define NUMBER_OF_CLIENTS 2
#define NUMBER_OF_LANES 4
#define NUMBER_OF_MINOR_DEVICES 3
#define SPAD_WRITE 0x55441122

enum MODELS {SmallMemoryModel=4, LargeMemoryModel=8};

// Address Map, offset from base

   /* constant VERSION_ADDR_C : slv(31 downto 0) := x"00000000"; */
   /* constant PHY_ADDR_C     : slv(31 downto 0) := x"00010000"; */
   /* constant BPI_ADDR_C     : slv(31 downto 0) := x"00030000"; */
   /* constant SPI0_ADDR_C    : slv(31 downto 0) := x"00040000"; */
   /* constant SPI1_ADDR_C    : slv(31 downto 0) := x"00050000"; */
   /* constant APP_ADDR_C     : slv(31 downto 0) := x"00800000"; */
//  MigToPcieWrapper : x"00800000"
//  HardwareSemi     : x"00C00000"

struct ClientReg {
  __u32 descAddrLo;
  __u32 descAddrHi;
  __u32 descFifoLo;
  __u32 descFifoHi;
  __u32 fifoDepth;
  __u32 readIndex;
  __u32 autoFill;
  __u32 rsvd_1c;
};

struct LaneReg {
  __u32 client;
  __u32 blockSize;
  __u32 blocksPause;
  __u32 rsvd_c;
  __u32 transferFifoDepth;
  __u32 memStatus;
  __u32 rsvd_18[2];
};

struct DaqReg {
  __u32 version;
  __u32 scratch;
  __u32 upTimeCnt;
  __u32 rsvd_C[0x3d];
  __u32 rsvd_100[0x1C0];
  __u32 buildStr[64];

  __u32     rsvd_00000900[0x00200000-0x240];
  __u32     params;
  __u32     reset;
  __u32     monSampleInterval;  // 200MHz counts
  __u32     monReadoutInterval; // monSample counts
  __u32     monEnable;
  __u32     monHistAddrLo;
  __u32     monHistAddrHi;
  __u32     monSampleCounter;
  __u32     monReadoutCounter;
  __u32     monStatus;
  __u32     rsvd_00800028[6];
  struct ClientReg clients[NUMBER_OF_CLIENTS];
  struct LaneReg   lanes  [NUMBER_OF_LANES];
  __u32     rsvd_00800080[0x00100000-0x40];
  __u32     rsvd_00c00000[0x00100000];
};

struct DaqDevice;

// Client structure
struct Client {
   struct ClientReg*  reg;

   void*       rxDescPage;
   dma_addr_t  rxDescHandle;

   void**      rxBuffer;
   dma_addr_t* rxHandle;

   __u32       readIndex;

   struct DaqDevice*  dev;
};

// Device structure
struct DaqDevice {

   // PCI address regions
   ulong             baseHdwr;
   ulong             baseLen;
   struct DaqReg*    reg;

   // Device structure
   int          major;
   struct Client client[NUMBER_OF_MINOR_DEVICES];
   struct cdev  cdev;

   // Async queue
   struct fasync_struct *async_queue;

   int irq;

   struct device* device;
   uint64_t*  monAddr;
   dma_addr_t monHandle;

   __u32 isOpen;    // lanes open mask
   __u32 openCount;
};

