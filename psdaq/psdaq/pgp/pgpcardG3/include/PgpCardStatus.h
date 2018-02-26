//---------------------------------------------------------------------------------
// Title         : Kernel Module For PGP To PCI Bridge Card
// Project       : PGP To PCI-E Bridge Card
//---------------------------------------------------------------------------------
// File          : PgpCardMod.h
// Author        : jackp
// Created       : Wed Jan  6 09:50:42 PST 2016
//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
// Copyright (c) 2016 by SLAC National Accelerator Laboratory. All rights reserved.
//---------------------------------------------------------------------------------
// Modification history:
//---------------------------------------------------------------------------------
#ifndef PGP_CARD_STATUS_H
#define PGP_CARD_STATUS_H

typedef struct {
    __u32 PgpLoopBack;
    __u32 PgpRxReset;
    __u32 PgpTxReset;
    __u32 PgpLocLinkReady;
    __u32 PgpRemLinkReady;
    __u32 PgpRxReady;
    __u32 PgpTxReady;
    __u32 PgpRxCount;
    __u32 PgpCellErrCnt;
    __u32 PgpLinkDownCnt;
    __u32 PgpLinkErrCnt;
    __u32 PgpFifoErr;
} PgpCardLinkStatus;

// Status Structure
typedef struct {

   // General Status
   __u32 Version;

   // Scratchpad
   __u32 ScratchPad;

   // PCI Status & Control Registers
   __u32 PciCommand;
   __u32 PciStatus;
   __u32 PciDCommand;
   __u32 PciDStatus;
   __u32 PciLCommand;
   __u32 PciLStatus;
   __u32 PciLinkState;
   __u32 PciFunction;
   __u32 PciDevice;
   __u32 PciBus;

   PgpCardLinkStatus PgpLink[4];

   // TX Descriptor Status
   __u32 TxDma3AFull;
   __u32 TxDma2AFull;
   __u32 TxDma1AFull;
   __u32 TxDma0AFull;
   __u32 TxReadReady;
   __u32 TxRetFifoCount;
   __u32 TxCount;
   __u32 TxBufferCount;
   __u32 TxRead;

   // RX Descriptor Status
   __u32 RxFreeEmpty;
   __u32 RxFreeFull;
   __u32 RxFreeValid;
   __u32 RxFreeFifoCount;
   __u32 RxReadReady;
   __u32 RxRetFifoCount;
   __u32 RxCount;
   __u32 RxBufferCount;
   __u32 RxWrite[8];
   __u32 RxRead[8];

} PgpCardStatus;

#endif
