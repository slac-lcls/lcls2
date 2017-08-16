//---------------------------------------------------------------------------------
// Title         : Kernel Module For PGP To PCI Bridge Card Status
// Project       : PGP To PCI-E Bridge Card
//---------------------------------------------------------------------------------
// File          : PgpCardG3Status.h
// Author        : jackp
// Created       : Wed Jan  6 09:49:47 PST 2016
//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
// Copyright (c) 2016 by SLAC National Accelerator Laboratory. All rights reserved.
//---------------------------------------------------------------------------------
// Modification history:
//---------------------------------------------------------------------------------

#ifndef __PGP_CARD_G3_STATUS_H__
#define __PGP_CARD_G3_STATUS_H__
// Status Structure
typedef struct {

   // General Status
   __u32 Version;
   __u32 SerialNumber[2];
   __u32 ScratchPad;
   __u32 BuildStamp[64];
   __u32 CountReset;
   __u32 CardReset;

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
   __u32 PciBaseHdwr;
   __u32 PciBaseLen;

   // PGP Status
   __u32 PpgRate;
   __u32 PgpLoopBack[8];
   __u32 PgpTxReset[8];
   __u32 PgpRxReset[8];
   __u32 PgpTxPllRst[2];
   __u32 PgpRxPllRst[2];
   __u32 PgpTxPllRdy[2];
   __u32 PgpRxPllRdy[2];
   __u32 PgpLocLinkReady[8];
   __u32 PgpRemLinkReady[8];
   __u32 PgpRxCount[8][4];
   __u32 PgpCellErrCnt[8];
   __u32 PgpLinkDownCnt[8];
   __u32 PgpLinkErrCnt[8];
   __u32 PgpFifoErrCnt[8];

   // EVR Status & Control Registers
   __u32 EvrRunCode[8];
   __u32 EvrAcceptCode[8];
   __u32 EvrEnHdrCheck[8][4];
   __u32 EvrEnable;
   __u32 EvrReady;
   __u32 EvrReset;
   __u32 EvrRunMask;
   __u32 EvrLaneStatus;
   __u32 EvrLaneEnable;
   __u32 EvrLaneMode;
   __u32 EvrPllRst;
   __u32 EvrErrCnt;
   __u32 EvrFiducial;
   __u32 EvrRunDelay[8];
   __u32 EvrAcceptDelay[8];
   __u32 EvrRunCodeCount[8];
   __u32 EvrLutDropCount[8];
   __u32 EvrAcceptCount[8];
   __u32 EvrLaneFiducials[8];

   // RX Descriptor Status
   __u32 RxFreeFull[8];
   __u32 RxFreeValid[8];
   __u32 RxFreeFifoCount[8];
   __u32 RxReadReady;
   __u32 RxRetFifoCount;
   __u32 RxCount;
   __u32 RxWrite[8];
   __u32 RxRead[8];

   // TX Descriptor Status
   __u32 TxDmaAFull[8];
   __u32 TxReadReady;
   __u32 TxRetFifoCount;
   __u32 TxCount;
   __u32 TxRead;

} PgpCardG3Status;

#endif
