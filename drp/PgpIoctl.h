//---------------------------------------------------------------------------------
// Title         : Kernel Module For PGP IOCTL commands
// Project       : PGP To PCI-E Bridge Card
//---------------------------------------------------------------------------------
// File          : PgpCardG3Mod.h
// Author        : jackp
// Created       : Tue Jan  5 13:36:36 PST 2016
//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
// Copyright (c) 2015 by SLAC National Accelerator Laboratory. All rights reserved.
//---------------------------------------------------------------------------------
// Modification history:
// 09/20/2013: created.
// 2015.10.26 recreated jackp@slac.stanford.edu
//---------------------------------------------------------------------------------

#ifndef __PGP_IOCTL_H__
#define __PGP_IOCTL_H__

//////////////////////
// IO Control Commands
//////////////////////

#define IOCTL_Normal_Write            1
#define IOCTL_Write_Scratch           2
#define IOCTL_Set_Debug               3
#define IOCTL_Count_Reset             4
// Set Loopback, Pass PGP Channel As Arg
#define IOCTL_Set_Loop                5
#define IOCTL_Clr_Loop                6
// Set RX Reset, Pass PGP Channel As Arg
#define IOCTL_Set_Rx_Reset            7
#define IOCTL_Clr_Rx_Reset            8
// Set TX Reset, Pass PGP Channel As Arg
#define IOCTL_Set_Tx_Reset            9
#define IOCTL_Clr_Tx_Reset           10
// Set EVR configuration
#define IOCTL_Evr_Enable             11
#define IOCTL_Evr_Disable            12
#define IOCTL_Evr_Set_Reset          13
#define IOCTL_Evr_Clr_Reset          14
#define IOCTL_Evr_Set_PLL_RST        15
#define IOCTL_Evr_Clr_PLL_RST        16
#define IOCTL_Evr_LaneModeFiducial   17
#define IOCTL_Evr_LaneModeNoFiducial 18
#define IOCTL_Evr_Fiducial           19
#define IOCTL_Evr_LaneEnable         20
#define IOCTL_Evr_LaneDisable        21
#define IOCTL_Evr_RunMask            22
#define IOCTL_Evr_RunCode            23
#define IOCTL_Evr_RunDelay           24
#define IOCTL_Evr_AcceptCode         25
#define IOCTL_Evr_AcceptDelay        26
#define IOCTL_Evr_En_Hdr_Check       27
// Read Status, Pass PgpG3CardStatus pointer as arg
#define IOCTL_Read_Status            28
#define IOCTL_Dump_Debug             29
#define IOCTL_Clear_Open_Clients     30
#define IOCTL_Clear_Polling          31
#define IOCTL_ClearFrameCounter      32
#define IOCTL_Add_More_Ports         33
#define IOCTL_Set_VC_Mask            34
#define IOCTL_Show_Version           35
#define IOCTL_Clear_Run_Count        36
#define IOCTL_End_Of_List            37
#endif
