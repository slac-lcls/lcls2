------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiDsPgp3.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-10
-- Last update: 2018-07-26
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: DtiApp's Top Level
-- 
--   Application interface to JungFrau.  Uses 10GbE.  Trigger is external TTL
--   (L0 only?). Control register access is external 1GbE link.
--
--   Intercept out-bound messages as register transactions for 10GbE core.
--   Use simulation embedding: ADDR(31:1) & RNW & DATA(31:0).
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 DAQ Software'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 DAQ Software', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.XpmPkg.all;
use work.DtiPkg.all;
use work.Pgp3Pkg.all;
use work.SsiPkg.all;

entity DtiDsPgp3 is
   generic (
      TPD_G               : time                := 1 ns;
      ID_G                : slv(7 downto 0)     := (others=>'0');
      ENABLE_TAG_G        : boolean             := false ;
      EN_AXIL_G           : boolean             := false ;
      DEBUG_G             : boolean             := false ;
      AXIL_BASE_ADDR_G    : slv(31 downto 0)    := (others=>'0') );
   port (
     amcClk          : in  sl;
     amcRst          : in  sl;
     amcRxP          : in  sl;
     amcRxN          : in  sl;
     amcTxP          : out sl;
     amcTxN          : out sl;
     fifoRst         : in  sl;
     -- Quad PLL Ports
     qplllock        : in  slv(1 downto 0);
     qplloutclk      : in  slv(1 downto 0);
     qplloutrefclk   : in  slv(1 downto 0);
     qpllRst         : out slv(1 downto 0);
     --
     axilClk         : in  sl;
     axilRst         : in  sl;
     axilReadMaster  : in  AxiLiteReadMasterType;
     axilReadSlave   : out AxiLiteReadSlaveType;
     axilWriteMaster : in  AxiLiteWriteMasterType;
     axilWriteSlave  : out AxiLiteWriteSlaveType;
     --
     ibRst           : in  sl;
     loopback        : in  sl;
     linkUp          : out sl;
     remLinkID       : out slv(31 downto 0);
     rxErrs          : out slv(31 downto 0);
     full            : out sl;
     monClk          : out sl;
     --
     obClk           : in  sl;
     obMaster        : in  AxiStreamMasterType;
     obSlave         : out AxiStreamSlaveType );
end DtiDsPgp3;

architecture top_level_app of DtiDsPgp3 is

  signal amcObMaster : AxiStreamMasterType;
  signal amcObSlave  : AxiStreamSlaveType;

--  signal pgpTxIn        : Pgp3TxInType := PGP3_TX_IN_INIT_C;
  signal pgpTxIn        : Pgp3TxInType := (
      disable      => '0',
      flowCntlDis  => '0',
      skpInterval  => X"0000FFF0",  -- bad default in PGP3_TX_IN_INIT_C
      opCodeEn     => '0',
      opCodeNumber => (others => '0'),
      opCodeData   => (others => '0'));
  signal pgpTxOut       : Pgp3TxOutType;
  signal pgpRxIn        : Pgp3RxInType := PGP3_RX_IN_INIT_C;
  signal pgpRxOut       : Pgp3RxOutType;
  signal pgpTxMasters   : AxiStreamMasterArray(NUM_DTI_VC_C-1 downto 0) := (others=>AXI_STREAM_MASTER_INIT_C);
  signal pgpTxSlaves    : AxiStreamSlaveArray (NUM_DTI_VC_C-1 downto 0);
  signal pgpRxMasters   : AxiStreamMasterArray(NUM_DTI_VC_C-1 downto 0);
  signal pgpRxCtrls     : AxiStreamCtrlArray  (NUM_DTI_VC_C-1 downto 0) := (others=>AXI_STREAM_CTRL_UNUSED_C);

  signal pgpClk         : sl;
  signal pgpRst         : sl;

begin

--  locTxIn.locData          <= ID_G;
  pgpRxIn.loopback         <= '0' & loopback & '0';
  linkUp                   <= pgpRxOut.linkReady;
--  remLinkID                <= pgpRxOut.remLinkData;
  monClk                   <= pgpClk;
  
  U_Fifo : entity work.AxiStreamFifo
    generic map (
      SLAVE_AXI_CONFIG_G  => US_OB_CONFIG_C,
      MASTER_AXI_CONFIG_G => PGP3_AXIS_CONFIG_C,
      FIFO_ADDR_WIDTH_G   => 9,
      PIPE_STAGES_G       => 2 )
    port map ( 
      -- Slave Port
      sAxisClk    => obClk,
      sAxisRst    => fifoRst,
      sAxisMaster => obMaster,
      sAxisSlave  => obSlave,
      -- Master Port
      mAxisClk    => pgpClk,
      mAxisRst    => fifoRst,
      mAxisMaster => amcObMaster,
      mAxisSlave  => amcObSlave );

  --U_RXERR : entity work.SynchronizerOneShotCnt
  --  generic map ( CNT_WIDTH_G => 32 )
  --  port map ( wrClk   => pgpClk,
  --             rdClk   => axilClk,
  --             cntRst  => fifoRst,
  --             rollOverEn => '1',
  --             dataIn  => pgpRxOut.linkError,
  --             dataOut => open,
  --             cntOut  => rxErrs );
  -- 64b/66b encoding doesn't trap many errors
  rxErrs                   <= (others=>'0');
  
  pgpTxMasters(0)          <= amcObMaster;
  amcObSlave               <= pgpTxSlaves(0);

  U_PgpFb : entity work.DtiPgp3Fb
    port map ( pgpClk       => pgpClk,
               pgpRst       => pgpRst,
               pgpRxOut     => pgpRxOut,
               rxAlmostFull => full,
               rxLinkId     => remLinkID );

  U_Pgp3 : entity work.Pgp3GthUs
    generic map ( NUM_VC_G     => NUM_DTI_VC_C,
--                  DEBUG_G      => DEBUG_G,
                  EN_DRP_G     => EN_AXIL_G,
                  EN_PGP_MON_G => EN_AXIL_G,
                  AXIL_CLK_FREQ_G  => 156.25e+6,
                  AXIL_BASE_ADDR_G => AXIL_BASE_ADDR_G )
    port map ( -- Stable Clock and Reset
               stableClk    => axilClk,
               stableRst    => axilRst,
               -- QPLL Interface
               qpllLock     => qplllock,
               qpllclk      => qplloutclk,
               qpllrefclk   => qplloutrefclk,
               qpllRst      => qpllRst,
               -- Gt Serial IO
               pgpGtTxP     => amcTxP,
               pgpGtTxN     => amcTxN,
               pgpGtRxP     => amcRxP,
               pgpGtRxN     => amcRxN,
               -- Clocking
               pgpClk       => pgpClk,
               pgpClkRst    => pgpRst,
               -- Non VC Tx Signals
               pgpTxIn      => pgpTxIn,
               pgpTxOut     => pgpTxOut,
               -- Non VC Rx Signals
               pgpRxIn      => pgpRxIn,
               pgpRxOut     => pgpRxOut,
               -- Frame TX Interface
               pgpTxMasters => pgpTxMasters,
               pgpTxSlaves  => pgpTxSlaves,
               -- Frame RX Interface
               pgpRxMasters => pgpRxMasters,
               pgpRxCtrl    => pgpRxCtrls,
               -- AXI-Lite Register Interface
               axilClk         => axilClk,
               axilRst         => axilRst,
               axilReadMaster  => axilReadMaster,
               axilReadSlave   => axilReadSlave,
               axilWriteMaster => axilWriteMaster,
               axilWriteSlave  => axilWriteSlave );

  end top_level_app;
