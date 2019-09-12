-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiDsPgp5Gb.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-10
-- Last update: 2017-12-11
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: DtiApp's Top Level
-- 
-- Note: Common-to-DtiApp interface defined here (see URL below)
--       https://confluence.slac.stanford.edu/x/rLyMCw
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

library unisim;
use unisim.vcomponents.all;

use work.StdRtlPkg.all;
use work.AxiStreamPkg.all;
use work.AxiLitePkg.all;
use work.DtiPkg.all;
use work.Pgp2bPkg.all;

entity DtiDsPgp5Gb is
   generic (
      TPD_G               : time                := 1 ns;
      ID_G                : slv(7 downto 0)     := x"00";
      INCLUDE_AXIL_G      : boolean             := false;
      DEBUG_G             : boolean             := false );
   port (
     coreClk         : in  sl;
     coreRst         : in  sl;
     gtRefClk        : in  sl;
     amcRxP          : in  sl;
     amcRxN          : in  sl;
     amcTxP          : out sl;
     amcTxN          : out sl;
     fifoRst         : in  sl;
     --
     axilClk         : in  sl := '0';
     axilRst         : in  sl := '0';
     axilReadMaster  : in  AxiLiteReadMasterType  := AXI_LITE_READ_MASTER_INIT_C;
     axilReadSlave   : out AxiLiteReadSlaveType;
     axilWriteMaster : in  AxiLiteWriteMasterType := AXI_LITE_WRITE_MASTER_INIT_C;
     axilWriteSlave  : out AxiLiteWriteSlaveType;
     --  App Interface
     ibRst           : in  sl;
     linkUp          : out sl;
     remLinkID       : out slv(7 downto 0);
     rxErrs          : out slv(31 downto 0);
     full            : out sl;
     --
     obClk           : in  sl;
     obMaster        : in  AxiStreamMasterType;
     obSlave         : out AxiStreamSlaveType );
end DtiDsPgp5Gb;

architecture rtl of DtiDsPgp5Gb is

  signal amcObMaster : AxiStreamMasterType;
  signal amcObSlave  : AxiStreamSlaveType;

  signal locTxIn        : Pgp2bTxInType := PGP2B_TX_IN_INIT_C;
  signal pgpTxIn        : Pgp2bTxInType;
  signal pgpTxOut       : Pgp2bTxOutType;
  signal pgpRxIn        : Pgp2bRxInType;
  signal pgpRxOut       : Pgp2bRxOutType;
  signal pgpTxMasters   : AxiStreamMasterArray(3 downto 0) := (others=>AXI_STREAM_MASTER_INIT_C);
  signal pgpTxSlaves    : AxiStreamSlaveArray (3 downto 0);
  signal pgpRxMasters   : AxiStreamMasterArray(3 downto 0);
  signal pgpRxCtrls     : AxiStreamCtrlArray  (3 downto 0) := (others=>AXI_STREAM_CTRL_UNUSED_C);

  signal pgpClk         : sl;
  signal pgpRst         : sl;

begin

  pgpRst <= ibRst;

  locTxIn.locData          <= ID_G;
  linkUp                   <= pgpRxOut.linkReady;
  remLinkID                <= pgpRxOut.remLinkData;
  
--  U_Fifo : entity work.AxiStreamFifoV2
  U_Fifo : entity work.AxiStreamFifo
    generic map (
      SLAVE_AXI_CONFIG_G  => US_OB_CONFIG_C,
      MASTER_AXI_CONFIG_G => SSI_PGP2B_CONFIG_C,
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

  U_RXERR : entity work.SynchronizerOneShotCnt
    generic map ( CNT_WIDTH_G => 32 )
    port map ( wrClk   => pgpClk,
               rdClk   => axilClk,
               cntRst  => fifoRst,
               rollOverEn => '1',
               dataIn  => pgpRxOut.linkError,
               dataOut => open,
               cntOut  => rxErrs );
  
  pgpTxMasters(0)          <= amcObMaster;
  amcObSlave               <= pgpTxSlaves(0);

  U_PgpFb : entity work.DtiPgpFb
    port map ( pgpClk       => pgpClk,
               pgpRst       => pgpRst,
               pgpRxOut     => pgpRxOut,
               rxAlmostFull => full );
  
  U_Pgp2b : entity work.MpsPgpFrontEnd
    generic map ( NUM_VC_EN_G => 1,
                  DEBUG_G     => DEBUG_G )
    port map ( pgpClk       => pgpClk,
               pgpRst       => pgpRst,
               stableClk    => axilClk,
               gtRefClk     => gtRefClk,
               txOutClk     => pgpClk,
               --
               pgpTxIn      => pgpTxIn,
               pgpTxOut     => pgpTxOut,
               pgpRxIn      => pgpRxIn,
               pgpRxOut     => pgpRxOut,
               -- Frame TX Interface
               pgpTxMasters => pgpTxMasters,
               pgpTxSlaves  => pgpTxSlaves,
               -- Frame RX Interface
               pgpRxMasters => pgpRxMasters,
               pgpRxCtrl    => pgpRxCtrls,
               -- GT Pins
               gtTxP        => amcTxP,
               gtTxN        => amcTxN,
               gtRxP        => amcRxP,
               gtRxN        => amcRxN );

  GEN_AXIL : if INCLUDE_AXIL_G generate
    U_Axi : entity work.Pgp2bAxi
      generic map ( AXI_CLK_FREQ_G => 156.25E+6 )
      port map ( -- TX PGP Interface (pgpTxClk)
        pgpTxClk         => pgpClk,
        pgpTxClkRst      => pgpRst,
        pgpTxIn          => pgpTxIn,
        pgpTxOut         => pgpTxOut,
        locTxIn          => locTxIn,
        -- RX PGP Interface (pgpRxClk)
        pgpRxClk         => pgpClk,
        pgpRxClkRst      => pgpRst,
        pgpRxIn          => pgpRxIn,
        pgpRxOut         => pgpRxOut,
        -- AXI-Lite Register Interface (axilClk domain)
        axilClk          => axilClk,
        axilRst          => axilRst,
        axilReadMaster   => axilReadMaster,
        axilReadSlave    => axilReadSlave,
        axilWriteMaster  => axilWriteMaster,
        axilWriteSlave   => axilWriteSlave );
  end generate;

  NOGEN_AXIL : if not INCLUDE_AXIL_G generate
    pgpTxIn        <= locTxIn;
    pgpRxIn        <= PGP2B_RX_IN_INIT_C;
    axilReadSlave  <= AXI_LITE_READ_SLAVE_INIT_C;
    axilWriteSlave <= AXI_LITE_WRITE_SLAVE_INIT_C;
  end generate;
  
end rtl;
