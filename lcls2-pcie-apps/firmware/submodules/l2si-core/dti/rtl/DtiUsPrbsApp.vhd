------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiUsPrbsApp.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-10
-- Last update: 2017-05-31
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: DtiApp's Top Level
-- 
--   Application interface to Prbs.  Uses 10GbE.  Trigger is external TTL
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
use work.DtiSimPkg.all;
use work.SsiPkg.all;

entity DtiUsPrbsApp is
   generic (
      TPD_G               : time                := 1 ns;
      ID_G                : slv(7 downto 0)     := (others=>'0');
      ENABLE_TAG_G        : boolean             := false ;
      AXIL_BASEADDR_G     : slv(31 downto 0)    := (others=>'0');
      DEBUG_G             : boolean             := false );
   port (
     amcClk          : in  sl;
     amcRst          : in  sl;
     status          : out DtiUsAppStatusType;
     amcRxP          : in  sl;
     amcRxN          : in  sl;
     amcTxP          : out sl;
     amcTxN          : out sl;
     fifoRst         : in  sl;
     -- Quad PLL Ports
     qplllock        : in  sl;
     qplloutclk      : in  sl;
     qplloutrefclk   : in  sl;
     qpllRst         : out sl;
     --
     ibClk           : in  sl;
     ibRst           : in  sl;
     ibMaster        : out AxiStreamMasterType;
     ibSlave         : in  AxiStreamSlaveType;
     linkUp          : out sl;
     rxErr           : out sl;
     --
     obClk           : in  sl;
     obRst           : in  sl;
     obTrig          : in  XpmPartitionDataType;
     obMaster        : in  AxiStreamMasterType;
     obSlave         : out AxiStreamSlaveType;
     --
     axilClk         : in  sl;
     axilRst         : in  sl;
     axilReadMaster  : in  AxiLiteReadMasterType;
     axilReadSlave   : out AxiLiteReadSlaveType;
     axilWriteMaster : in  AxiLiteWriteMasterType;
     axilWriteSlave  : out AxiLiteWriteSlaveType );
end DtiUsPrbsApp;

architecture top_level_app of DtiUsPrbsApp is

  type StateType is (S_IDLE, S_READ_HDR, S_READ_BID, S_PAYLOAD, S_DUMP, S_SKIP);
  type AxilStateType is (S_AXIL_IDLE, S_WR_ACK_WAIT, S_RD_ACK_WAIT, S_IB_ACK_WAIT);

  type ARegType is record
    --  Register state
    astate     : AxilStateType;
    rMaster    : AxiLiteReadMasterType;
    wMaster    : AxiLiteWriteMasterType;
    regmaster  : AxiStreamMasterType;
    regslave   : AxiStreamSlaveType;
    status     : DtiUsAppStatusType;
  end record;
  
  constant AREG_INIT_C : ARegType := (
    astate     => S_AXIL_IDLE,
    rMaster    => AXI_LITE_READ_MASTER_INIT_C,
    wMaster    => AXI_LITE_WRITE_MASTER_INIT_C,
    regmaster  => AXI_STREAM_MASTER_INIT_C,
    regslave   => AXI_STREAM_SLAVE_INIT_C,
    status     => DTI_US_APP_STATUS_INIT_C );

  signal a   : ARegType := AREG_INIT_C;
  signal ain : ARegType;

  constant AXIS_CONFIG_C : AxiStreamConfigType := (
    TSTRB_EN_C    => false,
    TDATA_BYTES_C => 8,
    TID_BITS_C    => 0,
    TDEST_BITS_C  => 0,
    TKEEP_MODE_C  => TKEEP_COMP_C,
    TUSER_BITS_C  => 0,
    TUSER_MODE_C  => TUSER_FIRST_LAST_C );
  
  signal regReadSlave   : AxiLiteReadSlaveType;
  signal regWriteSlave  : AxiLiteWriteSlaveType;

  signal dmaIbMaster    : AxiStreamMasterType;
  signal dmaIbSlave     : AxiStreamSlaveType;
  signal dmaObMaster    : AxiStreamMasterType;
  signal dmaObSlave     : AxiStreamSlaveType;

  signal regMaster      : AxiStreamMasterType;
  signal regSlave       : AxiStreamSlaveType;
  signal iqpllRst       : sl;

  constant MACADDR : slv(47 downto 0) := (x"0203005644" & ID_G);

  signal mAxilReadMasters  : AxiLiteReadMasterArray (1 downto 0);
  signal mAxilReadSlaves   : AxiLiteReadSlaveArray  (1 downto 0);
  signal mAxilWriteMasters : AxiLiteWriteMasterArray(1 downto 0);
  signal mAxilWriteSlaves  : AxiLiteWriteSlaveArray (1 downto 0);
  constant AXI_CROSSBAR_MASTERS_CONFIG_C : AxiLiteCrossbarMasterConfigArray(1 downto 0) := (
    0    => (
      baseAddr        => AXIL_BASEADDR_G + x"00000000",
      addrBits        => 12,
      connectivity    => x"FFFF"),
    1    => (
      baseAddr        => AXIL_BASEADDR_G + x"00001000",
      addrBits        => 12,
      connectivity    => x"FFFF") );

begin

  qpllRst <= iqpllRst;
  
  U_XBAR : entity work.AxiLiteCrossbar
    generic map (
      TPD_G              => TPD_G,
      DEC_ERROR_RESP_G   => AXI_RESP_DECERR_C,
      NUM_SLAVE_SLOTS_G  => 1,
      NUM_MASTER_SLOTS_G => AXI_CROSSBAR_MASTERS_CONFIG_C'length,
      MASTERS_CONFIG_G   => AXI_CROSSBAR_MASTERS_CONFIG_C)
    port map (
      axiClk           => axilClk,
      axiClkRst        => axilRst,
      sAxiWriteMasters(0) => axilWriteMaster,
      sAxiWriteSlaves (0) => axilWriteSlave,
      sAxiReadMasters (0) => axilReadMaster,
      sAxiReadSlaves  (0) => axilReadSlave,
      mAxiWriteMasters => mAxilWriteMasters,
      mAxiWriteSlaves  => mAxilWriteSlaves,
      mAxiReadMasters  => mAxilReadMasters,
      mAxiReadSlaves   => mAxilReadSlaves);

  U_PrbsTx : entity work.SsiPrbsTx
    generic map ( MASTER_AXI_STREAM_CONFIG_G => AXIS_CONFIG_C )
    port map ( mAxisClk            => amcClk,
               mAxisRst            => amcRst,
               mAxisMaster         => dmaObMaster,
               mAxisSlave          => dmaObSlave,
               locClk              => axilClk,
               axilReadMaster      => mAxilReadMasters(0),
               axilReadSlave       => mAxilReadSlaves (0),
               axilWriteMaster     => mAxilWriteMasters(0),
               axilWriteSlave      => mAxilWriteSlaves (0) );

  U_PrbxRx : entity work.SsiPrbsRx
    generic map ( MASTER_AXI_STREAM_CONFIG_G => AXIS_CONFIG_C )
    port map ( sAxisClk            => amcClk,
               sAxisRst            => amcRst,
               sAxisMaster         => dmaIbMaster,
               sAxisSlave          => dmaIbSlave,
               mAxisClk            => amcClk,
               axiClk              => axilClk,
               axiRst              => axilRst,
               axiReadMaster       => mAxilReadMasters(1),
               axiReadSlave        => mAxilReadSlaves (1),
               axiWriteMaster      => mAxilWriteMasters(1),
               axiWriteSlave       => mAxilWriteSlaves (1) );

  U_TenGigEth : entity work.TenGigEthGthUltraScale
    generic map (
      EN_AXI_REG_G => true )
    port map (
      localMac            => MACADDR,
      -- Streaming DMA Interface 
      dmaClk              => amcClk,
      dmaRst              => amcRst,
      dmaIbMaster         => dmaIbMaster,
      dmaIbSlave          => dmaIbSlave,
      dmaObMaster         => dmaObMaster,
      dmaObSlave          => dmaObSlave,
      -- Misc. Signals
      extRst              => '0',
      coreClk             => amcClk,
      phyClk              => open,
      phyRst              => open,
      phyReady            => linkUp,
      --
      qplllock            => qplllock,
      qplloutclk          => qplloutclk,
      qplloutrefclk       => qplloutrefclk,
      qpllRst             => iqpllRst,
      -- MGT Ports
      gtTxP               => amcTxP,
      gtTxN               => amcTxN,
      gtRxP               => amcRxP,
      gtRxN               => amcRxN );

     
  ibMaster <= regMaster;
  regSlave <= ibSlave;
  
  --
  --  Parse amcOb stream for register transactions or obTrig
  --
  acomb : process ( fifoRst, a, obMaster, regReadSlave, regWriteSlave, regSlave ) is
    variable v   : ARegType;
    variable reg : RegTransactionType;
  begin
    v := a;

    if fifoRst = '1' then
      v := AREG_INIT_C;
    end if;

    ain <= v;

    regMaster  <= a.regmaster;
    obSlave    <= v.regslave;
    status     <= a.status;

  end process;

  aseq : process (obClk) is
  begin
    if rising_edge(obClk) then
      a <= ain;
    end if;
  end process;
  
end top_level_app;
