-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiCoreEth.vhd
-- Author     : Larry Ruckman  <ruckman@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-21
-- Last update: 2018-03-26
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Common Carrier Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Common Carrier Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.SsiPkg.all;
use work.EthMacPkg.all;
use work.DtiPkg.all;

entity DtiCoreEth is
   generic (
      TPD_G            : time            := 1 ns;
      NAPP_LINKS_G     : integer         := 1;
      ETH_ADDR_C      : slv(31 downto 0) := x"00000000";
      AXI_ERROR_RESP_G : slv(1 downto 0) := AXI_RESP_DECERR_C);
   port (
      -- Local Configuration and status
      localMac          : in  slv(47 downto 0);  --  big-Endian configuration
      localIp           : in  slv(31 downto 0);  --  big-Endian configuration   
      ethPhyReady       : out sl;
      -- Master AXI-Lite Interface
      mAxilReadMasters  : out AxiLiteReadMasterArray (1 downto 0);
      mAxilReadSlaves   : in  AxiLiteReadSlaveArray  (1 downto 0);
      mAxilWriteMasters : out AxiLiteWriteMasterArray(1 downto 0);
      mAxilWriteSlaves  : in  AxiLiteWriteSlaveArray (1 downto 0);
      -- AXI-Lite Interface
      axilClk           : in  sl;
      axilRst           : in  sl;
      axilReadMaster    : in  AxiLiteReadMasterType;
      axilReadSlave     : out AxiLiteReadSlaveType;
      axilWriteMaster   : in  AxiLiteWriteMasterType;
      axilWriteSlave    : out AxiLiteWriteSlaveType;
      -- App Ethernet Interface
      obAppMasters      : in  AxiStreamMasterArray(NAPP_LINKS_G-1 downto 0);
      obAppSlaves       : out AxiStreamSlaveArray (NAPP_LINKS_G-1 downto 0);
      ibAppMasters      : out AxiStreamMasterArray(NAPP_LINKS_G-1 downto 0);
      ibAppSlaves       : in  AxiStreamSlaveArray (NAPP_LINKS_G-1 downto 0);
      ----------------
      -- Core Ports --
      ----------------   
      -- ETH Ports
      ethRxP           : in  slv(3 downto 0);
      ethRxN           : in  slv(3 downto 0);
      ethTxP           : out slv(3 downto 0);
      ethTxN           : out slv(3 downto 0);
      ethClkP          : in  sl;
      ethClkN          : in  sl);
end DtiCoreEth;

architecture mapping of DtiCoreEth is

   constant SERVER_SIZE_C : positive := 2+NAPP_LINKS_G;
   constant CLIENT_SIZE_C : positive := 1;

   constant NUM_AXI_MASTERS_C : natural := 3+NAPP_LINKS_G;

   constant PHY_INDEX_C    : natural := 0;
   constant UDP_INDEX_C    : natural := 1;
   constant RSSI_INDEX_C   : natural := 2;
   constant APP_INDEX_C    : natural := 3;

   constant PHY_ADDR_C    : slv(31 downto 0) := (ETH_ADDR_C + x"00000000");
   constant UDP_ADDR_C    : slv(31 downto 0) := (ETH_ADDR_C + x"00010000");
   constant RSSI_ADDR_C   : slv(31 downto 0) := (ETH_ADDR_C + x"00020000");
   constant APP_ADDR_C    : slv(31 downto 0) := (ETH_ADDR_C + x"00030000");
   
   function genAxiConfig return AxiLiteCrossbarMasterConfigArray is
     variable result : AxiLiteCrossbarMasterConfigArray(NUM_AXI_MASTERS_C-1 downto 0);
     variable i      : integer;
   begin
     result(PHY_INDEX_C) := ( baseAddr     => PHY_ADDR_C,
                              addrBits     => 16,
                              connectivity => X"FFFF" );
     result(UDP_INDEX_C) := ( baseAddr     => UDP_ADDR_C,
                              addrBits     => 16,
                              connectivity => X"FFFF" );
     result(RSSI_INDEX_C) := ( baseAddr     => RSSI_ADDR_C,
                              addrBits     => 16,
                              connectivity => X"FFFF" );
     result(NUM_AXI_MASTERS_C-1 downto APP_INDEX_C) :=
       genAxiLiteConfig(NAPP_LINKS_G, APP_ADDR_C, 16, 8);
     return result;
   end function;     

   constant AXI_CONFIG_C : AxiLiteCrossbarMasterConfigArray(NUM_AXI_MASTERS_C-1 downto 0) := genAxiConfig;

   function ServerPorts return PositiveArray is
      variable retConf   : PositiveArray(SERVER_SIZE_C-1 downto 0);
      variable baseIndex : positive;
   begin
      baseIndex := 8192;
      for i in SERVER_SIZE_C-1 downto 0 loop
         retConf(i) := baseIndex+i;
      end loop;
      return retConf;
   end function;

   function ClientPorts return PositiveArray is
      variable retConf   : PositiveArray(CLIENT_SIZE_C-1 downto 0);
      variable baseIndex : positive;
   begin
      baseIndex := 8192+SERVER_SIZE_C;
      for i in CLIENT_SIZE_C-1 downto 0 loop
         retConf(i) := baseIndex+i;
      end loop;
      return retConf;
   end function;

   signal ibMacMaster : AxiStreamMasterType;
   signal ibMacSlave  : AxiStreamSlaveType;
   signal obMacMaster : AxiStreamMasterType;
   signal obMacSlave  : AxiStreamSlaveType;

   signal obServerMasters : AxiStreamMasterArray(SERVER_SIZE_C-1 downto 0);
   signal obServerSlaves  : AxiStreamSlaveArray (SERVER_SIZE_C-1 downto 0);
   signal ibServerMasters : AxiStreamMasterArray(SERVER_SIZE_C-1 downto 0);
   signal ibServerSlaves  : AxiStreamSlaveArray (SERVER_SIZE_C-1 downto 0);

   signal obClientMasters : AxiStreamMasterArray(CLIENT_SIZE_C-1 downto 0);
   signal obClientSlaves  : AxiStreamSlaveArray (CLIENT_SIZE_C-1 downto 0);
   signal ibClientMasters : AxiStreamMasterArray(CLIENT_SIZE_C-1 downto 0);
   signal ibClientSlaves  : AxiStreamSlaveArray (CLIENT_SIZE_C-1 downto 0);

   signal axilWriteMasters : AxiLiteWriteMasterArray(NUM_AXI_MASTERS_C-1 downto 0);
   signal axilWriteSlaves  : AxiLiteWriteSlaveArray (NUM_AXI_MASTERS_C-1 downto 0);
   signal axilReadMasters  : AxiLiteReadMasterArray (NUM_AXI_MASTERS_C-1 downto 0);
   signal axilReadSlaves   : AxiLiteReadSlaveArray  (NUM_AXI_MASTERS_C-1 downto 0);

   constant ETH_AXIS_CONFIG_C : AxiStreamConfigType := ssiAxiStreamConfig(8, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 8);  -- Use 8 tDest bits
   constant APP_AXIS_CONFIG_C  : AxiStreamConfigArray(1 downto 0) := (others => ETH_AXIS_CONFIG_C);
   signal rssiIbMasters : AxiStreamMasterArray(1 downto 0);
   signal rssiIbSlaves  : AxiStreamSlaveArray (1 downto 0);
   signal rssiObMasters : AxiStreamMasterArray(1 downto 0);
   signal rssiObSlaves  : AxiStreamSlaveArray (1 downto 0);

   
   signal phyReady : sl;

   constant DEBUG_C : boolean := true;

   component ila_0
     port ( clk   : in sl;
            probe0 : in slv(255 downto 0) );
   end component;
   
begin

   GEN_DEBUG : if DEBUG_C generate
     U_ILA : ila_0
       port map ( clk       => axilClk,
                  probe0(0) => obAppMasters(0).tValid,
                  probe0(1) => obAppMasters(0).tLast,
                  probe0(2) => obAppMasters(1).tValid,
                  probe0(3) => obAppMasters(1).tLast,
                  probe0(4) => obAppMasters(2).tValid,
                  probe0(5) => obAppMasters(2).tLast,
                  probe0(6) => obAppMasters(3).tValid,
                  probe0(7) => obAppMasters(3).tLast,
                  probe0(8) => obServerMasters(0).tValid,
                  probe0(9) => obServerMasters(0).tLast,
                  probe0(10) => obServerMasters(1).tValid,
                  probe0(11) => obServerMasters(1).tLast,
                  probe0(12) => obServerMasters(2).tValid,
                  probe0(13) => obServerMasters(2).tLast,
                  probe0(14) => obServerMasters(3).tValid,
                  probe0(15) => obServerMasters(3).tLast,
                  probe0(16) => ibServerMasters(0).tValid,
                  probe0(17) => ibServerMasters(0).tLast,
                  probe0(18) => ibServerMasters(1).tValid,
                  probe0(19) => ibServerMasters(1).tLast,
                  probe0(20) => ibServerMasters(2).tValid,
                  probe0(21) => ibServerMasters(2).tLast,
                  probe0(22) => ibServerMasters(3).tValid,
                  probe0(23) => ibServerMasters(3).tLast,
                  probe0(24) => obServerSlaves (0).tReady,
                  probe0(25) => obServerSlaves (1).tReady,
                  probe0(26) => obServerSlaves (2).tReady,
                  probe0(27) => obServerSlaves (3).tReady,
                  probe0(28) => ibServerSlaves (0).tReady,
                  probe0(29) => ibServerSlaves (1).tReady,
                  probe0(30) => ibServerSlaves (2).tReady,
                  probe0(31) => ibServerSlaves (3).tReady,
                  probe0(255 downto 32) => (others=>'0') );
   end generate;
                  
   --------------------------
   -- AXI-Lite: Crossbar Core
   --------------------------  
   U_XBAR : entity work.AxiLiteCrossbar
      generic map (
         TPD_G              => TPD_G,
         DEC_ERROR_RESP_G   => AXI_ERROR_RESP_G,
         NUM_SLAVE_SLOTS_G  => 1,
         NUM_MASTER_SLOTS_G => NUM_AXI_MASTERS_C,
         MASTERS_CONFIG_G   => AXI_CONFIG_C)
      port map (
         axiClk              => axilClk,
         axiClkRst           => axilRst,
         sAxiWriteMasters(0) => axilWriteMaster,
         sAxiWriteSlaves(0)  => axilWriteSlave,
         sAxiReadMasters(0)  => axilReadMaster,
         sAxiReadSlaves(0)   => axilReadSlave,
         mAxiWriteMasters    => axilWriteMasters,
         mAxiWriteSlaves     => axilWriteSlaves,
         mAxiReadMasters     => axilReadMasters,
         mAxiReadSlaves      => axilReadSlaves);

   ----------------------
   -- 10 GigE ETH Module
   ----------------------
   U_Eth : entity work.XauiGthUltraScaleWrapper
      generic map (
         TPD_G            => TPD_G,
         EN_WDT_G         => true,
         -- ETH Configurations
--         XAUI_20GIGE_G    => false,
--         REF_CLK_FREQ_G   => AXI_CLK_FREQ_C,
         -- AXI-Lite Configurations
         AXI_ERROR_RESP_G => AXI_ERROR_RESP_G,
         -- AXI Streaming Configurations
         AXIS_CONFIG_G    => EMAC_AXIS_CONFIG_C)
      port map (
         -- Local Configurations
         localMac           => localMac,
         -- Streaming DMA Interface 
         dmaClk             => axilClk,
         dmaRst             => axilRst,
         dmaIbMaster        => obMacMaster,
         dmaIbSlave         => obMacSlave,
         dmaObMaster        => ibMacMaster,
         dmaObSlave         => ibMacSlave,
         -- Slave AXI-Lite Interface 
         axiLiteClk         => axilClk,
         axiLiteRst         => axilRst,
         axiLiteReadMaster  => axilReadMasters (PHY_INDEX_C),
         axiLiteReadSlave   => axilReadSlaves  (PHY_INDEX_C),
         axiLiteWriteMaster => axilWriteMasters(PHY_INDEX_C),
         axiLiteWriteSlave  => axilWriteSlaves (PHY_INDEX_C),
         -- Misc. Signals
         extRst             => axilRst,
         stableClk          => axilClk,
         phyReady           => phyReady,
         -- Transceiver Debug Interface
         gtTxPreCursor      => (others => '0'),  -- 0 dB
         gtTxPostCursor     => (others => '0'),  -- 0 dB
         gtTxDiffCtrl       => x"FFFF",          -- 1.080 V
         gtRxPolarity       => x"0",
         gtTxPolarity       => x"0",
         -- MGT Clock Port (156.25 MHz)
         gtClkP             => ethClkP,
         gtClkN             => ethClkN,
         -- MGT Ports
         gtTxP              => ethTxP,
         gtTxN              => ethTxN,
         gtRxP              => ethRxP,
         gtRxN              => ethRxN);

   U_Sync : entity work.Synchronizer
      generic map (
         TPD_G => TPD_G)
      port map (
         clk     => axilClk,
         dataIn  => phyReady,
         dataOut => ethPhyReady);    

   ----------------------
   -- IPv4/ARP/UDP Engine
   ----------------------
   U_UdpEngineWrapper : entity work.UdpEngineWrapper
      generic map (
         -- Simulation Generics
         TPD_G            => TPD_G,
         -- UDP Server Generics
         SERVER_EN_G      => true,
         SERVER_SIZE_G    => SERVER_SIZE_C,
         SERVER_PORTS_G   => ServerPorts,
         -- UDP Client Generics
         CLIENT_EN_G      => true,
         CLIENT_SIZE_G    => CLIENT_SIZE_C,
         CLIENT_PORTS_G   => ClientPorts,
         AXI_ERROR_RESP_G => AXI_ERROR_RESP_G,
         -- IPv4/ARP Generics
         CLK_FREQ_G       => AXI_CLK_FREQ_C,  -- In units of Hz
         COMM_TIMEOUT_G   => 30,  -- In units of seconds, Client's Communication timeout before re-ARPing
         VLAN_G           => false)     -- no VLAN       
      port map (
         -- Local Configurations
         localMac        => localMac,
         localIp         => localIp,
         -- Interface to Ethernet Media Access Controller (MAC)
         obMacMaster     => obMacMaster,
         obMacSlave      => obMacSlave,
         ibMacMaster     => ibMacMaster,
         ibMacSlave      => ibMacSlave,
         -- Interface to UDP Server engine(s)
         obServerMasters => obServerMasters,
         obServerSlaves  => obServerSlaves,
         ibServerMasters => ibServerMasters,
         ibServerSlaves  => ibServerSlaves,
         -- Interface to UDP Client engine(s)
         obClientMasters => obClientMasters,
         obClientSlaves  => obClientSlaves,
         ibClientMasters => ibClientMasters,
         ibClientSlaves  => ibClientSlaves,
         -- AXI-Lite Interface
         axilReadMaster  => axilReadMasters (UDP_INDEX_C),
         axilReadSlave   => axilReadSlaves  (UDP_INDEX_C),
         axilWriteMaster => axilWriteMasters(UDP_INDEX_C),
         axilWriteSlave  => axilWriteSlaves (UDP_INDEX_C),
         -- Clock and Reset
         clk             => axilClk,
         rst             => axilRst);

   --------------------------------------------------
   -- Legacy AXI-Lite Master without RSSI Server@8192
   --------------------------------------------------
   U_SRPv0 : entity work.SrpV0AxiLite
      generic map (
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => true,
         EN_32BIT_ADDR_G     => true,
         BRAM_EN_G           => true,
         GEN_SYNC_FIFO_G     => true,
         AXI_STREAM_CONFIG_G => EMAC_AXIS_CONFIG_C)   
      port map (
         -- Streaming Slave (Rx) Interface (sAxisClk domain) 
         sAxisClk            => axilClk,
         sAxisRst            => axilRst,
         sAxisMaster         => obServerMasters(0),
         sAxisSlave          => obServerSlaves (0),
         -- Streaming Master (Tx) Data Interface (mAxisClk domain)
         mAxisClk            => axilClk,
         mAxisRst            => axilRst,
         mAxisMaster         => ibServerMasters(0),
         mAxisSlave          => ibServerSlaves (0),
         -- AXI Lite Bus (axiLiteClk domain)
         axiLiteClk          => axilClk,
         axiLiteRst          => axilRst,
         mAxiLiteReadMaster  => mAxilReadMasters (0),
         mAxiLiteReadSlave   => mAxilReadSlaves  (0),
         mAxiLiteWriteMaster => mAxilWriteMasters(0),
         mAxiLiteWriteSlave  => mAxilWriteSlaves (0));   

   -----------------------------------------------
   -- Software's RSSI Server Interface@8193
   -----------------------------------------------
   U_RssiServer : entity work.RssiCoreWrapper
      generic map (
         TPD_G               => TPD_G,
         APP_STREAMS_G       => 2,
         APP_STREAM_ROUTES_G => (
            0                => X"00",  -- TDEST 0 routed to stream 0 (SRPv3)
            1                => X"01"), -- TDEST 1 routed to stream 1 (loopback)
         CLK_FREQUENCY_G     => AXI_CLK_FREQ_C,
         TIMEOUT_UNIT_G      => TIMEOUT_C,
         SERVER_G            => true,
         RETRANSMIT_ENABLE_G => true,
         WINDOW_ADDR_SIZE_G  => WINDOW_ADDR_SIZE_C,
         MAX_NUM_OUTS_SEG_G  => (2**WINDOW_ADDR_SIZE_C),
         PIPE_STAGES_G       => 1,
         APP_AXIS_CONFIG_G   => APP_AXIS_CONFIG_C,
         TSP_AXIS_CONFIG_G   => EMAC_AXIS_CONFIG_C,
         MAX_RETRANS_CNT_G   => MAX_RETRANS_CNT_C,
         MAX_CUM_ACK_CNT_G   => MAX_CUM_ACK_CNT_C)
      port map (
         clk_i             => axilClk,
         rst_i             => axilRst,
         -- Application Layer Interface
         sAppAxisMasters_i => rssiIbMasters,
         sAppAxisSlaves_o  => rssiIbSlaves,
         mAppAxisMasters_o => rssiObMasters,
         mAppAxisSlaves_i  => rssiObSlaves,
         -- Transport Layer Interface
         sTspAxisMaster_i  => obServerMasters(1),
         sTspAxisSlave_o   => obServerSlaves (1),
         mTspAxisMaster_o  => ibServerMasters(1),
         mTspAxisSlave_i   => ibServerSlaves (1),
         -- High level  Application side interface
         openRq_i          => '1',  -- Automatically start the connection without debug SRP channel
         closeRq_i         => '0',
         inject_i          => '0',
         -- AXI-Lite Interface
         axiClk_i          => axilClk,
         axiRst_i          => axilRst,
         axilReadMaster    => axilReadMasters (RSSI_INDEX_C),
         axilReadSlave     => axilReadSlaves  (RSSI_INDEX_C),
         axilWriteMaster   => axilWriteMasters(RSSI_INDEX_C),
         axilWriteSlave    => axilWriteSlaves (RSSI_INDEX_C));

   ------------------------------------------------
   -- AXI-Lite Master with RSSI Server: TDEST = 0x0
   ------------------------------------------------
   U_SRPv3 : entity work.SrpV3AxiLite
      generic map (
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => true,
         GEN_SYNC_FIFO_G     => true,
         AXI_STREAM_CONFIG_G => ETH_AXIS_CONFIG_C)
      port map (
         -- AXIS Slave Interface (sAxisClk domain)
         sAxisClk         => axilClk,
         sAxisRst         => axilRst,
         sAxisMaster      => rssiObMasters(0),
         sAxisSlave       => rssiObSlaves(0),
         -- AXIS Master Interface (mAxisClk domain) 
         mAxisClk         => axilClk,
         mAxisRst         => axilRst,
         mAxisMaster      => rssiIbMasters(0),
         mAxisSlave       => rssiIbSlaves(0),
         -- Master AXI-Lite Interface (axilClk domain)
         axilClk          => axilClk,
         axilRst          => axilRst,
         mAxilReadMaster  => mAxilReadMasters (1),
         mAxilReadSlave   => mAxilReadSlaves  (1),
         mAxilWriteMaster => mAxilWriteMasters(1),
         mAxilWriteSlave  => mAxilWriteSlaves (1));

   --------------------------------
   -- Loopback Channel: TDEST = 0x1
   --------------------------------
   rssiIbMasters(1) <= rssiObMasters(1);
   rssiObSlaves (1) <= rssiIbSlaves (1);


   GEN_LINK : for i in 0 to NAPP_LINKS_G-1 generate
     -----------------------------------------------
     -- Software's RSSI Server Interface@8194+i
     -----------------------------------------------
     U_RssiServer : entity work.DtiAppEthRssi
       generic map (
         TPD_G            => TPD_G,
         AXI_ERROR_RESP_G => AXI_ERROR_RESP_G,
         AXI_BASE_ADDR_G  => AXI_CONFIG_C(APP_INDEX_C+i).baseAddr,
         DEBUG_G          => ite(i>0, false, true) ) 
       port map (
         -- Slave AXI-Lite Interface
         axilClk          => axilClk,
         axilRst          => axilRst,
         axilReadMaster   => axilReadMasters (APP_INDEX_C+i),
         axilReadSlave    => axilReadSlaves  (APP_INDEX_C+i),
         axilWriteMaster  => axilWriteMasters(APP_INDEX_C+i),
         axilWriteSlave   => axilWriteSlaves (APP_INDEX_C+i),
         -- Application Interface
         obAppMaster      => obAppMasters(i),
         obAppSlave       => obAppSlaves (i),
         ibAppMaster      => ibAppMasters(i),
         ibAppSlave       => ibAppSlaves (i),
         -- Interface to UDP Server engines
         obServerMaster   => obServerMasters(2+i),
         obServerSlave    => obServerSlaves (2+i),
         ibServerMaster   => ibServerMasters(2+i),
         ibServerSlave    => ibServerSlaves (2+i) );
   end generate;

   obClientSlaves              <= (others => AXI_STREAM_SLAVE_FORCE_C);
   ibClientMasters             <= (others => AXI_STREAM_MASTER_INIT_C);

end mapping;
