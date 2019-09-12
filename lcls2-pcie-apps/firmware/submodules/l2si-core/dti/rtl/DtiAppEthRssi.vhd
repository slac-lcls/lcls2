-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiAppEthRssi.vhd
-- Author     : Larry Ruckman  <ruckman@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2016-02-23
-- Last update: 2018-03-28
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

entity DtiAppEthRssi is
   generic (
      TPD_G            : time             := 1 ns;
      AXI_ERROR_RESP_G : slv(1 downto 0)  := AXI_RESP_DECERR_C;
      AXI_BASE_ADDR_G  : slv(31 downto 0) := (others => '0');
      DEBUG_G          : boolean          := false );
   port (
      -- Slave AXI-Lite Interface
      axilClk          : in  sl;
      axilRst          : in  sl;
      axilReadMaster   : in  AxiLiteReadMasterType;
      axilReadSlave    : out AxiLiteReadSlaveType;
      axilWriteMaster  : in  AxiLiteWriteMasterType;
      axilWriteSlave   : out AxiLiteWriteSlaveType;
      -- Application Interface
      obAppMaster      : in  AxiStreamMasterType;  -- out to ethernet
      obAppSlave       : out AxiStreamSlaveType;
      ibAppMaster      : out AxiStreamMasterType;  -- in from ethernet
      ibAppSlave       : in  AxiStreamSlaveType;
      -- Interface to UDP Server engines
      obServerMaster   : in  AxiStreamMasterType;  -- in from ethernet
      obServerSlave    : out AxiStreamSlaveType;
      ibServerMaster   : out AxiStreamMasterType;  -- out to ethernet
      ibServerSlave    : in  AxiStreamSlaveType);
end DtiAppEthRssi;

architecture mapping of DtiAppEthRssi is

  constant NSTREAMS_C : integer := 1;
   --  APP Streams
   constant ETH_AXIS_CONFIG_C : AxiStreamConfigType := ssiAxiStreamConfig(8, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 8);  -- Use 8 tDest bits
   constant APP_AXIS_CONFIG_C  : AxiStreamConfigArray(NSTREAMS_C-1 downto 0) := (others => ETH_AXIS_CONFIG_C);
   signal rssiIbMasters : AxiStreamMasterArray(NSTREAMS_C-1 downto 0);
   signal rssiIbSlaves  : AxiStreamSlaveArray (NSTREAMS_C-1 downto 0);
   signal rssiObMasters : AxiStreamMasterArray(NSTREAMS_C-1 downto 0);
   signal rssiObSlaves  : AxiStreamSlaveArray (NSTREAMS_C-1 downto 0);

begin

   ----------------------------
   -- Software's RSSI Server --
   ----------------------------
   U_RssiServer : entity work.RssiCoreWrapper
      generic map (
         TPD_G               => TPD_G,
         APP_STREAMS_G       => NSTREAMS_C,
         APP_STREAM_ROUTES_G => (
            0                => X"00"),  -- TDEST 0 routed to stream 0
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
         sTspAxisMaster_i  => obServerMaster,
         sTspAxisSlave_o   => obServerSlave,
         mTspAxisMaster_o  => ibServerMaster,
         mTspAxisSlave_i   => ibServerSlave,
         -- High level  Application side interface
         openRq_i          => '1',  -- Automatically start the connection without debug SRP channel
         closeRq_i         => '0',
         inject_i          => '0',
         -- AXI-Lite Interface
         axiClk_i          => axilClk,
         axiRst_i          => axilRst,
         axilReadMaster    => axilReadMaster,
         axilReadSlave     => axilReadSlave,
         axilWriteMaster   => axilWriteMaster,
         axilWriteSlave    => axilWriteSlave);

   --------------------------------
   -- App Path: TDEST = 0x0
   --------------------------------
   ibAppMaster     <= rssiObMasters(0);
   rssiObSlaves(0) <= ibAppSlave;
   
   U_IbLimiter : entity work.SsiFrameLimiter
      generic map (
         TPD_G               => TPD_G,
         EN_TIMEOUT_G        => true,
         MAXIS_CLK_FREQ_G    => AXI_CLK_FREQ_C,
         TIMEOUT_G           => TIMEOUT_C,
         FRAME_LIMIT_G       => (4096/16),  -- 4kB limit
         COMMON_CLK_G        => true,
         SLAVE_FIFO_G        => false,
         MASTER_FIFO_G       => false,
         SLAVE_AXI_CONFIG_G  => ETH_AXIS_CONFIG_C,
         MASTER_AXI_CONFIG_G => ETH_AXIS_CONFIG_C)      
      port map (
         -- Slave Port
         sAxisClk    => axilClk,
         sAxisRst    => axilRst,
         sAxisMaster => obAppMaster,
         sAxisSlave  => obAppSlave,
         -- Master Port
         mAxisClk    => axilClk,
         mAxisRst    => axilRst,
         mAxisMaster => rssiIbMasters(0),
         mAxisSlave  => rssiIbSlaves (0));   
   
end mapping;
