-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiDsTenGigEthApp.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-10
-- Last update: 2017-04-10
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

use work.StdRtlPkg.all;
use work.AxiStreamPkg.all;
use work.AxiLitePkg.all;
use work.DtiPkg.all;

entity DtiDsTenGigEthApp is
   generic (
      TPD_G               : time                := 1 ns;
      ID_G                : integer             := 0 );
   port (
     amcClk          : in  sl;
     amcRst          : in  sl;
     amcRxP          : in  sl;
     amcRxN          : in  sl;
     amcTxP          : out sl;
     amcTxN          : out sl;
     fifoRst         : in  sl;
     qplllock        : in  sl;
     qplloutclk      : in  sl;
     qplloutrefclk   : in  sl;
     qpllRst         : out sl;
     --
     axilClk         : in  sl;
     axilRst         : in  sl;
     axilReadMaster  : in  AxiLiteReadMasterType;
     axilReadSlave   : out AxiLiteReadSlaveType;
     axilWriteMaster : in  AxiLiteWriteMasterType;
     axilWriteSlave  : out AxiLiteWriteSlaveType;
     --  App Interface
     ibRst           : in  sl;
     linkUp          : out sl;
     rxErr           : out sl;
     full            : out sl;
     --
     obClk           : in  sl;
     obMaster        : in  AxiStreamMasterType;
     obSlave         : out AxiStreamSlaveType );
end DtiDsTenGigEthApp;

architecture rtl of DtiDsTenGigEthApp is

  constant ID_C    : slv( 7 downto 0) := toSlv(ID_G,8);
  constant MACADDR : slv(47 downto 0) := (x"0203005645" & ID_C);

  --  IP/UDP header needs to know the length of the payload
  --  Assume 8kB (maybe pad)
  --
  constant LOCAL_IP   : slv(31 downto 0) := (x"c0a800" & ID_C);
  constant LOCAL_MAC  : slv(47 downto 0) := (x"0002030057" & ID_C);
  constant REMOTE_IP  : slv(31 downto 0) := (x"efff31" & ID_C);
  constant REMOTE_MAC : slv(47 downto 0) := (x"01005e7f31" & ID_C);

  signal ibUdpMaster : AxiStreamMasterType;
  signal ibUdpSlave  : AxiStreamSlaveType;
  signal obUdpMaster : AxiStreamMasterType;
  signal obUdpSlave  : AxiStreamSlaveType;
  signal obEthMaster : AxiStreamMasterType;
  signal obEthSlave  : AxiStreamSlaveType;

begin

  U_Fifo : entity work.AxiStreamFifo
    generic map (
      SLAVE_AXI_CONFIG_G  => US_OB_CONFIG_C,
      MASTER_AXI_CONFIG_G => US_OB_CONFIG_C )
    port map ( 
      -- Slave Port
      sAxisClk    => obClk,
      sAxisRst    => fifoRst,
      sAxisMaster => obMaster,
      sAxisSlave  => obSlave,
      -- Master Port
      mAxisClk    => amcClk,
      mAxisRst    => fifoRst,
      mAxisMaster => ibUdpMaster,
      mAxisSlave  => ibUdpSlave );

  U_Udp : entity work.UdpEngineTx
    generic map ( PORT_G(0) => 11000 )
    port map ( obUdpMaster   => obUdpMaster,
               obUdpSlave    => obUdpSlave,
               localIp       => LOCAL_IP,
               remotePort(0) => toSlv(11000,16),
               remoteIp  (0) => REMOTE_IP,
               remoteMac (0) => REMOTE_MAC,
               ibMasters (0) => ibUdpMaster,
               ibSlaves  (0) => ibUdpSlave,
               clk           => amcClk,
               rst           => fifoRst );

  U_Ip : entity work.IpV4EngineTx
    port map ( localMac             => LOCAL_MAC,
               obIpv4Master         => obEthMaster,
               obIpv4Slave          => obEthSlave,
               localhostMaster      => open,
               localhostSlave       => AXI_STREAM_SLAVE_INIT_C,
               obProtocolMasters(0) => obUdpMaster,
               obProtocolSlaves (0) => obUdpSlave,
               clk                  => amcClk,
               rst                  => amcRst );
  
  U_TenGigEth : entity work.TenGigEthGthUltraScale
    generic map (
      EN_AXI_REG_G => true )
    port map (
      localMac            => MACADDR,
      -- Streaming DMA Interface 
      dmaClk              => amcClk,
      dmaRst              => fifoRst,
      dmaIbMaster         => open,
      dmaIbSlave          => AXI_STREAM_SLAVE_INIT_C,
      dmaObMaster         => obEthMaster,
      dmaObSlave          => obEthSlave,
      -- Slave AXI-Lite Interface 
      axiLiteClk          => axilClk,
      axiLiteRst          => axilRst,
      axiLiteReadMaster   => axilReadMaster,
      axiLiteReadSlave    => axilReadSlave,
      axiLiteWriteMaster  => axilWriteMaster,
      axiLiteWriteSlave   => axilWriteSlave,
      -- Misc. Signals
      extRst              => '0',
      coreClk             => amcClk,
      phyClk              => open,
      phyRst              => open,
      phyReady            => linkUp,
      -- Transceiver Debug Interface
      --gtTxPreCursor       : in  slv(4 downto 0)                                := "00000";
      --gtTxPostCursor      : in  slv(4 downto 0)                                := "00000";
      --gtTxDiffCtrl        : in  slv(3 downto 0)                                := "1110";
      --gtRxPolarity        : in  sl                                             := '0';
      --gtTxPolarity        : in  sl                                             := '0';
      -- Quad PLL Ports
      qplllock            => qplllock,
      qplloutclk          => qplloutclk,
      qplloutrefclk       => qplloutrefclk,
      qpllRst             => qpllRst,
      -- MGT Ports
      gtTxP               => amcTxP,
      gtTxN               => amcTxN,
      gtRxP               => amcRxP,
      gtRxN               => amcRxN );

end rtl;
