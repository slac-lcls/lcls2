-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DebugBridgeWrapper.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2018-06-27
-- Last update: 2018-06-28
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 XPM Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the
-- top-level directory of this distribution and at:
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
-- No part of 'LCLS2 XPM Core', including this file,
-- may be copied, modified, propagated, or distributed except according to
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;

entity DebugBridgeWrapper is
   port (
      ----------------------
      -- Top Level Interface
      ----------------------
      axilClk            : in    sl;
      axilRst            : in    sl;
      axilReadMaster     : in    AxiLiteReadMasterType;
      axilReadSlave      : out   AxiLiteReadSlaveType;
      axilWriteMaster    : in    AxiLiteWriteMasterType;
      axilWriteSlave     : out   AxiLiteWriteSlaveType );
end DebugBridgeWrapper;

architecture mapping of DebugBridgeWrapper is

  COMPONENT debug_bridge_0
    PORT (
      s_axi_aclk : IN STD_LOGIC;
      s_axi_aresetn : IN STD_LOGIC;
      S_AXI_araddr : IN STD_LOGIC_VECTOR(4 DOWNTO 0);
      S_AXI_arprot : IN STD_LOGIC_VECTOR(2 DOWNTO 0);
      S_AXI_arready : OUT STD_LOGIC;
      S_AXI_arvalid : IN STD_LOGIC;
      S_AXI_awaddr : IN STD_LOGIC_VECTOR(4 DOWNTO 0);
      S_AXI_awprot : IN STD_LOGIC_VECTOR(2 DOWNTO 0);
      S_AXI_awready : OUT STD_LOGIC;
      S_AXI_awvalid : IN STD_LOGIC;
      S_AXI_bready : IN STD_LOGIC;
      S_AXI_bresp : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
      S_AXI_bvalid : OUT STD_LOGIC;
      S_AXI_rdata : OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
      S_AXI_rready : IN STD_LOGIC;
      S_AXI_rresp : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
      S_AXI_rvalid : OUT STD_LOGIC;
      S_AXI_wdata : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
      S_AXI_wready : OUT STD_LOGIC;
      S_AXI_wstrb : IN STD_LOGIC_VECTOR(3 DOWNTO 0);
      S_AXI_wvalid : IN STD_LOGIC
      );
  END COMPONENT;

  signal aresetn : sl;

begin

  aresetn <= not axilRst;
  
  U_DebugBridge : debug_bridge_0
    port map ( s_axi_aclk    => axilClk,
               s_axi_aresetn => aresetn,
               S_AXI_araddr  => axilReadMaster .araddr(4 downto 0),
               S_AXI_arprot  => axilReadMaster .arprot,
               S_AXI_arready => axilReadSlave  .arready,
               S_AXI_arvalid => axilReadMaster .arvalid,
               S_AXI_awaddr  => axilWriteMaster.awaddr(4 downto 0),
               S_AXI_awprot  => axilWriteMaster.awprot,
               S_AXI_awready => axilWriteSlave .awready,
               S_AXI_awvalid => axilWriteMaster.awvalid,
               S_AXI_bready  => axilWriteMaster.bready,
               S_AXI_bresp   => axilWriteSlave .bresp,
               S_AXI_bvalid  => axilWriteSlave .bvalid,
               S_AXI_rdata   => axilReadSlave  .rdata,
               S_AXI_rready  => axilReadMaster .rready,
               S_AXI_rresp   => axilReadSlave  .rresp,
               S_AXI_rvalid  => axilReadSlave  .rvalid,
               S_AXI_wdata   => axilWriteMaster.wdata,
               S_AXI_wready  => axilWriteSlave .wready,
               S_AXI_wstrb   => axilWriteMaster.wstrb,
               S_AXI_wvalid  => axilWriteMaster.wvalid
               );

end mapping;
