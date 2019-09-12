-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : HSRepeater.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-08
-- Last update: 2017-07-07
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
use work.i2cPkg.all;

entity HSRepeater is
   generic (
     AXI_ERROR_RESP_G : slv(1 downto 0) := AXI_RESP_DECERR_C;
     AXI_BASEADDR_G   : slv(31 downto 0) := (others=>'0') );
   port (
      ----------------------
      -- Top Level Interface
      ----------------------
      axilClk            : in    sl;
      axilRst            : in    sl;
      axilReadMaster     : in    AxiLiteReadMasterType;
      axilReadSlave      : out   AxiLiteReadSlaveType;
      axilWriteMaster    : in    AxiLiteWriteMasterType;
      axilWriteSlave     : out   AxiLiteWriteSlaveType;
      --
      hsrScl             : inout Slv3Array(1 downto 0);
      hsrSda             : inout Slv3Array(1 downto 0) );
end HSRepeater;

architecture mapping of HSRepeater is

  constant DEVICE_MAP_C : I2cAxiLiteDevArray(0 downto 0) := (
    0 => MakeI2cAxiLiteDevType( "1011000", 8, 8, '0' ) );

  constant AXI_CROSSBAR_MASTERS_CONFIG_C : AxiLiteCrossbarMasterConfigArray(5 downto 0) := (
     0 => (
       baseAddr     => AXI_BASEADDR_G+x"00000000",
       addrBits     => 16,
       connectivity => x"FFFF"),
     1 => (
       baseAddr     => AXI_BASEADDR_G+x"00010000",
       addrBits     => 16,
       connectivity => x"FFFF"),
     2 => (
       baseAddr     => AXI_BASEADDR_G+x"00020000",
       addrBits     => 16,
       connectivity => x"FFFF"),
     3 => (
       baseAddr     => AXI_BASEADDR_G+x"00030000",
       addrBits     => 16,
       connectivity => x"FFFF"),
     4 => (
       baseAddr     => AXI_BASEADDR_G+x"00040000",
       addrBits     => 16,
       connectivity => x"FFFF"),
     5 => (
       baseAddr     => AXI_BASEADDR_G+x"00050000",
       addrBits     => 16,
       connectivity => x"FFFF") );

  
   signal axilReadMasters  : AxiLiteReadMasterArray (5 downto 0);
   signal axilReadSlaves   : AxiLiteReadSlaveArray  (5 downto 0);
   signal axilWriteMasters : AxiLiteWriteMasterArray(5 downto 0);
   signal axilWriteSlaves  : AxiLiteWriteSlaveArray (5 downto 0);

begin

  U_XBAR : entity work.AxiLiteCrossbar
    generic map (
      DEC_ERROR_RESP_G   => AXI_ERROR_RESP_G,
      NUM_SLAVE_SLOTS_G  => 1,
      NUM_MASTER_SLOTS_G => 6,
      MASTERS_CONFIG_G   => AXI_CROSSBAR_MASTERS_CONFIG_C)
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

  GEN_AMC : for i in 0 to 1 generate
    GEN_I2C : for j in 0 to 2 generate
      U_I2C : entity work.AxiI2cRegMaster
        generic map ( DEVICE_MAP_G => DEVICE_MAP_C,
                      AXI_CLK_FREQ_G => 156.25E+6 )
        port map ( scl => hsrScl(i)(j),
                   sda => hsrSda(i)(j),
                   axiReadMaster   => axilReadMasters (i*3+j),
                   axiReadSlave    => axilReadSlaves  (i*3+j),
                   axiWriteMaster  => axilWriteMasters(i*3+j),
                   axiWriteSlave   => axilWriteSlaves (i*3+j),
                   axiClk          => axilClk,
                   axiRst          => axilRst );
    end generate;
  end generate;

end mapping;
