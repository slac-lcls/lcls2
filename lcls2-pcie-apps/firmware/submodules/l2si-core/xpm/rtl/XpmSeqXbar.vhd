-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmSeqXbar.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-25
-- Last update: 2018-03-24
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Timing Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Timing Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.TPGPkg.all;
use work.XpmSeqPkg.all;

entity XpmSeqXbar is
   generic (
      TPD_G            : time             := 1 ns;
      AXIL_BASEADDR_G  : slv(31 downto 0) := (others=>'0') );
   port (
      -- AXI-Lite Interface (on axiClk domain)
      axiClk          : in  sl;
      axiRst          : in  sl;
      axiReadMaster   : in  AxiLiteReadMasterType;
      axiReadSlave    : out AxiLiteReadSlaveType;
      axiWriteMaster  : in  AxiLiteWriteMasterType;
      axiWriteSlave   : out AxiLiteWriteSlaveType;
      -- Configuration/Status (on clk domain)
      clk             : in  sl;
      rst             : in  sl;
      status          : in  XpmSeqStatusType;
      config          : out XpmSeqConfigType );
end XpmSeqXbar;

architecture xbar of XpmSeqXbar is

   constant SEQSTATE_INDEX_C  : natural := 0;
   constant SEQJUMP_INDEX_C   : natural := 1;
   constant SEQMEM_INDEX_C    : natural := 2;
   constant NUM_AXI_MASTERS_C : natural := 3;

   constant AXI_CROSSBAR_MASTERS_CONFIG_C : AxiLiteCrossbarMasterConfigArray(NUM_AXI_MASTERS_C-1 downto 0) := genAxiLiteConfig( 3, AXIL_BASEADDR_G, 16, 14 );

   signal mAxilWriteMasters : AxiLiteWriteMasterArray(NUM_AXI_MASTERS_C-1 downto 0);
   signal mAxilWriteSlaves  : AxiLiteWriteSlaveArray (NUM_AXI_MASTERS_C-1 downto 0);
   signal mAxilReadMasters  : AxiLiteReadMasterArray (NUM_AXI_MASTERS_C-1 downto 0);
   signal mAxilReadSlaves   : AxiLiteReadSlaveArray  (NUM_AXI_MASTERS_C-1 downto 0);

   signal syncWriteMaster : AxiLiteWriteMasterType;
   signal syncWriteSlave  : AxiLiteWriteSlaveType;
   signal syncReadMaster  : AxiLiteReadMasterType;
   signal syncReadSlave   : AxiLiteReadSlaveType;

   signal mConfig : XpmSeqConfigArray(NUM_AXI_MASTERS_C-1 downto 0);

begin

  U_AxiLiteAsync : entity work.AxiLiteAsync
    generic map (
      TPD_G => TPD_G)
    port map (
      sAxiClk         => axiClk,
      sAxiClkRst      => axiRst,
      sAxiReadMaster  => axiReadMaster,
      sAxiReadSlave   => axiReadSlave,
      sAxiWriteMaster => axiWriteMaster,
      sAxiWriteSlave  => axiWriteSlave,
      mAxiClk         => clk,
      mAxiClkRst      => rst,
      mAxiReadMaster  => syncReadMaster,
      mAxiReadSlave   => syncReadSlave,
      mAxiWriteMaster => syncWriteMaster,
      mAxiWriteSlave  => syncWriteSlave );

   --------------------------
   -- AXI-Lite: Crossbar Core
   --------------------------  
   U_XBAR : entity work.AxiLiteCrossbar
      generic map (
         TPD_G              => TPD_G,
         NUM_SLAVE_SLOTS_G  => 1,
         NUM_MASTER_SLOTS_G => NUM_AXI_MASTERS_C,
         MASTERS_CONFIG_G   => AXI_CROSSBAR_MASTERS_CONFIG_C)
      port map (
         axiClk              => clk,
         axiClkRst           => rst,
         sAxiWriteMasters(0) => syncWriteMaster,
         sAxiWriteSlaves(0)  => syncWriteSlave,
         sAxiReadMasters(0)  => syncReadMaster,
         sAxiReadSlaves(0)   => syncReadSlave,
         mAxiWriteMasters    => mAxilWriteMasters,
         mAxiWriteSlaves     => mAxilWriteSlaves,
         mAxiReadMasters     => mAxilReadMasters,
         mAxiReadSlaves      => mAxilReadSlaves);

   U_SeqJumpReg : entity work.XpmSeqJumpReg
      port map (
         axiReadMaster  => mAxilReadMasters (SEQJUMP_INDEX_C),
         axiReadSlave   => mAxilReadSlaves  (SEQJUMP_INDEX_C),
         axiWriteMaster => mAxilWriteMasters(SEQJUMP_INDEX_C),
         axiWriteSlave  => mAxilWriteSlaves (SEQJUMP_INDEX_C),
         status         => status,
         config         => mConfig          (SEQJUMP_INDEX_C),
         axiClk         => clk,
         axiRst         => rst);

   U_SeqStateReg : entity work.XpmSeqStateReg
      port map (
         axiReadMaster  => mAxilReadMasters (SEQSTATE_INDEX_C),
         axiReadSlave   => mAxilReadSlaves  (SEQSTATE_INDEX_C),
         axiWriteMaster => mAxilWriteMasters(SEQSTATE_INDEX_C),
         axiWriteSlave  => mAxilWriteSlaves (SEQSTATE_INDEX_C),
         status         => status,
         config         => mConfig          (SEQSTATE_INDEX_C),
         axiClk         => clk,
         axiRst         => rst);

   U_SeqMemReg : entity work.XpmSeqMemReg
      port map (
         axiReadMaster  => mAxilReadMasters (SEQMEM_INDEX_C),
         axiReadSlave   => mAxilReadSlaves  (SEQMEM_INDEX_C),
         axiWriteMaster => mAxilWriteMasters(SEQMEM_INDEX_C),
         axiWriteSlave  => mAxilWriteSlaves (SEQMEM_INDEX_C),
         status         => status,
         config         => mConfig          (SEQMEM_INDEX_C),
         axiClk         => clk,
         axiRst         => rst);

   -------------------------------
   -- Configuration Register
   -------------------------------  
   comb : process (mConfig) is
      variable v : XpmSeqConfigType;
   begin
      v               := mConfig(SEQMEM_INDEX_C  );
      v.seqJumpConfig := mConfig(SEQJUMP_INDEX_C ).seqJumpConfig;
      v.seqEnable     := mConfig(SEQSTATE_INDEX_C).seqEnable;
      v.seqRestart    := mConfig(SEQSTATE_INDEX_C).seqRestart;
      config          <= v;
   end process comb;

end xbar;
