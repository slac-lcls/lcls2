-------------------------------------------------------------------------------
-- File       : TimeToolCore.vhd
-- Company    : SLAC National Accelerator Laboratory
-------------------------------------------------------------------------------
-- Description: TimeTool Core Module
-------------------------------------------------------------------------------
-- This file is part of 'axi-pcie-core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'axi-pcie-core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.SsiPkg.all;
use work.AppPkg.all;

entity TimeToolCore is
   generic (
      TPD_G           : time             := 1 ns;
      AXI_BASE_ADDR_G : slv(31 downto 0) := x"00C0_0000");
   port (
      -- Clock and Reset
      axilClk         : in  sl;
      axilRst         : in  sl;
      -- Trigger Event streams (axilClk domain)
      trigMaster      : in  AxiStreamMasterType;
      trigSlave       : out AxiStreamSlaveType;
      -- DMA Interfaces (axilClk domain)
      dataInMaster    : in  AxiStreamMasterType;
      dataInSlave     : out AxiStreamSlaveType;
      eventMaster     : out AxiStreamMasterType;
      eventSlave      : in  AxiStreamSlaveType;
      -- AXI-Lite Interface (axilClk domain)
      axilReadMaster  : in  AxiLiteReadMasterType;
      axilReadSlave   : out AxiLiteReadSlaveType;
      axilWriteMaster : in  AxiLiteWriteMasterType;
      axilWriteSlave  : out AxiLiteWriteSlaveType);
end TimeToolCore;

architecture mapping of TimeToolCore is

   constant NUM_AXIS_MASTERS_G      : positive := 2;

   constant NUM_AXIL_MASTERS_C : natural  := NUM_AXIS_MASTERS_G+2;

   constant EVENT_INDEX_C      : natural  := 0;
   constant FEX_INDEX_C        : natural  := 1;
   constant PRESCALE_INDEX_C   : natural  := 2;
   constant BYPASS_INDEX_C     : natural  := 3;

   subtype  AXIL_INDEX_RANGE_C is integer range NUM_AXIL_MASTERS_C-1 downto 0;

   constant AXIL_CONFIG_C : AxiLiteCrossbarMasterConfigArray(AXIL_INDEX_RANGE_C) := genAxiLiteConfig(NUM_AXIL_MASTERS_C, AXI_BASE_ADDR_G, 20, 16);

   signal axilWriteMasters            : AxiLiteWriteMasterArray(AXIL_INDEX_RANGE_C);
   signal axilWriteSlaves             : AxiLiteWriteSlaveArray(AXIL_INDEX_RANGE_C);
   signal axilReadMasters             : AxiLiteReadMasterArray(AXIL_INDEX_RANGE_C);
   signal axilReadSlaves              : AxiLiteReadSlaveArray(AXIL_INDEX_RANGE_C);

   subtype DSP_INDEX_RANGE_C is integer range NUM_AXIL_MASTERS_C-2 downto 1;

   signal dataInMasters               : AxiStreamMasterArray(DSP_INDEX_RANGE_C);
   signal dataInSlaves                : AxiStreamSlaveArray(DSP_INDEX_RANGE_C);

   signal dataIbMasters               : AxiStreamMasterArray(DSP_INDEX_RANGE_C);
   signal dataIbSlaves                : AxiStreamSlaveArray(DSP_INDEX_RANGE_C);

   signal dspObMasters                : AxiStreamMasterArray(DSP_INDEX_RANGE_C);
   signal dspObSlaves                 : AxiStreamSlaveArray(DSP_INDEX_RANGE_C);

   signal dspMasters                  : AxiStreamMasterArray(DSP_INDEX_RANGE_C);
   signal dspSlaves                   : AxiStreamSlaveArray(DSP_INDEX_RANGE_C);

   signal byPassToTimeToolMaster      : AxiStreamMasterType;
   signal byPassToTimeToolSlave       : AxiStreamSlaveType;

   signal timeToolToByPassMaster      : AxiStreamMasterType;
   signal timeToolToByPassSlave       : AxiStreamSlaveType;



begin

   -------------
   -- ByPass Module
   -------------
   U_TimetoolBypass : entity work.TimetoolBypass
      generic map (
         TPD_G                => TPD_G,
         DMA_AXIS_CONFIG_G    => DMA_AXIS_CONFIG_C)
      port map (
         -- System Clock and Reset
         sysClk               => axilClk,
         sysRst               => axilRst,
         -- DMA Interface (sysClk domain)
         dataInMaster         => dataInMaster,
         dataInSlave          => dataInSlave,
         dataOutMaster        => eventMaster,
         dataOutSlave         => eventSlave,

         fromTimeToolMaster   => timeToolToByPassMaster,
         fromTimeToolSlave    => timeToolToByPassSlave,

         toTimeToolMaster     => byPassToTimeToolMaster,
         toTimeToolSlave      => byPassToTimeToolSlave,


         -- AXI-Lite Interface (sysClk domain)
         axilReadMaster       => axilReadMasters(BYPASS_INDEX_C),
         axilReadSlave        => axilReadSlaves(BYPASS_INDEX_C),
         axilWriteMaster      => axilWriteMasters(BYPASS_INDEX_C),
         axilWriteSlave       => axilWriteSlaves(BYPASS_INDEX_C));


   ----------------------
   -- AXI Stream Repeater
   ----------------------
   U_AxiStreamRepeater : entity work.AxiStreamRepeater
      generic map (
         TPD_G         => TPD_G,
         NUM_MASTERS_G => NUM_AXIS_MASTERS_G)
      port map (
         -- Clock and reset
         axisClk      => axilClk,
         axisRst      => axilRst,
         -- Slave
         sAxisMaster  => byPassToTimeToolMaster,
         sAxisSlave   => byPassToTimeToolSlave,
         -- Masters
         mAxisMasters => dataInMasters,
         mAxisSlaves  => dataInSlaves);

   --------------------
   -- AXI-Lite Crossbar
   --------------------
   U_XBAR : entity work.AxiLiteCrossbar
      generic map (
         TPD_G              => TPD_G,
         NUM_SLAVE_SLOTS_G  => 1,
         NUM_MASTER_SLOTS_G => NUM_AXIL_MASTERS_C,
         MASTERS_CONFIG_G   => AXIL_CONFIG_C)
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


   ----------------------------------------         
   -- FIFO between Repeater and DSP Modules
   ----------------------------------------    
   GEN_IB :
   for i in DSP_INDEX_RANGE_C generate
      U_FIFO : entity work.AxiStreamFifoV2
         generic map (
            -- General Configurations
            TPD_G               => TPD_G,
            SLAVE_READY_EN_G    => true,
            VALID_THOLD_G       => 1,
            -- FIFO configurations
            BRAM_EN_G           => true,
            GEN_SYNC_FIFO_G     => true,
            FIFO_ADDR_WIDTH_G   => 9,
            -- AXI Stream Port Configurations
            SLAVE_AXI_CONFIG_G  => DMA_AXIS_CONFIG_C,
            MASTER_AXI_CONFIG_G => DMA_AXIS_CONFIG_C)
         port map (
            -- Slave Port
            sAxisClk    => axilClk,
            sAxisRst    => axilRst,
            sAxisMaster => dataInMasters(i),
            sAxisSlave  => dataInSlaves(i),
            -- Master Port
            mAxisClk    => axilClk,
            mAxisRst    => axilRst,
            mAxisMaster => dataIbMasters(i),
            mAxisSlave  => dataIbSlaves(i));
   end generate GEN_IB;

   -------------
   -- FEX Module
   -------------
   U_TimeToolFEX : entity work.TimeToolFEX
      generic map (
         TPD_G               => TPD_G,
         AXI_BASE_ADDR_G     => AXIL_CONFIG_C(FEX_INDEX_C).baseAddr,
         DMA_AXIS_CONFIG_G   => DMA_AXIS_CONFIG_C)
      port map (
         -- System Clock and Reset
         axilClk         => axilClk,
         axilRst         => axilRst,
         -- DMA Interface (sysClk domain)
         dataInMaster       => dataIbMasters(FEX_INDEX_C),
         dataInSlave        => dataIbSlaves(FEX_INDEX_C),
         eventMaster        => dspObMasters(FEX_INDEX_C),
         eventSlave         => dspObSlaves(FEX_INDEX_C),
         -- AXI-Lite Interface (sysClk domain)
         axilReadMaster  => axilReadMasters(FEX_INDEX_C),
         axilReadSlave   => axilReadSlaves(FEX_INDEX_C),
         axilWriteMaster => axilWriteMasters(FEX_INDEX_C),
         axilWriteSlave  => axilWriteSlaves(FEX_INDEX_C));

   -------------------
   -- Prescaler Module
   -------------------
   U_TimeToolPrescaler : entity work.TimeToolPrescaler
      generic map (
         TPD_G             => TPD_G,
         DMA_AXIS_CONFIG_G => DMA_AXIS_CONFIG_C)
      port map (
         -- System Clock and Reset
         sysClk          => axilClk,
         sysRst          => axilRst,
         -- DMA Interface (sysClk domain)
         dataInMaster    => dataIbMasters(PRESCALE_INDEX_C),
         dataInSlave     => dataIbSlaves(PRESCALE_INDEX_C),
         dataOutMaster   => dspObMasters(PRESCALE_INDEX_C),
         dataOutSlave    => dspObSlaves(PRESCALE_INDEX_C),
         -- AXI-Lite Interface (sysClk domain)
         axilReadMaster  => axilReadMasters(PRESCALE_INDEX_C),
         axilReadSlave   => axilReadSlaves(PRESCALE_INDEX_C),
         axilWriteMaster => axilWriteMasters(PRESCALE_INDEX_C),
         axilWriteSlave  => axilWriteSlaves(PRESCALE_INDEX_C));

   ---------------------------------------------    
   -- FIFO between DSP Modules and Event Builder
   ---------------------------------------------    
   GEN_OB :
   for i in DSP_INDEX_RANGE_C generate
      U_FIFO : entity work.AxiStreamFifoV2
         generic map (
            -- General Configurations
            TPD_G               => TPD_G,
            SLAVE_READY_EN_G    => true,
            VALID_THOLD_G       => 1,
            -- FIFO configurations
            BRAM_EN_G           => true,
            GEN_SYNC_FIFO_G     => true,
            FIFO_ADDR_WIDTH_G   => 9,
            -- AXI Stream Port Configurations
            SLAVE_AXI_CONFIG_G  => DMA_AXIS_CONFIG_C,
            MASTER_AXI_CONFIG_G => DMA_AXIS_CONFIG_C)
         port map (
            -- Slave Port
            sAxisClk    => axilClk,
            sAxisRst    => axilRst,
            sAxisMaster => dspObMasters(i),
            sAxisSlave  => dspObSlaves(i),
            -- Master Port
            mAxisClk    => axilClk,
            mAxisRst    => axilRst,
            mAxisMaster => dspMasters(i),
            mAxisSlave  => dspSlaves(i));
   end generate GEN_OB;

   ----------------------
   -- EventBuilder Module
   ----------------------
   U_EventBuilder : entity work.AxiStreamBatcherEventBuilder
      generic map (
         TPD_G         => TPD_G,
         NUM_SLAVES_G  => NUM_AXIS_MASTERS_G+1,
         AXIS_CONFIG_G => DMA_AXIS_CONFIG_C)
      port map (
         -- Clock and Reset
         axisClk                         => axilClk,
         axisRst                         => axilRst,
         -- AXI-Lite Interface (axisClk domain)
         axilReadMaster                  => axilReadMasters(EVENT_INDEX_C),
         axilReadSlave                   => axilReadSlaves(EVENT_INDEX_C),
         axilWriteMaster                 => axilWriteMasters(EVENT_INDEX_C),
         axilWriteSlave                  => axilWriteSlaves(EVENT_INDEX_C),
         -- Inbound Master AXIS Interfaces
         sAxisMasters(EVENT_INDEX_C)     => trigMaster,
         sAxisMasters(DSP_INDEX_RANGE_C) => dspMasters,
         -- Inbound Slave AXIS Interfaces
         sAxisSlaves(EVENT_INDEX_C)      => trigSlave,
         sAxisSlaves(DSP_INDEX_RANGE_C)  => dspSlaves,
         -- Outbound AXIS
         mAxisMaster                     => timeToolToByPassMaster,
         mAxisSlave                      => timeToolToByPassSlave);

end mapping;
