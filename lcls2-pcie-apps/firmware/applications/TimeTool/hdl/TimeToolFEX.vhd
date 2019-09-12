-------------------------------------------------------------------------------
-- File       : TimeToolFEX.vhd
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
--
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

entity TimeToolFEX is
   generic (
      TPD_G             : time                := 1 ns;
      AXI_BASE_ADDR_G   : slv(31 downto 0)    := x"00C1_0000";
      DMA_AXIS_CONFIG_G : AxiStreamConfigType := ssiAxiStreamConfig(16, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 8, 2));
   port (
      -- Clock and Reset
      axilClk         : in  sl;
      axilRst         : in  sl;
      -- DMA Interfaces (axilClk domain). One for data into server, two for reloadable coefficients and config from server, one trig master?
      dataInMaster    : in  AxiStreamMasterType ;
      dataInSlave     : out AxiStreamSlaveType;
      eventMaster     : out AxiStreamMasterType;
      eventSlave      : in  AxiStreamSlaveType;
      -- AXI-Lite Interface (axilClk domain)
      axilReadMaster  : in  AxiLiteReadMasterType;
      axilReadSlave   : out AxiLiteReadSlaveType;
      axilWriteMaster : in  AxiLiteWriteMasterType;
      axilWriteSlave  : out AxiLiteWriteSlaveType);
end TimeToolFEX;

architecture mapping of TimeToolFEX is

   constant NUM_AXIS_MASTERS_G : positive := 13;
   constant NUM_AXIL_MASTERS_C : natural  := 8;
   

   subtype  AXIL_INDEX_RANGE_C is integer range NUM_AXIL_MASTERS_C-1 downto 0;
   subtype  AXIS_INDEX_RANGE_C is integer range NUM_AXIS_MASTERS_G-1 downto 0;


   constant EVENTBUILDER_L                  : natural  := 0;
   constant EVCFILTER_L                     : natural  := 1;
   constant FIR_COEF_L                      : natural  := 2;
   constant FRAMEIIR_L                      : natural  := 3;
   constant NULLFILTER_L                    : natural  := 4;
   constant PEAKFINDER_L                    : natural  := 5;
   constant PRESCALER_L                     : natural  := 6;
   constant SUBTRACTOR_L                    : natural  := 7;


   constant REPEATER1_2_EVCFILTER           : natural  := 0;
   constant REPEATER1_2_SUBTRACTOR          : natural  := 1;
   constant EVCFILTER_2_REPEATER2           : natural  := 2;
   constant REPEATER2_2_NULLFILTER          : natural  := 3;
   constant REPEATER2_2_PRESCALER           : natural  := 4;
   constant PRESCALER_2_EVENTBUILDER        : natural  := 5;
   constant NULLFILTER_2_FRAMEIIR           : natural  := 6;
   constant FRAMEIIR_2_SUBTRACTOR           : natural  := 7;
   constant SUBTRACTOR_2_FIR                : natural  := 8;
   constant FIR_2_PEAKFINDER                : natural  := 9;
   constant PEAKFINDER_2_EVENTBUILDER       : natural  := 10;

   constant AXIL_TO_FIR_COEF                : natural  := 11;
   constant AXIL_TO_FIR_CONFIG              : natural  := 12;

   

   

   constant AXIL_CONFIG_C : AxiLiteCrossbarMasterConfigArray(AXIL_INDEX_RANGE_C) := genAxiLiteConfig(NUM_AXIL_MASTERS_C, AXI_BASE_ADDR_G, 16, 12);

   signal axilWriteMasters            : AxiLiteWriteMasterArray(AXIL_INDEX_RANGE_C);
   signal axilWriteSlaves             : AxiLiteWriteSlaveArray(AXIL_INDEX_RANGE_C);
   signal axilReadMasters             : AxiLiteReadMasterArray(AXIL_INDEX_RANGE_C);
   signal axilReadSlaves              : AxiLiteReadSlaveArray(AXIL_INDEX_RANGE_C);

   signal axisMasters                 : AxiStreamMasterArray(AXIS_INDEX_RANGE_C);
   signal axisSlaves                  : AxiStreamSlaveArray(AXIS_INDEX_RANGE_C);





begin

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


   ----------------------
   -- AXI Stream Repeater
   ----------------------
   U_AxiStreamRepeater_1 : entity work.AxiStreamRepeater
      generic map (
         TPD_G         => TPD_G,
         NUM_MASTERS_G => 2)
      port map (
         -- Clock and reset
         axisClk      => axilClk,
         axisRst      => axilRst,
         -- Slave
         sAxisMaster  => dataInMaster,
         sAxisSlave   => dataInSlave,
         -- Masters
         mAxisMasters(0)   => axisMasters(REPEATER1_2_EVCFILTER),
         mAxisMasters(1)   => axisMasters(REPEATER1_2_SUBTRACTOR),
         mAxisSlaves(0)    => axisSlaves(REPEATER1_2_EVCFILTER),
         mAxisSlaves(1)    => axisSlaves(REPEATER1_2_SUBTRACTOR));


   -------------------
   -- Prescaler Module. Place holder for event code based filter.
   -------------------
   EVCFILTER : entity work.TimeToolPrescaler
      generic map (
         TPD_G             => TPD_G,
         DMA_AXIS_CONFIG_G => DMA_AXIS_CONFIG_G)
      port map (
         -- System Clock and Reset
         sysClk          => axilClk,
         sysRst          => axilRst,
         -- DMA Interface (sysClk domain)
         dataInMaster    => axisMasters(REPEATER1_2_EVCFILTER),
         dataInSlave     => axisSlaves(REPEATER1_2_EVCFILTER),
         dataOutMaster   => axisMasters(EVCFILTER_2_REPEATER2),
         dataOutSlave    => axisSlaves(EVCFILTER_2_REPEATER2),
         -- AXI-Lite Interface (sysClk domain)
         axilReadMaster  => axilReadMasters(EVCFILTER_L),
         axilReadSlave   => axilReadSlaves(EVCFILTER_L),
         axilWriteMaster => axilWriteMasters(EVCFILTER_L),
         axilWriteSlave  => axilWriteSlaves(EVCFILTER_L));

   ----------------------
   -- AXI Stream Repeater
   ----------------------
   U_AxiStreamRepeater_2 : entity work.AxiStreamRepeater
      generic map (
         TPD_G         => TPD_G,
         NUM_MASTERS_G => 2)
      port map (
         -- Clock and reset
         axisClk      => axilClk,
         axisRst      => axilRst,
         -- Slave
         sAxisMaster  => axisMasters(EVCFILTER_2_REPEATER2),
         sAxisSlave   => axisSlaves(EVCFILTER_2_REPEATER2),
         -- Masters
         mAxisMasters(0) => axisMasters(REPEATER2_2_PRESCALER),
         mAxisMasters(1) => axisMasters(REPEATER2_2_NULLFILTER),
         mAxisSlaves(0)  => axisSlaves(REPEATER2_2_PRESCALER),
         mAxisSlaves(1)  => axisSlaves(REPEATER2_2_NULLFILTER));


   -------------------
   -- Prescaler Module
   -------------------
   PRESCALER : entity work.TimeToolPrescaler
      generic map (
         TPD_G             => TPD_G,
         DMA_AXIS_CONFIG_G => DMA_AXIS_CONFIG_G)
      port map (
         -- System Clock and Reset
         sysClk          => axilClk,
         sysRst          => axilRst,
         -- DMA Interface (sysClk domain)
         dataInMaster    => axisMasters(REPEATER2_2_PRESCALER),
         dataInSlave     => axisSlaves(REPEATER2_2_PRESCALER),
         dataOutMaster   => axisMasters(PRESCALER_2_EVENTBUILDER),
         dataOutSlave    => axisSlaves(PRESCALER_2_EVENTBUILDER),
         -- AXI-Lite Interface (sysClk domain)
         axilReadMaster  => axilReadMasters(PRESCALER_L),
         axilReadSlave   => axilReadSlaves(PRESCALER_L),
         axilWriteMaster => axilWriteMasters(PRESCALER_L),
         axilWriteSlave  => axilWriteSlaves(PRESCALER_L));

   -------------------
   -- background subtraction
   -------------------

   U_NullPacketFilter : entity work.NullPacketFilter
      generic map (
         TPD_G             => TPD_G,
         DMA_AXIS_CONFIG_G => DMA_AXIS_CONFIG_G)
      port map (
         -- System Clock and Reset
         sysClk          => axilClk,
         sysRst          => axilRst,
         -- DMA Interface (sysClk domain)
         dataInMaster    => axisMasters(REPEATER2_2_NULLFILTER),
         dataInSlave     => axisSlaves(REPEATER2_2_NULLFILTER),
         dataOutMaster   => axisMasters(NULLFILTER_2_FRAMEIIR),
         dataOutSlave    => axisSlaves(NULLFILTER_2_FRAMEIIR),
         -- AXI-Lite Interface (sysClk domain)
         axilReadMaster  => axilReadMasters(NULLFILTER_L),
         axilReadSlave   => axilReadSlaves(NULLFILTER_L),
         axilWriteMaster => axilWriteMasters(NULLFILTER_L),
         axilWriteSlave  => axilWriteSlaves(NULLFILTER_L));


   U_FrameIIR : entity work.FrameIIR
      generic map (
         TPD_G             => TPD_G,
         DMA_AXIS_CONFIG_G => DMA_AXIS_CONFIG_G)
      port map (
         -- System Clock and Reset
         sysClk          => axilClk,
         sysRst          => axilRst,
         -- DMA Interface (sysClk domain)
         dataInMaster    => axisMasters(NULLFILTER_2_FRAMEIIR),
         dataInSlave     => axisSlaves(NULLFILTER_2_FRAMEIIR),
         dataOutMaster   => axisMasters(FRAMEIIR_2_SUBTRACTOR),
         dataOutSlave    => axisSlaves(FRAMEIIR_2_SUBTRACTOR),
         -- AXI-Lite Interface (sysClk domain)
         axilReadMaster  => axilReadMasters(FRAMEIIR_L),
         axilReadSlave   => axilReadSlaves(FRAMEIIR_L),
         axilWriteMaster => axilWriteMasters(FRAMEIIR_L),
         axilWriteSlave  => axilWriteSlaves(FRAMEIIR_L));

   U_FrameSubtractor : entity work.FrameSubtractor
      generic map (
         TPD_G             => TPD_G,
         DMA_AXIS_CONFIG_G => DMA_AXIS_CONFIG_G)
      port map (
         -- System Clock and Reset
         sysClk           => axilClk,
         sysRst           => axilRst,
         -- DMA Interface (sysClk domain)
         dataInMaster     => axisMasters(REPEATER1_2_SUBTRACTOR),
         dataInSlave      => axisSlaves(REPEATER1_2_SUBTRACTOR),
         dataOutMaster    => axisMasters(SUBTRACTOR_2_FIR),
         dataOutSlave     => axisSlaves(SUBTRACTOR_2_FIR),
         -- Pedestal DMA Interfaces  (sysClk domain)
         pedestalInMaster =>  axisMasters(FRAMEIIR_2_SUBTRACTOR),
         pedestalInSlave  =>  axisSlaves(FRAMEIIR_2_SUBTRACTOR),
         -- AXI-Lite Interface (sysClk domain)
         axilReadMaster  => axilReadMasters(SUBTRACTOR_L),
         axilReadSlave   => axilReadSlaves(SUBTRACTOR_L),
         axilWriteMaster => axilWriteMasters(SUBTRACTOR_L),
         axilWriteSlave  => axilWriteSlaves(SUBTRACTOR_L));


   --------------------
   -- Axi lite to fir coefficient module
   -------------------- 


   --u_axilToFirCoef:entity work.AXILtoFIRcoef
   --generic map(
   --   TPD_G             => TPD_G,
   --   DMA_AXIS_CONFIG_G => DMA_AXIS_CONFIG_G,
   --   DEBUG_G           => true )
   --port map(
          -- System Interface
   --       sysClk           =>   axilClk,
   --       sysRst           =>   axilRst,
          -- DMA Interfaces  (sysClk domain)
   --       dataOutMaster     => axisMasters(AXIL_TO_FIR_COEF),
   --       dataOutSlave      => axisSlaves(AXIL_TO_FIR_COEF),
   --       configOutMaster   => axisMasters(AXIL_TO_FIR_CONFIG),
   --       configOutSlave    => axisSlaves(AXIL_TO_FIR_CONFIG),

          -- AXI-Lite Interface
   --      axilReadMaster  => axilReadMasters(FIR_COEF_L),
   --      axilReadSlave   => axilReadSlaves(FIR_COEF_L),
   --      axilWriteMaster => axilWriteMasters(FIR_COEF_L),
   --      axilWriteSlave  => axilWriteSlaves(FIR_COEF_L));

   --------------------
   -- Surf wrapped FIR filter
   -------------------- 


   --edge_to_peak: entity work.FrameFIR
   --generic map(
      --TPD_G             => TPD_G,
      --DMA_AXIS_CONFIG_G => DMA_AXIS_CONFIG_G,
      --DEBUG_G           => true )
   --port map(
      -- System Interface
      --sysClk          => axilClk,
      --sysRst          => axilRst,
      -- DMA Interfaces  (sysClk domain)
      --dataInMaster    => axisMasters(SUBTRACTOR_2_FIR),
      --dataInSlave     => axisSlaves(SUBTRACTOR_2_FIR),
      --dataOutMaster   => axisMasters(FIR_2_PEAKFINDER),
      --dataOutSlave    => axisSlaves(FIR_2_PEAKFINDER),
      -- coefficient reload  (sysClk domain)
      --reloadInMaster  => axisMasters(AXIL_TO_FIR_COEF),
      --reloadInSlave   => axisSlaves(AXIL_TO_FIR_COEF),
      --configInMaster  => axisMasters(AXIL_TO_FIR_CONFIG),
      --configInSlave   => axisSlaves(AXIL_TO_FIR_CONFIG));


     --U_FileInput : entity work.AxiStreamToFile
     --    generic map (
     --       TPD_G              => TPD_G,
     --       BYTE_SIZE_C        => 2+1,
     --      DMA_AXIS_CONFIG_G  => DMA_AXIS_CONFIG_G,
     --       CLK_PERIOD_G       => 10 ns)
     --    port map (
     --       sysClk         => axilClk,
     --       sysRst         => axilRst,
     --       dataInMaster   => axisMasters(FIR_2_PEAKFINDER));
     --       --dataInSlave    => appOutSlave

   --------------------
   -- Peak finder
   -------------------- 

     U_FramePeakFinder : entity work.FramePeakFinder
      generic map (
         TPD_G             => TPD_G,
         DMA_AXIS_CONFIG_G => DMA_AXIS_CONFIG_G)
      port map (
         -- System Clock and Reset
         sysClk          => axilClk,
         sysRst          => axilRst,
         -- DMA Interface (sysClk domain)
         dataInMaster    => axisMasters(SUBTRACTOR_2_FIR),
         dataInSlave     => axisSlaves(SUBTRACTOR_2_FIR),
         dataOutMaster   => axisMasters(PEAKFINDER_2_EVENTBUILDER),
         dataOutSlave    => axisSlaves(PEAKFINDER_2_EVENTBUILDER),
         -- AXI-Lite Interface (sysClk domain)
         axilReadMaster  => axilReadMasters(PEAKFINDER_L),
         axilReadSlave   => axilReadSlaves(PEAKFINDER_L),
         axilWriteMaster => axilWriteMasters(PEAKFINDER_L),
         axilWriteSlave  => axilWriteSlaves(PEAKFINDER_L)
         );



   ----------------------
   -- EventBuilder Module
   ----------------------
   U_EventBuilder : entity work.AxiStreamBatcherEventBuilder
      generic map (
         TPD_G         => TPD_G,
         NUM_SLAVES_G  => 2,
         AXIS_CONFIG_G => DMA_AXIS_CONFIG_G)
      port map (
         -- Clock and Reset
         axisClk             => axilClk,
         axisRst             => axilRst,
         -- AXI-Lite Interface (axisClk domain)
         axilReadMaster      => axilReadMasters(EVENTBUILDER_L),
         axilReadSlave       => axilReadSlaves(EVENTBUILDER_L),
         axilWriteMaster     => axilWriteMasters(EVENTBUILDER_L),
         axilWriteSlave      => axilWriteSlaves(EVENTBUILDER_L),
         -- Inbound Master AXIS Interfaces
         sAxisMasters(0)     => axisMasters(PEAKFINDER_2_EVENTBUILDER),
         sAxisMasters(1)     => axisMasters(PRESCALER_2_EVENTBUILDER),
         -- Inbound Slave AXIS Interfaces
         sAxisSlaves(0)      => axisSlaves(PEAKFINDER_2_EVENTBUILDER),
         sAxisSlaves(1)      => axisSlaves(PRESCALER_2_EVENTBUILDER),
         -- Outbound AXIS
         mAxisMaster         => eventMaster,
         mAxisSlave          => eventSlave);

end mapping;
