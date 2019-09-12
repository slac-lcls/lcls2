-------------------------------------------------------------------------------
-- File       : TimeToolKcu1500.vhd
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2017-10-24
-- Last update: 2018-11-08
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- This file is part of 'axi-pcie-dev'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'axi-pcie-dev', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

use work.StdRtlPkg.all;
use work.AxiPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.AxiPciePkg.all;
use work.TimingPkg.all;
use work.Pgp2bPkg.all;
use work.SsiPkg.all;
use work.TestingPkg.all;

use STD.textio.all;
use ieee.std_logic_textio.all;

entity TBPrescaledIIRSubtraction is end TBPrescaledIIRSubtraction;

architecture testbed of TBPrescaledIIRSubtraction is

   constant TEST_OUTPUT_FILE_NAME : string := TEST_FILE_PATH & "/output_results.dat";

   constant AXI_BASE_ADDR_G   : slv(31 downto 0) := x"00C0_0000";

   constant TPD_G             : time             := 1 ns;

   constant DMA_SIZE_C        : positive         := 1;

   constant NUM_MASTERS_G     : positive         := 3;

   ----------------------------
   ----------------------------
   ----------------------------
   constant NUM_AXIL_MASTERS_C  : natural := 4;

   constant PRESCALE_INDEX_C           : natural := 0;
   constant NULL_FILTER_INDEX_C        : natural := 1;
   constant FRAME_IIR_INDEX_C          : natural := 2;
   constant FRAME_SUBTRACTOR_INDEX_C   : natural := 3;


   constant NUM_REPEATER_OUTS          : natural := 2;


   subtype AXIL_INDEX_RANGE_C is integer range NUM_AXIL_MASTERS_C-1 downto 0;

   constant AXIL_CONFIG_C  : AxiLiteCrossbarMasterConfigArray(AXIL_INDEX_RANGE_C) := genAxiLiteConfig(NUM_AXIL_MASTERS_C, AXI_BASE_ADDR_G, 20, 16);

 
   ----------------------------
   ----------------------------
   ----------------------------


   constant DMA_AXIS_CONFIG_G : AxiStreamConfigType := ssiAxiStreamConfig(16, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 8, 2);


   constant CLK_PERIOD_G : time := 10 ns;

   constant SRC_CONFIG_C : AxiStreamConfigType := (
      TSTRB_EN_C    => false,
      TDATA_BYTES_C => 4, -- 128 bits
      TDEST_BITS_C  => 0,
      TID_BITS_C    => 0,
      TKEEP_MODE_C  => TKEEP_COMP_C,
      TUSER_BITS_C  => 2,
      TUSER_MODE_C  => TUSER_FIRST_LAST_C);

   signal userClk156   : sl;
   signal dmaClk       : sl;
   signal dmaRst       : sl;

   signal appInMaster  : AxiStreamMasterType;
   signal appInSlave   : AxiStreamSlaveType;
   signal appOutMaster : AxiStreamMasterType;
   signal appOutSlave  : AxiStreamSlaveType;

   signal PrescalerToNullFilterMaster  : AxiStreamMasterType;
   signal PrescalerToNullFilterSlave   : AxiStreamSlaveType;

   signal NullFilterToFrameIIRMaster  : AxiStreamMasterType;
   signal NullFilterToFrameIIRSlave   : AxiStreamSlaveType;

   signal FrameIIRToSubtractorMaster  : AxiStreamMasterType;
   signal FrameIIRToSubtractorSlave  : AxiStreamSlaveType;


   signal axilWriteMaster : AxiLiteWriteMasterType := AXI_LITE_WRITE_MASTER_INIT_C;
   signal axilWriteSlave  : AxiLiteWriteSlaveType  := AXI_LITE_WRITE_SLAVE_INIT_C;
   signal axilReadMaster  : AxiLiteReadMasterType  := AXI_LITE_READ_MASTER_INIT_C;
   signal axilReadSlave   : AxiLiteReadSlaveType   := AXI_LITE_READ_SLAVE_INIT_C;

   signal axilWriteMasters : AxiLiteWriteMasterArray(AXIL_INDEX_RANGE_C);
   signal axilWriteSlaves  : AxiLiteWriteSlaveArray(AXIL_INDEX_RANGE_C);
   signal axilReadMasters  : AxiLiteReadMasterArray(AXIL_INDEX_RANGE_C);
   signal axilReadSlaves   : AxiLiteReadSlaveArray(AXIL_INDEX_RANGE_C);

   subtype REPEATER_INDEX_RANGE_C is integer range NUM_REPEATER_OUTS-1 downto 0;

   signal dataInMasters : AxiStreamMasterArray(REPEATER_INDEX_RANGE_C);
   signal dataInSlaves  : AxiStreamSlaveArray(REPEATER_INDEX_RANGE_C);

   signal dataIbMasters : AxiStreamMasterArray(REPEATER_INDEX_RANGE_C);
   signal dataIbSlaves  : AxiStreamSlaveArray(REPEATER_INDEX_RANGE_C);

   signal axiClk   : sl;
   signal axiRst   : sl;

   signal axilClk   : sl;
   signal axilRst   : sl;

   file file_RESULTS : text;

begin

   appOutSlave.tReady <= '1';
   appInSlave.tReady  <= '1';
   axilClk            <= axiClk;
   axilRst            <= axiRst;
   
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

   --------------------
   -- Clocks and Resets
   --------------------
   U_axilClk_2 : entity work.ClkRst
      generic map (
         CLK_PERIOD_G      => CLK_PERIOD_G,
         RST_START_DELAY_G => 0 ns,
         RST_HOLD_TIME_G   => 1000 ns)
      port map (
         clkP => dmaClk,
         rst  => dmaRst);


   --------------------
   -- Clocks and Resets
   --------------------
   U_axilClk : entity work.ClkRst
      generic map (
         CLK_PERIOD_G      => CLK_PERIOD_G,
         RST_START_DELAY_G => 0 ns,
         RST_HOLD_TIME_G   => 1000 ns)
      port map (
         clkP => axiClk,
         rst  => axiRst);

   --------------------
   -- Test data
   --------------------  

      U_CamOutput : entity work.FileToAxiStreamSim
         generic map (
            TPD_G         => TPD_G,
            BYTE_SIZE_C   => 2+1,
            AXIS_CONFIG_G => SRC_CONFIG_C)
         port map (
            axiClk      => axiClk,
            axiRst      => axiRst,
            mAxisMaster => appInMaster,
            mAxisSlave  => appInSlave);

   ----------------------
   -- AXI Stream Repeater
   ----------------------
   U_AxiStreamRepeater : entity work.AxiStreamRepeater
      generic map (
         TPD_G         => TPD_G,
         NUM_MASTERS_G => 2)
      port map (
         -- Clock and reset
         axisClk      => axilClk,
         axisRst      => axilRst,
         -- Slave
         sAxisMaster  => appInMaster,
         --sAxisSlave   => appInSlave, --this pin can only be driven once in simulation
         -- Masters
         mAxisMasters => dataInMasters,
         mAxisSlaves  => dataInSlaves);

   ----------------------------------------         
   -- FIFO between Repeater and DSP Modules
   ----------------------------------------    
   GEN_IB :
   for i in REPEATER_INDEX_RANGE_C generate
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
            SLAVE_AXI_CONFIG_G  => DMA_AXIS_CONFIG_G,
            MASTER_AXI_CONFIG_G => DMA_AXIS_CONFIG_G)
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

   --------------------
   -- Modules to be tested
   --------------------  


   U_TimeToolPrescaler : entity work.TimeToolPrescaler
      generic map (
         TPD_G             => TPD_G,
         DMA_AXIS_CONFIG_G => DMA_AXIS_CONFIG_G)
      port map (
         -- System Clock and Reset
         sysClk          => dmaClk,
         sysRst          => dmaRst,
         -- DMA Interface (sysClk domain)
         dataInMaster    => dataIbMasters(0),
         dataInSlave     => dataIbSlaves(0),
         dataOutMaster   => PrescalerToNullFilterMaster,
         dataOutSlave    => PrescalerToNullFilterSlave,
         -- AXI-Lite Interface (sysClk domain)
         axilReadMaster  => axilReadMasters(PRESCALE_INDEX_C),
         axilReadSlave   => axilReadSlaves(PRESCALE_INDEX_C),
         axilWriteMaster => axilWriteMasters(PRESCALE_INDEX_C),
         axilWriteSlave  => axilWriteSlaves(PRESCALE_INDEX_C));

   U_NullPacketFilter : entity work.NullPacketFilter
      generic map (
         TPD_G             => TPD_G,
         DMA_AXIS_CONFIG_G => DMA_AXIS_CONFIG_G)
      port map (
         -- System Clock and Reset
         sysClk          => dmaClk,
         sysRst          => dmaRst,
         -- DMA Interface (sysClk domain)
         dataInMaster    => PrescalerToNullFilterMaster,
         dataInSlave     => PrescalerToNullFilterSlave,
         dataOutMaster   => NullFilterToFrameIIRMaster,
         dataOutSlave    => NullFilterToFrameIIRSlave,
         -- AXI-Lite Interface (sysClk domain)
         axilReadMaster  => axilReadMasters(NULL_FILTER_INDEX_C),
         axilReadSlave   => axilReadSlaves(NULL_FILTER_INDEX_C),
         axilWriteMaster => axilWriteMasters(NULL_FILTER_INDEX_C),
         axilWriteSlave  => axilWriteSlaves(NULL_FILTER_INDEX_C));


   U_FrameIIR : entity work.FrameIIR
      generic map (
         TPD_G             => TPD_G,
         DMA_AXIS_CONFIG_G => DMA_AXIS_CONFIG_G)
      port map (
         -- System Clock and Reset
         sysClk          => dmaClk,
         sysRst          => dmaRst,
         -- DMA Interface (sysClk domain)
         dataInMaster    => NullFilterToFrameIIRMaster,
         dataInSlave     => NullFilterToFrameIIRSlave,
         dataOutMaster   => FrameIIRToSubtractorMaster,
         dataOutSlave    => FrameIIRToSubtractorSlave,
         -- AXI-Lite Interface (sysClk domain)
         axilReadMaster  => axilReadMasters(FRAME_IIR_INDEX_C),
         axilReadSlave   => axilReadSlaves(FRAME_IIR_INDEX_C),
         axilWriteMaster => axilWriteMasters(FRAME_IIR_INDEX_C),
         axilWriteSlave  => axilWriteSlaves(FRAME_IIR_INDEX_C));

   U_FrameSubtractor : entity work.FrameSubtractor
      generic map (
         TPD_G             => TPD_G,
         DMA_AXIS_CONFIG_G => DMA_AXIS_CONFIG_G)
      port map (
         -- System Clock and Reset
         sysClk           => dmaClk,
         sysRst           => dmaRst,
         -- DMA Interface (sysClk domain)
         dataInMaster     => dataIbMasters(1),
         dataInSlave      => dataIbSlaves(1),
         dataOutMaster    => appOutMaster,
         dataOutSlave     => appOutSlave,
         -- Pedestal DMA Interfaces  (sysClk domain)
         pedestalInMaster =>  FrameIIRToSubtractorMaster,
         pedestalInSlave  =>  FrameIIRToSubtractorSlave,
         -- AXI-Lite Interface (sysClk domain)
         axilReadMaster  => axilReadMasters(FRAME_SUBTRACTOR_INDEX_C),
         axilReadSlave   => axilReadSlaves(FRAME_SUBTRACTOR_INDEX_C),
         axilWriteMaster => axilWriteMasters(FRAME_SUBTRACTOR_INDEX_C),
         axilWriteSlave  => axilWriteSlaves(FRAME_SUBTRACTOR_INDEX_C));

  ---------------------------------
   -- AXI-Lite Register Transactions
   ---------------------------------
   test : process is
      variable debugData : slv(31 downto 0) := (others => '0');
   begin
      debugData := x"1111_1111";
      ------------------------------------------
      -- Wait for the AXI-Lite reset to complete
      ------------------------------------------
      wait until axiRst = '1';
      wait until axiRst = '0';

      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C0_0004", x"5", true);  --prescaler
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C2_0004", x"3", true);  --iir time constant
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C3_0004", x"0", true);  --iir time constant

   end process test;

   ---------------------------------
   -- save_file
   ---------------------------------
   save_to_file : process is
      variable to_file              : AxiStreamMasterType := AXI_STREAM_MASTER_INIT_C;
      variable v_OLINE              : line; 
      constant c_WIDTH              : natural := 128;
      constant test_data_to_file    : slv(c_WIDTH -1 downto 0) := (others => '0');

   begin

      to_file := appOutMaster;

      file_open(file_RESULTS, TEST_OUTPUT_FILE_NAME, write_mode);

      while true loop

            --write(v_OLINE, appInMaster.tData(c_WIDTH -1 downto 0), right, c_WIDTH);
            write(v_OLINE, appOutMaster.tData(c_WIDTH-1 downto 0), right, c_WIDTH);
            writeline(file_RESULTS, v_OLINE);

            wait for CLK_PERIOD_G;

      end loop;
      
      file_close(file_RESULTS);

   end process save_to_file;

end testbed;
