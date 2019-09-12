-------------------------------------------------------------------------------
-- File       : TBAxiStreamFIR.vhd
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

entity TBAxiStreamFIR is end TBAxiStreamFIR;

architecture testbed of TBAxiStreamFIR is

   constant TEST_OUTPUT_FILE_NAME : string := TEST_FILE_PATH & "/output_results.dat";

   constant AXI_BASE_ADDR_G   : slv(31 downto 0) := x"00C0_0000";

   constant TPD_G             : time             := 1 ns;

   constant DMA_SIZE_C        : positive         := 1;
 
   ----------------------------
   ----------------------------
   ----------------------------


   constant DMA_AXIS_CONFIG_G           : AxiStreamConfigType := ssiAxiStreamConfig(16, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 8, 2);
   constant DMA_AXIS_DOWNSIZED_CONFIG_G : AxiStreamConfigType := ssiAxiStreamConfig(1, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 1, 2);

   constant CLK_PERIOD_G : time := 10 ns;

   signal appInMaster                 : AxiStreamMasterType  :=    AXI_STREAM_MASTER_INIT_C;
   signal appInMaster_pix_rev         : AxiStreamMasterType  :=    AXI_STREAM_MASTER_INIT_C;        
   signal appInSlave                  : AxiStreamSlaveType   :=    AXI_STREAM_SLAVE_INIT_C;

   signal appOutMaster                : AxiStreamMasterType  :=    AXI_STREAM_MASTER_INIT_C;
   signal appOutMaster_pix_rev         : AxiStreamMasterType :=    AXI_STREAM_MASTER_INIT_C;        
   signal appOutSlave                 : AxiStreamSlaveType   :=    AXI_STREAM_SLAVE_INIT_C;

   signal resizeFIFOToFIRMaster       : AxiStreamMasterType  :=    AXI_STREAM_MASTER_INIT_C;
   signal resizeFIFOToFIRSlave        : AxiStreamSlaveType   :=    AXI_STREAM_SLAVE_INIT_C;

   signal FIRToResizeFIFOMaster       : AxiStreamMasterType  :=    AXI_STREAM_MASTER_INIT_C;
   signal FIRToResizeFIFOSlave        : AxiStreamSlaveType   :=    AXI_STREAM_SLAVE_INIT_C;

   signal axiClk                      : sl;
   signal axiRst                      : sl;

   signal delayedAxiClk               : sl                  :=   '0';

   component fir_compiler_0
      port (aclk                    : std_logic;
            s_axis_data_tvalid      : std_logic;
            s_axis_data_tready      : out std_logic;
            s_axis_data_tdata       : std_logic_vector(7 downto 0);
            s_axis_data_tlast       : std_logic;
            m_axis_data_tvalid      : out std_logic;
            m_axis_data_tready      : std_logic;
            m_axis_data_tdata       : out std_logic_vector(7 downto 0);
            m_axis_data_tlast       : out std_logic);
   end component;

begin

   ------------------------------------------------------------------------------------------------------------
   ------------------------------------------------------------------------------------------------------------
   --Component for reversing byte order.  Needed for making FIFO down size compatible with FIR IP Core---------
   ------------------------------------------------------------------------------------------------------------
   ------------------------------------------------------------------------------------------------------------

   appInMaster_pix_rev.tValid <= appInMaster.tValid;
   appInMaster_pix_rev.tLast  <= appInMaster.tLast;
   
   APP_IN_PIXEL_SWAP: for i in 0 to DMA_AXIS_CONFIG_G.TDATA_BYTES_C-1 generate

        appInMaster_pix_rev.tData(i*8+7 downto i*8) <= appInMaster.tData(( (DMA_AXIS_CONFIG_G.TDATA_BYTES_C-1-i)*8+7) downto ((DMA_AXIS_CONFIG_G.TDATA_BYTES_C-1-i)*8));

   end generate APP_IN_PIXEL_SWAP;
   --
   --
   appOutMaster_pix_rev.tValid <= appOutMaster.tValid;
   appOutMaster_pix_rev.tLast  <= appOutMaster.tLast;
   
   APP_OUT_PIXEL_SWAP: for i in 0 to DMA_AXIS_CONFIG_G.TDATA_BYTES_C-1 generate

        appOutMaster_pix_rev.tData(i*8+7 downto i*8) <= appOutMaster.tData(( (DMA_AXIS_CONFIG_G.TDATA_BYTES_C-1-i)*8+7) downto ((DMA_AXIS_CONFIG_G.TDATA_BYTES_C-1-i)*8));

   end generate APP_OUT_PIXEL_SWAP;


   ------------------------------------------------
   ------------------------------------------------
   ------------------------------------------------
   ------------------------------------------------
   ------------------------------------------------

   delayedAxiClk <= axiClk after CLK_PERIOD_G/8;

   --------------------
   -- Clocks and Resets
   --------------------
   U_axilClk : entity work.ClkRst
      generic map (
         CLK_PERIOD_G      => CLK_PERIOD_G,
         RST_START_DELAY_G => 1 ns,
         RST_HOLD_TIME_G   => 50 ns)
      port map (
         clkP => axiClk,
         rst  => axiRst);

   --------------------
   -- Test data
   --------------------  

      U_CamOutput : entity work.FileToAxiStream
         generic map (
            TPD_G              => TPD_G,
            BYTE_SIZE_C        => 2+1,
            DMA_AXIS_CONFIG_G  => DMA_AXIS_CONFIG_G,
            CLK_PERIOD_G       => 10 ns)
         port map (
            sysClk         => axiClk,
            sysRst         => axiRst,
            dataOutMaster  => appInMaster,
            dataOutSlave   => appInSlave);

      U_down_size_test : entity work.AxiStreamFifoV2
         generic map (
            -- General Configurations
            TPD_G               => TPD_G,
            SLAVE_READY_EN_G    => true,
            VALID_THOLD_G       => 1,
            -- FIFO configurations
            BRAM_EN_G           => true,
            GEN_SYNC_FIFO_G     => true,
            FIFO_ADDR_WIDTH_G   => 9,
            FIFO_PAUSE_THRESH_G => 500,
            -- AXI Stream Port Configurations
            SLAVE_AXI_CONFIG_G  => DMA_AXIS_CONFIG_G,
            MASTER_AXI_CONFIG_G => DMA_AXIS_DOWNSIZED_CONFIG_G)
         port map (
            -- Slave Port
            sAxisClk    => axiClk,
            sAxisRst    => axiRst,
            sAxisMaster => appInMaster_pix_rev, --appInMaster,
            sAxisSlave  => appInSlave,
            -- Master Port
            mAxisClk    => axiClk,
            mAxisRst    => axiRst,
            mAxisMaster => resizeFIFOToFIRMaster,
            mAxisSlave  => resizeFIFOToFIRSlave
            );

        dut : fir_compiler_0
          port map (
            aclk                            => delayedAxiClk,
            s_axis_data_tvalid              => resizeFIFOToFIRMaster.tValid,
            s_axis_data_tready              => resizeFIFOToFIRSlave.tReady,
            s_axis_data_tdata               => resizeFIFOToFIRMaster.tData(7 downto 0),
            s_axis_data_tlast               => resizeFIFOToFIRMaster.tLast,
            m_axis_data_tvalid              => FIRToResizeFIFOMaster.tValid,
            m_axis_data_tready              => FIRToResizeFIFOSlave.tReady,
            m_axis_data_tdata               => FIRToResizeFIFOMaster.tData(7 downto 0),
            m_axis_data_tlast               => FIRToResizeFIFOMaster.tLast
            );

      U_up_size_test : entity work.AxiStreamFifoV2
         generic map (
            -- General Configurations
            TPD_G               => TPD_G,
            SLAVE_READY_EN_G    => true,
            VALID_THOLD_G       => 1,
            -- FIFO configurations
            BRAM_EN_G           => true,
            GEN_SYNC_FIFO_G     => true,
            FIFO_ADDR_WIDTH_G   => 9,
            FIFO_PAUSE_THRESH_G => 500,
            -- AXI Stream Port Configurations
            SLAVE_AXI_CONFIG_G  => DMA_AXIS_DOWNSIZED_CONFIG_G,
            MASTER_AXI_CONFIG_G => DMA_AXIS_CONFIG_G)
         port map (
            -- Slave Port
            sAxisClk    => axiClk,
            sAxisRst    => axiRst,
            sAxisMaster => FIRToResizeFIFOMaster,
            sAxisSlave  => FIRToResizeFIFOSlave,
            -- Master Port
            mAxisClk    => axiClk,
            mAxisRst    => axiRst,
            mAxisMaster => appOutMaster,
            mAxisSlave  => appOutSlave);


      U_FileInput : entity work.AxiStreamToFile
         generic map (
            TPD_G              => TPD_G,
            BYTE_SIZE_C        => 2+1,
            DMA_AXIS_CONFIG_G  => DMA_AXIS_CONFIG_G,
            CLK_PERIOD_G       => 10 ns)
         port map (
            sysClk         => axiClk,
            sysRst         => axiRst,
            dataInMaster   => appOutMaster_pix_rev,
            dataInSlave    => appOutSlave);

end testbed;
