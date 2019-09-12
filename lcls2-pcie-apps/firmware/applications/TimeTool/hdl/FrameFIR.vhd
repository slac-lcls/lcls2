------------------------------------------------------------------------------
-- File       : FrameFIR.vhd
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2017-12-04
-- Last update: 2017-12-04
-------------------------------------------------------------------------------
-- Description:
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
--use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_signed.all;
use ieee.numeric_std.ALL;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.AxiPkg.all;
use work.SsiPkg.all;
use work.AxiPciePkg.all;
use work.TimingPkg.all;
use work.Pgp2bPkg.all;

library unisim;
use unisim.vcomponents.all;

-------------------------------------------------------------------------------
-- This file wraps the xilinx FIR IP core.
-------------------------------------------------------------------------------

entity FrameFIR is
   generic (
      TPD_G             : time                := 1 ns;
      DMA_AXIS_CONFIG_G : AxiStreamConfigType := ssiAxiStreamConfig(16, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 8, 2);
      DEBUG_G           : boolean             := true );
   port (
      -- System Interface
      sysClk          : in    sl;
      sysRst          : in    sl;
      -- DMA Interfaces  (sysClk domain)
      dataInMaster    : in    AxiStreamMasterType;
      dataInSlave     : out   AxiStreamSlaveType;
      dataOutMaster   : out   AxiStreamMasterType;
      dataOutSlave    : in    AxiStreamSlaveType;
      -- coefficient reload  (sysClk domain)
      reloadInMaster    : in    AxiStreamMasterType;
      reloadInSlave     : out   AxiStreamSlaveType;
      configInMaster    : in    AxiStreamMasterType;
      configInSlave     : out   AxiStreamSlaveType);
end FrameFIR;

architecture mapping of FrameFIR is
 
   --constant DMA_AXIS_CONFIG_G           : AxiStreamConfigType := ssiAxiStreamConfig(16, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 8, 2);
   constant DMA_AXIS_DOWNSIZED_CONFIG_G : AxiStreamConfigType := ssiAxiStreamConfig(1, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 1, 2);

---------------------------------------
-------record intitial value-----------
---------------------------------------

   signal appInMaster                 : AxiStreamMasterType  :=    AXI_STREAM_MASTER_INIT_C;
   signal appInMaster_pix_rev         : AxiStreamMasterType  :=    AXI_STREAM_MASTER_INIT_C;        
   signal appInSlave                  : AxiStreamSlaveType   :=    AXI_STREAM_SLAVE_INIT_C;

   signal appOutMaster                : AxiStreamMasterType  :=    AXI_STREAM_MASTER_INIT_C;
   signal appOutMaster_pix_rev        : AxiStreamMasterType  :=    AXI_STREAM_MASTER_INIT_C;        
   signal appOutSlave                 : AxiStreamSlaveType   :=    AXI_STREAM_SLAVE_INIT_C;

   signal resizeFIFOToFIRMaster       : AxiStreamMasterType  :=    AXI_STREAM_MASTER_INIT_C;
   signal resizeFIFOToFIRSlave        : AxiStreamSlaveType   :=    AXI_STREAM_SLAVE_INIT_C;

   signal FIRToResizeFIFOMaster       : AxiStreamMasterType  :=    AXI_STREAM_MASTER_INIT_C;
   signal FIRToResizeFIFOSlave        : AxiStreamSlaveType   :=    AXI_STREAM_SLAVE_INIT_C;


   signal event_s_reload_tlast_missing     : sl              :=    '0';
   signal event_s_reload_tlast_unexpected  : sl              :=    '0';

   signal not_sysRst                       : sl              :=    '1';



 component fir_compiler_1
      port (aclk                    : std_logic;
            aresetn                 : std_logic;
            s_axis_data_tvalid      : std_logic;
            s_axis_data_tready      : out std_logic;
            s_axis_data_tdata       : std_logic_vector(7 downto 0);
            s_axis_data_tlast       : std_logic;
            s_axis_config_tvalid    : std_logic;
            s_axis_config_tready    : out std_logic;
            s_axis_config_tdata     : std_logic_vector(7 downto 0);
            s_axis_reload_tvalid    : std_logic;
            s_axis_reload_tready    : out std_logic;
            s_axis_reload_tdata     : std_logic_vector(7 downto 0);
            s_axis_reload_tlast     : std_logic;
            m_axis_data_tvalid      : out std_logic;
            m_axis_data_tready      : std_logic;
            m_axis_data_tdata       : out std_logic_vector(7 downto 0);
            m_axis_data_tlast       : out std_logic;
            event_s_reload_tlast_missing    : out std_logic;
            event_s_reload_tlast_unexpected : out std_logic);
   end component;


begin

   not_sysRst    <= not sysRst;

   appInMaster   <= dataInMaster;
   dataInSlave   <= appInSlave;

   dataOutMaster <= appOutMaster_pix_rev;
   appOutSlave   <= dataOutSlave;
   
   --------------------------------
   --byte order change for input---
   --------------------------------

   appInMaster_pix_rev.tValid <= appInMaster.tValid;
   appInMaster_pix_rev.tLast  <= appInMaster.tLast;
   
   APP_IN_PIXEL_SWAP: for i in 0 to DMA_AXIS_CONFIG_G.TDATA_BYTES_C-1 generate

        appInMaster_pix_rev.tData(i*8+7 downto i*8) <= appInMaster.tData(( (DMA_AXIS_CONFIG_G.TDATA_BYTES_C-1-i)*8+7) downto ((DMA_AXIS_CONFIG_G.TDATA_BYTES_C-1-i)*8));

   end generate APP_IN_PIXEL_SWAP;


   --------------------------------
   --byte order change for output--
   --------------------------------

   appOutMaster_pix_rev.tValid <= appOutMaster.tValid;
   appOutMaster_pix_rev.tLast  <= appOutMaster.tLast;
   
   APP_OUT_PIXEL_SWAP: for i in 0 to DMA_AXIS_CONFIG_G.TDATA_BYTES_C-1 generate

        appOutMaster_pix_rev.tData(i*8+7 downto i*8) <= appOutMaster.tData(( (DMA_AXIS_CONFIG_G.TDATA_BYTES_C-1-i)*8+7) downto ((DMA_AXIS_CONFIG_G.TDATA_BYTES_C-1-i)*8));

   end generate APP_OUT_PIXEL_SWAP;

   --------------------------------
   --------------------------------
   --------------------------------


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
            sAxisClk    => sysClk,
            sAxisRst    => sysRst,
            sAxisMaster => appInMaster_pix_rev, --appInMaster,
            sAxisSlave  => appInSlave,
            -- Master Port
            mAxisClk    => sysClk,
            mAxisRst    => sysRst,
            mAxisMaster => resizeFIFOToFIRMaster,
            mAxisSlave  => resizeFIFOToFIRSlave
            );


       dut : fir_compiler_1
          port map (
            aclk                            => sysClk,
            aresetn                         => not_sysRst,
            s_axis_data_tvalid              => resizeFIFOToFIRMaster.tValid,
            s_axis_data_tready              => resizeFIFOToFIRSlave.tReady,
            s_axis_data_tdata               => resizeFIFOToFIRMaster.tData(7 downto 0),
            s_axis_data_tlast               => resizeFIFOToFIRMaster.tLast,
            s_axis_config_tvalid            => configInMaster.tValid,
            s_axis_config_tready            => configInSlave.tReady,
            s_axis_config_tdata             => configInMaster.tData(7 downto 0),
            s_axis_reload_tvalid            => reloadInMaster.tValid,
            s_axis_reload_tready            => reloadInSlave.tReady,
            s_axis_reload_tdata             => reloadInMaster.tData(7 downto 0),
            s_axis_reload_tlast             => reloadInMaster.tLast,
            m_axis_data_tvalid              => FIRToResizeFIFOMaster.tValid,
            m_axis_data_tready              => FIRToResizeFIFOSlave.tReady,
            m_axis_data_tdata               => FIRToResizeFIFOMaster.tData(7 downto 0),
            m_axis_data_tlast               => FIRToResizeFIFOMaster.tLast,
            event_s_reload_tlast_missing    => event_s_reload_tlast_missing,
            event_s_reload_tlast_unexpected => event_s_reload_tlast_unexpected
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
            sAxisClk    => sysClk,
            sAxisRst    => sysRst,
            sAxisMaster => FIRToResizeFIFOMaster,
            sAxisSlave  => FIRToResizeFIFOSlave,
            -- Master Port
            mAxisClk    => sysClk,
            mAxisRst    => sysRst,
            mAxisMaster => appOutMaster,
            mAxisSlave  => appOutSlave);


end mapping;
