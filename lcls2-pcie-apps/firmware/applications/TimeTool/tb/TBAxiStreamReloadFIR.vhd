-------------------------------------------------------------------------------
-- File       : TBAxiStreamReloadFIR.vhd
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

entity TBAxiStreamReloadFIR is end TBAxiStreamReloadFIR;

architecture testbed of TBAxiStreamReloadFIR is
  
   constant FIR_COEF_FILE_NAME : string    := TEST_FILE_PATH & "/fir_coef.dat";
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
   constant T_HOLD       : time := 100 ps;
   
   file fir_coef_file    : text;

   signal appInMaster                 : AxiStreamMasterType  :=    AXI_STREAM_MASTER_INIT_C;        
   signal appInSlave                  : AxiStreamSlaveType   :=    AXI_STREAM_SLAVE_INIT_C;

   signal appOutMaster                : AxiStreamMasterType  :=    AXI_STREAM_MASTER_INIT_C;        
   signal appOutSlave                 : AxiStreamSlaveType   :=    AXI_STREAM_SLAVE_INIT_C;

   signal reloadInMaster              : AxiStreamMasterType  :=    AXI_STREAM_MASTER_INIT_C;        
   signal reloadInSlave               : AxiStreamSlaveType   :=    AXI_STREAM_SLAVE_INIT_C;

   signal configInMaster              : AxiStreamMasterType  :=    AXI_STREAM_MASTER_INIT_C;        
   signal configInSlave               : AxiStreamSlaveType   :=    AXI_STREAM_SLAVE_INIT_C;

   -- Event signals
   signal event_s_reload_tlast_missing    : std_logic  :=  '0';  -- reloadInMaster.tLast low at end of reload packet
   signal event_s_reload_tlast_unexpected : std_logic  :=  '0';  -- reloadInMaster.tLast high not at end of reload packet

   signal axiClk                      : sl;
   signal axiRst                      : sl;




begin


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

   --------------------
   -- Surf wrapped FIR filter
   -------------------- 


   edge_to_peak: entity work.FrameFIR
   generic map(
      TPD_G             => TPD_G,
      DMA_AXIS_CONFIG_G => DMA_AXIS_CONFIG_G,
      DEBUG_G           => true )
   port map(
      -- System Interface
      sysClk          => axiClk,
      sysRst          => axiRst,
      -- DMA Interfaces  (sysClk domain)
      dataInMaster    => appInMaster,
      dataInSlave     => appInSlave,
      dataOutMaster   => appOutMaster,
      dataOutSlave    => appOutSlave,
      -- coefficient reload  (sysClk domain)
      reloadInMaster  => reloadInMaster,
      reloadInSlave   => reloadInSlave,
      configInMaster  => configInMaster,
      configInSlave   => configInSlave);





      U_FileInput : entity work.AxiStreamToFile
         generic map (
            TPD_G              => TPD_G,
            BYTE_SIZE_C        => 2+1,
            DMA_AXIS_CONFIG_G  => DMA_AXIS_CONFIG_G,
            CLK_PERIOD_G       => 10 ns)
         port map (
            sysClk         => axiClk,
            sysRst         => axiRst,
            dataInMaster   => appOutMaster,
            dataInSlave    => appOutSlave);


   ------------------------------------------------
   ------------------------------------------------
   ------------------------------------------------
   ------------------------------------------------
   ------------------------------------------------

   reload_coeffs : process is
        variable v_ILINE      : line;
        variable my_coef      : slv(7 downto 0);
        begin

           file_open(fir_coef_file,FIR_COEF_FILE_NAME ,read_mode);


           wait for 1 us;


           for coef in 0 to 31 loop

              readline(fir_coef_file,v_ILINE);
              read(v_ILINE,my_coef);

              reloadInMaster.tValid <= '1';
              reloadInMaster.tData <= (others => '0');  -- clear unused bits of TDATA

              reloadInMaster.tData(7 downto 0) <= my_coef;


              if coef = 31 then
                reloadInMaster.tLast <= '1';  -- signal last transaction in reload packet
              else
                reloadInMaster.tLast <= '0';
              end if;

              loop
                wait until rising_edge(axiClk);
                exit when reloadInSlave.tReady = '1';
              end loop;
              wait for T_HOLD;
            end loop;
            reloadInMaster.tLast  <= '0';
            reloadInMaster.tValid <= '0';

            -- A packet on the config slave channel signals that the new coefficients should now be used.
            -- The config packet is required only for signalling: its data is irrelevant.
            configInMaster.tValid <= '1';
            configInMaster.tData  <= (others => '0');  -- don't care about TDATA - it is unused
            loop
              wait until rising_edge(axiClk);
              exit when configInSlave.tReady = '1';
            end loop;
            wait for T_HOLD;
            wait for 10 ms;

        end process reload_coeffs;
   


end testbed;
