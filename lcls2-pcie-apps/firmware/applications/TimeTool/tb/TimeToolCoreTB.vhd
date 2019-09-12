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

entity TimeToolCoreTB is end TimeToolCoreTB;

architecture testbed of TimeToolCoreTB is

   constant T_HOLD                     : time               := 100 ps;
   constant FIR_COEF_FILE_NAME         : string             := TEST_FILE_PATH & "/fir_coef.dat";
   constant TEST_OUTPUT_FILE_NAME      : string             := TEST_FILE_PATH & "/output_results.dat";

   constant AXI_BASE_ADDR_G            : slv(31 downto 0)   := x"00C0_0000";

   constant TPD_G                      : time               := 1 ns;

   constant DMA_SIZE_C                 : positive           := 1;

   constant NUM_MASTERS_G              : positive           := 3;

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

   signal userClk156      : sl;
   signal dmaClk          : sl;
   signal dmaRst          : sl;

   signal appInMaster     : AxiStreamMasterType    :=  AXI_STREAM_MASTER_INIT_C;
   signal appInSlave      : AxiStreamSlaveType     :=  AXI_STREAM_SLAVE_INIT_C;
   signal appOutMaster    : AxiStreamMasterType    :=  AXI_STREAM_MASTER_INIT_C;
   signal appOutSlave     : AxiStreamSlaveType     :=  AXI_STREAM_SLAVE_INIT_C;

   signal axilWriteMaster : AxiLiteWriteMasterType :=  AXI_LITE_WRITE_MASTER_INIT_C;
   signal axilWriteSlave  : AxiLiteWriteSlaveType  :=  AXI_LITE_WRITE_SLAVE_INIT_C;
   signal axilReadMaster  : AxiLiteReadMasterType  :=  AXI_LITE_READ_MASTER_INIT_C;
   signal axilReadSlave   : AxiLiteReadSlaveType   :=  AXI_LITE_READ_SLAVE_INIT_C;


   signal reloadInMaster  : AxiStreamMasterType    :=  AXI_STREAM_MASTER_INIT_C;        
   signal reloadInSlave   : AxiStreamSlaveType     :=  AXI_STREAM_SLAVE_INIT_C;

   signal configInMaster  : AxiStreamMasterType    :=  AXI_STREAM_MASTER_INIT_C;        
   signal configInSlave   : AxiStreamSlaveType     :=  AXI_STREAM_SLAVE_INIT_C;

   signal axiClk        : sl;
   signal axiRst        : sl;

   signal axilClk       : sl;
   signal axilRst       : sl;

   file file_RESULTS    : text;
   file fir_coef_file   : text;

begin

   appOutSlave.tReady <= '1';
   appInSlave.tReady  <= '1';
   axilClk            <= axiClk;
   axilRst            <= axiRst;
   

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

   -----------------
   -- Time Tool Core
   -----------------
   U_TimeToolCore : entity work.TimeToolCore
      generic map (
         TPD_G           => TPD_G,
         AXI_BASE_ADDR_G => AXI_BASE_ADDR_G)
      port map (
         -- System Clock and Reset
         axilClk         => axiClk,
         axilRst         => axiRst,
         -- Trigger Event streams (axilClk domain)
         --trigMaster      => trigMaster,  -- takes too long too simulate
         trigMaster        => AXI_STREAM_MASTER_INIT_C,
         --trigSlave       => trigSlave,   -- takes too long too simulate
         -- DMA Interface (sysClk domain)
         dataInMaster     => appInMaster,
         --dataInSlave    => appInSlave,
         eventMaster      => appOutMaster,
         eventSlave       => appOutSlave,
         -- AXI-Lite Interface (sysClk domain)
         axilReadMaster  => axilReadMaster,
         axilReadSlave   => axilReadSlave,
         axilWriteMaster => axilWriteMaster,
         axilWriteSlave  => axilWriteSlave);


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

      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C2_0004", x"0", true);  --prescaler
      --axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C1_0004", x"3", true);  --fex placehold add value commented out. will be deleted once fex_placeholder is replaced by actual fex
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C0_0fd0", x"1", true);  --event builder bypass
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C3_0004", x"0", true);  --time tool core by pass. this by passes everything. used for debugging determining whether DSP block is source of problem.
   

      --actual fex axi write data
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C1_1004", x"0", true);  --event code filter that is being simulated by a prescaler
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C1_3004", x"3", true);  --iir filter
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C1_0fd0", x"0", true);  --event builder inside fex. only being used for batching purposes, not for time stamping.
                                                                                               --don't bypass anything since no tpm is being used.

      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C1_2000", x"7f7f7f7f", true); --axi lite to FIR coeffcients
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C1_2004", x"7f7f7f7f", true); --axi lite to FIR coeffcients
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C1_2008", x"7f7f7f7f", true); --axi lite to FIR coeffcients
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C1_200c", x"7f7f7f7f", true); --axi lite to FIR coeffcients
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C1_2010", x"81818181", true); --axi lite to FIR coeffcients
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C1_2014", x"81818181", true); --axi lite to FIR coeffcients
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C1_2018", x"81818181", true); --axi lite to FIR coeffcients
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C1_201c", x"81818181", true); --axi lite to FIR coeffcients

      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C1_2020", x"1", true); --axi lite to FIR coeffcients
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C1_6004", x"0", true);  --prescaler
      axiLiteBusSimWrite (axiClk, axilWriteMaster, axilWriteSlave, x"00C1_7004", x"0", true);  --subtraction toggle

       

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
