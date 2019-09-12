------------------------------------------------------------------------------
-- File       : AxiStreamToFile.vhd
-- Company    : SLAC National Accelerator Laboratory

-- Last update: 2019-03-22
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
use ieee.numeric_std.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.AxiPkg.all;
use work.SsiPkg.all;
use work.AxiPciePkg.all;
use work.TimingPkg.all;
use work.Pgp2bPkg.all;

use STD.textio.all;
use ieee.std_logic_textio.all;
use work.TestingPkg.all;

library unisim;
use unisim.vcomponents.all;

-------------------------------------------------------------------------------
-- This file performs the the prescaling, or the amount of raw data which is stored
-------------------------------------------------------------------------------

entity AxiStreamToFile is
   generic (
      TPD_G              : time                := 1 ns;
      DMA_AXIS_CONFIG_G  : AxiStreamConfigType := ssiAxiStreamConfig(16, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 8, 2);
      DEBUG_G            : boolean             := true;
      BYTE_SIZE_C        : positive            := 1;
      BITS_PER_TRANSFER  : natural             := 128;
      CLK_PERIOD_G       : time                := 23 ns);
   port (
      -- System Interface
      sysClk          : in  sl;
      sysRst          : in  sl;
      -- DMA Interfaces  (sysClk domain)
      dataInMaster        : in    AxiStreamMasterType;
      dataInSlave         : out   AxiStreamSlaveType :=    AXI_STREAM_SLAVE_INIT_C );
end AxiStreamToFile;

architecture mapping of AxiStreamToFile is

   constant TEST_OUTPUT_FILE_NAME : string              := TEST_FILE_PATH & "/output_results.dat";
   constant PSEUDO_RAND_COEF      : slv(31 downto 0)    := (0=>'1',1=>'1',others=>'0');
   constant INT_CONFIG_C          : AxiStreamConfigType := ssiAxiStreamConfig(dataBytes => 16, tDestBits => 0);
   constant c_WIDTH               : natural := 128;




---------------------------------------
-------record intitial value-----------
---------------------------------------

   signal inMaster                 : AxiStreamMasterType   :=    AXI_STREAM_MASTER_INIT_C;
   signal inSlave                  : AxiStreamSlaveType    :=    AXI_STREAM_SLAVE_INIT_C;  
   signal outCtrl                  : AxiStreamCtrlType     :=    AXI_STREAM_CTRL_INIT_C;

   signal fileClk                  : sl := '0';
   signal fileRst                  : sl := '0';

   signal clocked_tvalid           : sl := '1';

   signal pseudo_random            : slv(31 downto 0)      :=    (others => '0')  ;

   file file_RESULTS : text;

begin
   --------------------------
   --write file
   --------------------------
   clocked_tvalid <= fileClk and inMaster.tValid;

   file_open(file_RESULTS, TEST_OUTPUT_FILE_NAME, write_mode);

   --------------------
   -- Clocks and Resets
   --------------------
   U_axilClk_2 : entity work.ClkRst
      generic map (
         CLK_PERIOD_G      => CLK_PERIOD_G,
         RST_START_DELAY_G => 0  ns,
         RST_HOLD_TIME_G   => 1000 ns)
      port map (
         clkP => fileClk,
         rst  => fileRst);

   ---------------------------------
   -- Input FIFO
   ---------------------------------
   U_InFifo: entity work.AxiStreamFifoV2
      generic map (
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => true,
         --GEN_SYNC_FIFO_G     => true,
         FIFO_ADDR_WIDTH_G   => 11,
         FIFO_PAUSE_THRESH_G => 2036,
         SLAVE_AXI_CONFIG_G  => DMA_AXIS_CONFIG_G,
         MASTER_AXI_CONFIG_G => INT_CONFIG_C)
      port map (
         sAxisClk    => sysClk,
         sAxisRst    => sysRst,
         sAxisMaster => dataInMaster,
         sAxisSlave  => dataInSlave,
         mAxisClk    => fileClk,
         mAxisRst    => fileRst,
         mAxisMaster => inMaster,
         mAxisSlave  => inSlave);

   save_to_file : process is
      variable v_OLINE              : line; 
      constant c_WIDTH              : natural := 128;
      constant test_data_to_file    : slv(c_WIDTH -1 downto 0) := (others => '0');

   begin
      
      inSlave.tReady <= '1';
      file_open(file_RESULTS, TEST_OUTPUT_FILE_NAME, write_mode);

      loop

            if inMaster.tValid ='1' then

                  write(v_OLINE, inMaster.tData(c_WIDTH-1 downto 0), right, c_WIDTH);
                  writeline(file_RESULTS, v_OLINE);

            end if;

            wait until rising_edge(fileClk);

      end loop;
      
      file_close(file_RESULTS);

   end process save_to_file;


end mapping;
