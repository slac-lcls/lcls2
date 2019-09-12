------------------------------------------------------------------------------
-- File       : AxiStreamSimToFileTwoProcess.vhd
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

entity AxiStreamSimToFileTwoProcess is
   generic (
      TPD_G              : time                := 1 ns;
      DMA_AXIS_CONFIG_G  : AxiStreamConfigType := ssiAxiStreamConfig(16, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 8, 2);
      DEBUG_G            : boolean             := true;
      BYTE_SIZE_C        : positive            := 1;
      BITS_PER_TRANSFER  : natural             := 128);
   port (
      -- System Interface
      sysClk          : in  sl;
      sysRst          : in  sl;
      -- DMA Interfaces  (sysClk domain)
      dataInMaster        : in    AxiStreamMasterType;
      dataInSlave         : out   AxiStreamSlaveType);
end AxiStreamSimToFileTwoProcess;

architecture mapping of AxiStreamSimToFileTwoProcess is

   constant TEST_OUTPUT_FILE_NAME : string              := TEST_FILE_PATH & "/output_results.dat";
   constant PSEUDO_RAND_COEF      : slv(31 downto 0)    := (0=>'1',1=>'1',others=>'0');
   constant INT_CONFIG_C          : AxiStreamConfigType := ssiAxiStreamConfig(dataBytes => 16, tDestBits => 0);
   constant c_WIDTH               : natural := 128;


   type StateType is (
      IDLE_S,
      MOVE_S);

   type RegType is record
      master         : AxiStreamMasterType;
      slave          : AxiStreamSlaveType;
      state          : StateType;
      pseudo_random  : slv(31 downto 0);
      validate_state : slv(31 downto 0);
   end record RegType;

   constant REG_INIT_C : RegType := (
      master         => AXI_STREAM_MASTER_INIT_C,
      slave          => AXI_STREAM_SLAVE_INIT_C,
      state          => IDLE_S,
      pseudo_random  => (others => '0'),
      validate_state => (others => '0'));

---------------------------------------
-------record intitial value-----------
---------------------------------------


   signal r                        : RegType := REG_INIT_C;
   signal rin                      : RegType;

   signal inMaster                 : AxiStreamMasterType   :=    AXI_STREAM_MASTER_INIT_C;
   signal inSlave                  : AxiStreamSlaveType    :=    AXI_STREAM_SLAVE_INIT_C;  
   signal outCtrl                  : AxiStreamCtrlType     :=    AXI_STREAM_CTRL_INIT_C;

   signal fileClk                  : sl := '0';
   signal fileRst                  : sl := '0';


   signal pseudo_random            : slv(31 downto 0)      :=    (others => '0')  ;

   file file_RESULTS : text;

begin
   --------------------------
   --write file
   --------------------------

   file_open(file_RESULTS, TEST_OUTPUT_FILE_NAME, write_mode);

   --------------------
   -- Clocks and Resets
   --------------------
   U_axilClk_2 : entity work.ClkRst
      generic map (
         CLK_PERIOD_G      => 23 ns,
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
         GEN_SYNC_FIFO_G     => true,
         FIFO_ADDR_WIDTH_G   => 9,
         FIFO_PAUSE_THRESH_G => 500,
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

   ---------------------------------
   -- Application
   ---------------------------------
   comb : process (r,fileRst,inMaster) is
      variable v           : RegType;
      variable v_OLINE     : line;

   begin
      
      --------------------------
      --latch previous state
      --------------------------
      v := r;


      --------------------------
      --setting slave state and loading data
      --------------------------
      v.slave.tReady  :=  '1';
      --v.slave.tReady  :=  pseudo_random(6);
      v.Master        :=  inMaster;

      case r.state is

         when IDLE_S =>
            ------------------------------
           if v.slave.tReady = '1' and v.Master.tValid ='1' then

              v.state := MOVE_S;

           else
              v.state := IDLE_S;

           end if;

           v.slave.tReady  :=  '0';

         when MOVE_S =>
           if v.slave.tReady = '1' and v.Master.tValid ='1' then
             --write(v_OLINE, v.master.tData(c_WIDTH-1 downto 0), right, c_WIDTH);
             write(v_OLINE, v.validate_state);
             writeline(file_RESULTS, v_OLINE);

             v.validate_state := v.validate_state+'1';

            --if v.master.tLast ='1' then
            --      v.validate_state := v.validate_state - 127;            
            --end if;

           else
              v.state := IDLE_S;
                   
           end if;

      end case;



      -------------
      -- Reset
      -------------
      if (fileRst = '1') then
         v := REG_INIT_C;
      end if;

      -- Register the variable for next clock cycle
      rin <= v;

      -- Outputs 
      inSlave        <= v.slave;

   end process comb;

   seq : process (fileClk) is
   begin
      if (rising_edge(fileClk)) then
         r <= rin after TPD_G;
      end if;
   end process seq;


end mapping;
