------------------------------------------------------------------------------
-- File       : FileToAxiStream.vhd
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

entity FileToAxiStream is
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
      dataOutMaster   : out AxiStreamMasterType  := AXI_STREAM_MASTER_INIT_C;
      dataOutSlave    : in  AxiStreamSlaveType   := AXI_STREAM_SLAVE_INIT_C );
end FileToAxiStream;

architecture mapping of FileToAxiStream is

   constant TEST_INPUT_FILE_NAME : string := TEST_FILE_PATH & "/sim_input_data.dat";
   constant c1_WIDTH      : natural := 8;
   constant c2_WIDTH      : natural := 2;
   signal r_ADD_TERM1     : std_logic_vector(BITS_PER_TRANSFER-1 downto 0) := (others => '0');
   signal r_ADD_TERM2     : sl := '0';

   constant INT_CONFIG_C  : AxiStreamConfigType := ssiAxiStreamConfig(dataBytes => 16, tDestBits => 0);
   constant PGP2BTXIN_LEN : integer             := 19;
   constant FRAME_PERIOD  : slv(31 downto 0)    := (12=>'1',others => '0');

   file file_VECTORS : text;


   type StateType is (
      IDLE_S,
      MOVE_S);

   type RegType is record
      master           : AxiStreamMasterType;
      slave            : AxiStreamSlaveType;

      v_ADD_TERM1      : std_logic_vector(BITS_PER_TRANSFER-1 downto 0);
      v_ADD_TERM2      : sl;
      v_SPACE          : character;

      state            : StateType;
      counter          : slv(31 downto 0);
   end record RegType;
   
   ---------------------------------------
   -------record intitial value-----------
   ---------------------------------------
   constant REG_INIT_C : RegType := (
      master           => AXI_STREAM_MASTER_INIT_C,
      slave            => AXI_STREAM_SLAVE_INIT_C,

      v_ADD_TERM1      => (others => '0'),
      v_ADD_TERM2      => '0',
      v_SPACE          => ' ',

      state          => IDLE_S,
      counter => (others => '0'));



   signal fileClk                : sl := '0';
   signal fileRst                : sl := '0';
   signal fileClk_subharmonics   : slv(31 downto 0)      :=    (others => '0')  ;


   signal r                      : RegType := REG_INIT_C;
   signal rin                    : RegType := REG_INIT_C;

   signal outCtrl                : AxiStreamCtrlType  :=    AXI_STREAM_CTRL_INIT_C;

begin

   --------------------------
   --read file
   --------------------------
   file_open(file_VECTORS, TEST_INPUT_FILE_NAME,  read_mode);

   --------------------
   -- Clocks and Resets
   --------------------
   U_axilClk_2 : entity work.ClkRst
      generic map (
         CLK_PERIOD_G      => CLK_PERIOD_G,
         RST_START_DELAY_G => 1  ns,
         RST_HOLD_TIME_G   => 50 ns)
      port map (
         clkP => fileClk,
         rst  => fileRst);

   ---------------------------------
   -- Application
   ---------------------------------
   sim : process is
      variable v           : RegType;
      variable v_ILINE     : line;
      variable v_ADD_TERM1 : std_logic_vector(BITS_PER_TRANSFER-1 downto 0);
      variable v_ADD_TERM2 : sl := '0';
      variable v_SPACE     : character;

   begin
      

       loop
            wait until rising_edge(fileClk);
                  v := r;
                  v.master.tValid := '0';



                  if  v.counter >= FRAME_PERIOD then
                        v.master.tValid := '1';

                
                        readline(file_VECTORS, v_ILINE);
                        read(v_ILINE, v.v_ADD_TERM1);
                        read(v_ILINE, v.v_SPACE);           -- read in the space character
                        read(v_ILINE, v.v_ADD_TERM2);

                        v.master.tData(BITS_PER_TRANSFER-1 downto 0)        := v.v_ADD_TERM1;
                        v.master.tLast                                      := v.v_ADD_TERM2;
                        v.master.tKeep(BITS_PER_TRANSFER/8-1 downto 0)      := (others=>'1');

                        if v.master.tLast ='1' then       
                              v.counter := (others => '0');

                        end if;

                  else
                        v.counter      := v.counter+1;
                        v.master.tLast := '0';
                        
                  end if;

                  


                  -------------
                  -- Reset
                  -------------
                  if (fileRst = '1') then
                     v := REG_INIT_C;
                  end if;

                  -- Register the variable for next clock cycle
                  --rin      <= v;
                  r        <= v after TPD_G;
      end loop;
      
   end process sim;

--   seq : process (fileClk) is
--   begin
--      if (rising_edge(fileClk)) then
--         r <= rin after TPD_G;
--         fileClk_subharmonics <= fileClk_subharmonics+'1' after TPD_G;
--     end if;
--   end process seq;

   ---------------------------------
   -- Output FIFO
   ---------------------------------
   U_OutFifo : entity work.AxiStreamFifoV2
      generic map (
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => false,
         --GEN_SYNC_FIFO_G     => true,
         FIFO_ADDR_WIDTH_G   => 11,
         FIFO_PAUSE_THRESH_G => 2036,
         SLAVE_AXI_CONFIG_G  => INT_CONFIG_C,
         MASTER_AXI_CONFIG_G => DMA_AXIS_CONFIG_G)
      port map (
         sAxisClk    => fileClk,
         sAxisRst    => fileRst,
         sAxisMaster => r.Master,
         sAxisCtrl   => outCtrl,
         mAxisClk    => sysClk,
         mAxisRst    => sysRst,
         mAxisMaster => dataOutMaster,
         mAxisSlave  => dataOutSlave);

end mapping;
